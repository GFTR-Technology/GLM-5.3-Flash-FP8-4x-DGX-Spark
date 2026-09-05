# TP=6 — running GLM-5.3-Flash FP8 on six Sparks

The published recipe is TP=4. This page is what it takes to run a tensor
parallel size that does **not** divide the model, TP=6 in particular.

Short version: it is not only the attention heads. Four dimensions get padded,
one is handled by hosting a whole copy per rank, and one of the padded four (the
vocab) has nothing to do with attention at all.

## What breaks at TP=6

Values are from the shipped checkpoint — `config.json` and the safetensors
headers of `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`. The "sharded?" column is
from `scripts/glm53-tp-probe.sh` run against the real image, not from reading
the config: a dimension only has to divide if this build actually splits it.

| Dimension | Config key | Value | ÷6 | Sharded? | Fix | Per rank |
|---|---|---|---|---|---|---|
| MLA heads (11 layers + MTP) | `num_attention_heads` | 64 | ✗ | yes | pad **66** | 11 |
| KDA heads (34 layers) | `linear_attn_config.num_heads` | 64 | ✗ | yes | pad **66** | 11 |
| MoE expert intermediate | `moe_intermediate_size` | 2048 | ✗ | yes | pad **2304** | 384 |
| Vocabulary | `vocab_size` | 154880 | ✗ | yes | pad **154944** | 25824 |
| Sparse indexer heads | `index_n_heads` | 32 | ✗ | **no** (`disable_tp`) | nothing | 32 |
| Vision tower | `vision_config.*` | 1024 / 16 / 4096 | ✗ | opt-out | encoder DP | whole |
| Dense MLP (first 3 layers) | `intermediate_size` | 12288 | ✓ | yes | — | 2048 |
| Routed experts | `n_routed_experts` | 288 | ✓ | yes | — | — |

The padded extents keep blockwise-FP8 alignment: MLA head dim is 256 and KDA's
is 128, so head padding is automatically a multiple of 128; the MoE intermediate
rounds to a multiple of `tp * 128` (2304 / 6 = 384 = 3 × 128).

### Two things the probe settled

Both were open when this shim was written and neither is answerable from the
config; the model code in this image is a custom drop, so the probe is the only
authority. Re-run it after any image bump — these are properties of the *build*,
not of the checkpoint.

**The sparse indexer is not TP-sharded, so it is not padded.** Its only parallel
module is built with `disable_tp=True`:

```
models/glm5next/nvidia/attention.py:259  self.wk_weights_proj = MergedColumnParallelLinear(
                                     ...  disable_tp=True)
```

All 32 heads therefore live whole on every rank and never reach `divide()`.
The probe's `[tp-divide]` sweep confirms this from the other side: five call
sites, all in `kda.py` and `multimodal.py`, none in the indexer. Padding it
anyway would be worse than pointless — the config would advertise 36 heads
against 32 rows of unpadded weight and loading would fail a shape check. Hence
`indexer` is *not* in `DEFAULT_GROUPS` and `make_overlay.py` leaves
`index_n_heads` at 32. (The implementation already carries its own head-count
fixup at `attention.py:394` for checkpoints that ship `index_n_heads=16`.)

**The vision tower supports encoder DP but never declares it — and the gate is
real.** The tower honours the mode correctly: `multimodal.py:141` sets
`tp_size = 1` under DP and seven vision linears are built
`disable_tp=use_data_parallel`. But `use_data_parallel` comes from
`is_vit_use_data_parallel()` (`multimodal.py:54,101,140,310,375`), a helper that
reads the *resolved* config — and no file in the implementation contains
`supports_encoder_tp_data`. vLLM core does gate on that attribute:

```
model_executor/models/interfaces.py:152  supports_encoder_tp_data: ClassVar[bool] = False
model_executor/models/interfaces.py:545  return getattr(model, "supports_encoder_tp_data", False)
config/model.py:764                      mm_encoder_tp_mode == "data"
config/model.py:771                      mm_encoder_tp_mode = "weights"
config/multimodal.py:179                 mm_encoder_tp_mode: MMEncoderTPMode = "weights"
```

So without the marker, `--mm-encoder-tp-mode data` is resolved back to
`"weights"` in `ModelConfig`, `is_vit_use_data_parallel()` returns False in every
worker, and the 16-head tower shards into `divide(16, 6)` at
`multimodal.py:150`. The shim sets the marker on the
`*ForConditionalGeneration` class (`patch_encoder_dp`). This is the one piece of
the shim that is not padding at all, and it is required.

**Where the marker has to be set is not where you see it.** The downgrade is
decided in `ModelConfig.__post_init__`, from a registry inspection that vLLM
runs in a **subprocess** (`python -m vllm.model_executor.models.registry`) whose
output is captured and discarded unless it fails. The shim reaches that
subprocess because it inherits `PYTHONPATH=/opt/glm53` and `GLM53_TP_PAD` from
the container environment, so `site` loads `sitecustomize.py` there too and the
import hook fires when the subprocess imports the model module. You will not see
the shim's `supports_encoder_tp_data=True` line from that process — only from
the ranks. The signal that it *failed* is vLLM's own fallback warning about
`--mm-encoder-tp-mode` during start-up.

### `head_dim` is pinned when the MLA group is on

`Glm5NextTextConfig` derives `head_dim` from `hidden_size // num_attention_heads`
when the key is absent, so bumping 64 → 66 would silently drag head_dim 64 → 62.
This checkpoint ships `head_dim` explicitly (as `0` — MLA uses `qk_nope_head_dim`
and `v_head_dim` instead), so nothing moves today. `make_overlay.py` writes the
pre-pad value anyway if a future release drops the key.

### The vocab is a separate blocker

vLLM's `pad_vocab_size()` rounds to 64 and ignores `tp_size`
(`vllm/model_executor/layers/vocab_parallel_embedding.py`). 154880 is already a
multiple of 64, so nothing is padded, and `vocab_range_from_global_vocab_size()`
then calls `divide(154880, 6)` and asserts. The shim rebinds `pad_vocab_size` to
round to `lcm(64, tp)` = 192 → 154944.

`vocab_size` in the config is deliberately **not** touched: the lm_head and
embedding on disk still have 154880 rows, and vLLM's own vocab padding fills the
remainder.

## The padding rule

One rule, one exception.

* **Column-parallel weights** (output dim is sharded) — padded slots get a
  *cyclic copy* of the real leading slots.
* **Row-parallel weights** (input dim is sharded) — padded columns are **zeroed**.

The output-side zeros are what make a padded head or expert slot contribute
exactly nothing. The input-side replication is deliberate and load-bearing: if
the input side were zeroed too, the activations feeding `o_proj` / `down_proj`
would contain entire all-zero 128-element blocks, and DeepGEMM's dynamic
per-block FP8 quantisation takes `amax` over each block — `amax == 0` gives
0/0 → NaN. Copying real weights keeps every activation block well scaled while
the output-side zeros still cancel the contribution.

The replicate fill cycles from index 0 and every affected tensor is head-major,
so head `H+i` consistently receives head `i`'s weights across `q_proj`,
`A_log`, `dt_bias`, `b_proj` and the rest. Padded heads are coherent duplicates
of real heads, not a mix.

**The exception:** `indexer.weights_proj` has one row per indexer head and is
the *combining* weight for the per-head index logits. It is semantically an
output-side tensor, so padded heads are zeroed there — otherwise they would
perturb the top-k token selection. (Any uniform positive rescale of the index
logits is harmless because top-k is scale invariant; a *contribution* is not.)
This rule is kept and tested but does not fire by default, because the indexer
turned out not to be sharded; it exists for a build that starts sharding it.

FP8 `weight_scale_inv` companions follow their weight: replicated where the
weight is replicated, filled with **1.0** where it is zeroed. The quantised
value is 0 there, so the scale only has to be finite and non-zero.

## Cost

Weight bytes, computed from the real tensor shapes:

| | TP=4, unpadded | TP=6, padded |
|---|---|---|
| Total weights | ~304 GiB | ~341 GiB (+12% — the MoE is ~95% of the model) |
| Per rank | ~76 GiB | ~57 GiB |
| Left for KV + activations at `gmu=0.85` | ~27 GiB | ~47 GiB |

Padding costs ~6 GiB per rank; TP=6 still frees ~19 GiB per rank over TP=4.

**KV does not scale with TP the way weights do.** MLA keeps its compressed KV
latent replicated on every rank (`decode_context_parallel_size=1`), so a wider
TP buys KV room only through the weight memory it frees. Expect roughly
1.7–1.8× the KV pool, not 1.5× linear — confirm against the engine's own
"GPU KV cache size" print rather than trusting this paragraph.

**The lane table in the README (15 / 5 / 3 seqs) was measured at TP=4.** Re-bench
with `scripts/bench_lanes.py` before trusting those numbers on six boxes.

## Running it

```bash
# cluster.env: five worker IPs instead of three. TP comes from the node count.
WORKER_IPS="10.0.0.12 10.0.0.13 10.0.0.14 10.0.0.15 10.0.0.16"
```

1. **Probe the image first.** The glm5next implementation in this image is a
   custom drop, and the defaults above were chosen from one probe run against
   one image. Re-confirm after any image bump.

   ```bash
   ./scripts/glm53-tp-probe.sh          # read-only, safe while an engine is live
   ```

   Read four things out of the report:
   * `[tp-divide]` — every `divide()` call site in the implementation. All of
     them must be covered by a pad group. As of the probed image there are five,
     in `kda.py` and `multimodal.py`, plus three bare asserts; a sixth would mean
     a dimension this shim does not know about.
   * **indexer** — if `wk_weights_proj` is no longer `disable_tp=True`, it *is*
     sharded and you need `GLM53_TP_PAD_GROUPS=mla,kda,indexer,moe,vocab`.
   * **vision tower** — the `encoder-DP gate in vLLM core` section should still
     show `interfaces.py: supports_encoder_tp_data` and the
     `config/model.py: mm_encoder_tp_mode = "weights"` downgrade. As long as it
     does, `patch_encoder_dp` is load-bearing and `--mm-encoder-tp-mode data`
     alone is not enough.
   * do `DefaultModelLoader._get_weights_iterator` and `pad_vocab_size` exist?
     (`vllm.model_executor.model_loader.loader` reported absent is fine — it is
     the pre-0.9 fallback seam.)

2. **Boot.** Nothing else changes; the launcher notices that TP does not divide
   and wires the shim in by itself.

   ```bash
   GLM53_TP_PAD_DEBUG=1 GLM53_LANE=200k ./scripts/glm53-serve.sh
   ./scripts/glm53-serve.sh logs
   ```

3. **Check the padding log against this table.** Under the default groups the
   shim rewrites exactly 75030 of the checkpoint's 76108 tensors:

   | rule | count | | rule | count |
   |---|---|---|---|---|
   | `mla q_b` | 12 | | `kda qkv_proj` | 102 |
   | `mla q_b scale` | 12 | | `kda conv1d` | 102 |
   | `mla kv_b` | 12 | | `kda gate b_proj` | 68 |
   | `mla o_proj` | 12 | | `kda beta` | 34 |
   | `mla o_proj scale` | 12 | | `kda A_log` | 34 |
   | `moe gate/up` | 24854 | | `kda dt_bias` | 34 |
   | `moe gate/up scale` | 24854 | | `kda o_proj` | 34 |
   | `moe down` | 12427 | | `moe down scale` | 12427 |

   12 = 11 MLA layers + the MTP layer. 43 = 42 sparse-MLP layers + the MTP
   layer, × 289 (288 routed + 1 shared) for the MoE counts. Turning the indexer
   group on adds 24 more (`indexer wq_b` 12, `indexer weights_proj` 12) for
   75054 total.

4. **Verify the output, not just the boot.** Same prompt, `temperature=0`, run
   once at TP=4 and once at TP=6 and compare token by token. If the padding is
   right the first few dozen tokens are identical; FP8 reduction order differs
   across shard counts, so long completions may diverge later.

## Knobs

| Variable | Effect |
|---|---|
| `GLM53_TP_PAD_DEBUG=1` | log every rewritten tensor (name, shapes, fill mode) |
| `GLM53_TP_PAD_GROUPS=mla,kda,moe,vocab` | the default — indexer padding off |
| `GLM53_TP_PAD_GROUPS=mla,kda,indexer,moe,vocab` | add the indexer, if a build starts sharding it |
| `GLM53_TP_PAD_STRICT=0` | warn instead of aborting on an unrecognised extent |
| `GLM53_MM_ENCODER_TP_MODE=` | drop `--mm-encoder-tp-mode` (older vLLM has no such flag) |

Switching a group off is an explicit claim that the dimension is not TP-sharded
in this build, so the plan validator stops demanding divisibility for it. That
is exactly why `indexer` is off by default. The launcher echoes the active set
on its banner (`groups  kda,mla,moe,vocab`) so a stale override is visible
before the containers start.

## How it is wired

```
scripts/glm53-node-launch.sh      detects 64 % TP != 0, mounts the shim,
                                  sets PYTHONPATH/GLM53_TP_PAD, serves /model-tp6
scripts/glm53-container-entrypoint.sh
                                  builds /model-tp6: symlinks to every real file
                                  plus one rewritten config.json (zero copy)
patches/glm53_tp_pad/sitecustomize.py
                                  post-import hook; picked up by every rank
                                  because vLLM's mp executor spawns fresh
                                  interpreters and site always imports it
patches/glm53_tp_pad/glm53_tp_pad.py
                                  the plan, the rules, the tensor surgery,
                                  and the two vLLM patches
```

Two seams into vLLM plus one class-attribute marker, all narrow:

* `DefaultModelLoader._get_weights_iterator` — wrapped so tensors arrive padded
  *before* any per-param `weight_loader` narrows them. vLLM's column/row
  parallel sharding needs no changes at all.
* `pad_vocab_size` — rebound to be tp-aware.
* `supports_encoder_tp_data = True` set on the model class, so
  `--mm-encoder-tp-mode data` is not downgraded to weight sharding.

Everything else rides on the config overlay.

## Why not the alternatives

* **Repack the checkpoint offline.** Robust and vLLM-version-proof, but it costs
  ~340 GiB of extra disk per padded TP and a multi-hour conversion.
* **Expert parallel instead of padding the MoE.** 288 / 6 = 48 divides cleanly,
  so `--enable-expert-parallel` would avoid the +36 GiB entirely. It is the
  better memory trade if it works; `docs/DESIGN.md` records DeepEP failing to
  import in this image, so it needs vLLM's built-in all-to-all. Worth revisiting.
* **TP=2 × PP=3 on six nodes.** Every dimension divides by 2, so this needs no
  model surgery at all and lands at the same ~51 GiB per rank. Different latency
  profile (pipeline stages instead of an all-reduce per layer) and not benched
  here, but it is the zero-risk way to use six boxes.

## Tests

```bash
python3 tests/test_tp_pad.py                    # or: pytest tests/test_tp_pad.py
```

Runs without torch (plan maths, rule matching against real checkpoint names,
injection hygiene) and with torch adds the numerical equivalence tests: MLA
attention, MoE SwiGLU and the KDA gated recurrence all produce bit-comparable
outputs padded vs unpadded, indexer top-k selects an identical token set, and
the rejected input-side-zeroing scheme is shown to produce the dead
quantisation block described above.

Best run inside the image, against the exact torch build:

```bash
docker run --rm -v "$PWD:/repo:ro" --entrypoint python3 \
  glm53-flash:sm121-v8 /repo/tests/test_tp_pad.py
```
