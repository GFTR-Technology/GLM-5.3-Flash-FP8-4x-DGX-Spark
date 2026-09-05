# TP=6 — running GLM-5.3-Flash FP8 on six Sparks

The published recipe is TP=4. This page is what it takes to run a tensor
parallel size that does **not** divide the model, TP=6 in particular.

Short version: it is not only the attention heads. Six independent dimensions
break, and one of them (the vocab) has nothing to do with attention at all.

## What breaks at TP=6

Values are from the shipped checkpoint — `config.json` and the safetensors
headers of `dealignai/GLM-5.3-Flash-UNCENSORED-FP8`.

| Dimension | Config key | Value | ÷6 | Padded to | Per rank |
|---|---|---|---|---|---|
| MLA heads (11 layers + MTP) | `num_attention_heads` | 64 | ✗ | **66** | 11 |
| Sparse indexer heads | `index_n_heads` | 32 | ✗ | **36** | 6 |
| KDA heads (34 layers) | `linear_attn_config.num_heads` | 64 | ✗ | **66** | 11 |
| MoE expert intermediate | `moe_intermediate_size` | 2048 | ✗ | **2304** | 384 |
| Vocabulary | `vocab_size` | 154880 | ✗ | **154944** | 25824 |
| Vision tower | `vision_config.*` | 1024 / 16 / 4096 | ✗ | encoder DP | — |
| Dense MLP (first 3 layers) | `intermediate_size` | 12288 | ✓ | — | 2048 |
| Routed experts | `n_routed_experts` | 288 | ✓ | — | — |

The padded extents keep blockwise-FP8 alignment: MLA head dim is 256 and KDA's
is 128, so head padding is automatically a multiple of 128; the MoE intermediate
rounds to a multiple of `tp * 128` (2304 / 6 = 384 = 3 × 128).

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
   custom drop; the shim's attachment points need confirming once.

   ```bash
   ./scripts/glm53-tp-probe.sh          # read-only, safe while an engine is live
   ```

   Read three things out of the report:
   * does the **indexer** use parallel linears? If it is replicated, turn its
     padding off (see below) — padding a replicated tensor breaks shape checks.
   * does the **vision tower** use parallel linears, and does the model declare
     `supports_encoder_tp_data`? If it uses parallel linears and does *not*
     support encoder DP, `--mm-encoder-tp-mode data` silently falls back to
     weight sharding and TP=6 will still fail in the tower.
   * do `DefaultModelLoader._get_weights_iterator` and `pad_vocab_size` exist?

2. **Boot.** Nothing else changes; the launcher notices that TP does not divide
   and wires the shim in by itself.

   ```bash
   GLM53_TP_PAD_DEBUG=1 GLM53_LANE=200k ./scripts/glm53-serve.sh
   ./scripts/glm53-serve.sh logs
   ```

3. **Check the padding log against this table.** With all groups on, the shim
   rewrites exactly 75054 of the checkpoint's 76108 tensors:

   | rule | count | | rule | count |
   |---|---|---|---|---|
   | `mla q_b` | 12 | | `kda qkv_proj` | 102 |
   | `mla q_b scale` | 12 | | `kda conv1d` | 102 |
   | `mla kv_b` | 12 | | `kda gate b_proj` | 68 |
   | `mla o_proj` | 12 | | `kda beta` | 34 |
   | `mla o_proj scale` | 12 | | `kda A_log` | 34 |
   | `indexer wq_b` | 12 | | `kda dt_bias` | 34 |
   | `indexer weights_proj` | 12 | | `kda o_proj` | 34 |
   | `moe gate/up` | 24854 | | `moe down` | 12427 |
   | `moe gate/up scale` | 24854 | | `moe down scale` | 12427 |

   12 = 11 MLA layers + the MTP layer. 43 = 42 sparse-MLP layers + the MTP
   layer, × 289 (288 routed + 1 shared) for the MoE counts.

4. **Verify the output, not just the boot.** Same prompt, `temperature=0`, run
   once at TP=4 and once at TP=6 and compare token by token. If the padding is
   right the first few dozen tokens are identical; FP8 reduction order differs
   across shard counts, so long completions may diverge later.

## Knobs

| Variable | Effect |
|---|---|
| `GLM53_TP_PAD_DEBUG=1` | log every rewritten tensor (name, shapes, fill mode) |
| `GLM53_TP_PAD_GROUPS=mla,kda,moe,vocab` | pad only these; here, indexer padding off |
| `GLM53_TP_PAD_STRICT=0` | warn instead of aborting on an unrecognised extent |
| `GLM53_MM_ENCODER_TP_MODE=` | drop `--mm-encoder-tp-mode` (older vLLM has no such flag) |

Switching a group off is an explicit claim that the dimension is not TP-sharded
in this build, so the plan validator stops demanding divisibility for it.

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

Two seams into vLLM, both narrow:

* `DefaultModelLoader._get_weights_iterator` — wrapped so tensors arrive padded
  *before* any per-param `weight_loader` narrows them. vLLM's column/row
  parallel sharding needs no changes at all.
* `pad_vocab_size` — rebound to be tp-aware.

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
