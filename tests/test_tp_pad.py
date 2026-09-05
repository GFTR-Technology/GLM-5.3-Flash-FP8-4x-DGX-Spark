"""Tests for the TP head/dim padding shim.

Runs under pytest, or standalone:

    python3 tests/test_tp_pad.py

The torch-backed tests skip themselves when torch is missing, so the plan math,
the rule table and the real-checkpoint name matching can still be checked on a
box without a deep learning stack.

The numerical tests are the point of this file. Shape tests only prove vLLM will
accept the tensors; the equivalence tests prove the padded heads/experts
contribute exactly nothing, which is the actual claim the shim makes.
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "patches", "glm53_tp_pad"),
)

import glm53_tp_pad as pad  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - exercised on torch-less boxes
    torch = None

SHIM_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "patches", "glm53_tp_pad"
)


def requires_torch():
    if torch is None:
        raise SkipTest("torch not installed")


class SkipTest(Exception):
    pass


# ---------------------------------------------------------------------------
# Plan math — no torch needed
# ---------------------------------------------------------------------------


def test_tp6_plan_matches_the_documented_targets():
    plan = pad.plan_for_tp(6)
    assert plan.active
    assert plan.mla_heads == 66
    assert plan.idx_heads == 36
    assert plan.kda_heads == 66
    assert plan.moe_intermediate == 2304
    assert plan.vocab_pad_to == 192


def test_tp4_needs_no_padding():
    # The shipped 4-Spark path must stay byte-for-byte the old behaviour.
    plan = pad.plan_for_tp(4)
    assert not plan.active
    assert plan.mla_heads == 64
    assert plan.moe_intermediate == 2048
    assert build_rule_count(plan) == 0


def test_tp8_needs_no_padding():
    assert not pad.plan_for_tp(8).active


def build_rule_count(plan):
    return len(pad.build_rules(plan))


def test_every_tp_up_to_16_yields_a_shardable_block_aligned_plan():
    for tp in range(1, 17):
        plan = pad.plan_for_tp(tp)  # raises if a check fails
        assert plan.mla_heads % tp == 0
        assert plan.kda_heads % tp == 0
        assert plan.idx_heads % tp == 0
        assert plan.moe_intermediate % tp == 0
        # Blockwise FP8 shards must stay whole 128-blocks on every rank.
        assert (plan.moe_intermediate // tp) % pad.FP8_BLOCK == 0
        assert (plan.mla_heads * pad.MLA_QK_HEAD_DIM // tp) % pad.FP8_BLOCK == 0
        padded_vocab = -(-pad.VOCAB_SIZE // plan.vocab_pad_to) * plan.vocab_pad_to
        assert padded_vocab % tp == 0
        assert padded_vocab % 64 == 0


def test_vocab_padding_for_tp6_is_the_next_multiple_of_192():
    # 154880 is a multiple of 64 but not of 6; upstream would hand
    # divide(154880, 6) and assert.
    assert 154880 % 6 != 0
    plan = pad.plan_for_tp(6)
    padded = -(-154880 // plan.vocab_pad_to) * plan.vocab_pad_to
    assert padded == 154944
    assert padded % 6 == 0 and padded % 64 == 0


def test_groups_can_be_disabled_individually():
    plan = pad.plan_for_tp(6, ["mla"])
    assert plan.mla_heads == 66
    assert plan.kda_heads == 64
    assert plan.moe_intermediate == 2048
    whats = {r.what for r in pad.build_rules(plan)}
    assert any(w.startswith("mla") for w in whats)
    assert not any(w.startswith("kda") for w in whats)


# ---------------------------------------------------------------------------
# Rule matching against real checkpoint names — no torch needed
# ---------------------------------------------------------------------------

L = "model.language_model.layers"

# (name, shape, expected rule `what` or None)  — shapes from the safetensors
# headers of dealignai/GLM-5.3-Flash-UNCENSORED-FP8.
REAL_TENSORS = [
    # MLA layer 3
    (f"{L}.3.self_attn.q_b_proj.weight", (16384, 1536), "mla q_b"),
    (f"{L}.3.self_attn.q_b_proj.weight_scale_inv", (128, 12), "mla q_b scale"),
    (f"{L}.3.self_attn.kv_b_proj.weight", (32768, 512), "mla kv_b"),
    (f"{L}.3.self_attn.o_proj.weight", (4096, 16384), "mla o_proj"),
    (f"{L}.3.self_attn.o_proj.weight_scale_inv", (32, 128), "mla o_proj scale"),
    (f"{L}.3.self_attn.indexer.wq_b.weight", (4096, 1536), "indexer wq_b"),
    (f"{L}.3.self_attn.indexer.weights_proj.weight", (32, 4096), "indexer weights_proj"),
    # MTP layer 45 is MLA-shaped and must be padded the same way
    (f"{L}.45.self_attn.q_b_proj.weight", (16384, 1536), "mla q_b"),
    (f"{L}.45.self_attn.o_proj.weight", (4096, 16384), "mla o_proj"),
    # MLA tensors that must be left alone
    (f"{L}.3.self_attn.q_a_proj.weight", (1536, 4096), None),
    (f"{L}.3.self_attn.kv_a_proj_with_mqa.weight", (512, 4096), None),
    (f"{L}.3.self_attn.kv_a_layernorm.weight", (512,), None),
    (f"{L}.3.self_attn.indexer.wk.weight", (128, 4096), None),
    (f"{L}.3.self_attn.indexer.k_norm.weight", (128,), None),
    (f"{L}.3.self_attn.indexer.index_kpool_compress_gate", (128, 4096), None),
    # KDA layer 0
    (f"{L}.0.self_attn.q_proj.weight", (8192, 4096), "kda qkv_proj"),
    (f"{L}.0.self_attn.k_proj.weight", (8192, 4096), "kda qkv_proj"),
    (f"{L}.0.self_attn.v_proj.weight", (8192, 4096), "kda qkv_proj"),
    (f"{L}.0.self_attn.q_conv1d.weight", (8192, 1, 4), "kda conv1d"),
    (f"{L}.0.self_attn.f_b_proj.weight", (8192, 128), "kda gate b_proj"),
    (f"{L}.0.self_attn.g_b_proj.weight", (8192, 128), "kda gate b_proj"),
    (f"{L}.0.self_attn.b_proj.weight", (64, 4096), "kda beta"),
    (f"{L}.0.self_attn.A_log", (64,), "kda A_log"),
    (f"{L}.0.self_attn.dt_bias", (8192,), "kda dt_bias"),
    (f"{L}.0.self_attn.o_proj.weight", (4096, 8192), "kda o_proj"),
    # KDA tensors that must be left alone
    (f"{L}.0.self_attn.f_a_proj.weight", (128, 4096), None),
    (f"{L}.0.self_attn.g_a_proj.weight", (128, 4096), None),
    (f"{L}.0.self_attn.o_norm.weight", (128,), None),
    # MoE
    (f"{L}.4.mlp.experts.0.gate_proj.weight", (2048, 4096), "moe gate/up"),
    (f"{L}.4.mlp.experts.287.up_proj.weight", (2048, 4096), "moe gate/up"),
    (f"{L}.4.mlp.experts.0.gate_proj.weight_scale_inv", (16, 32), "moe gate/up scale"),
    (f"{L}.4.mlp.experts.0.down_proj.weight", (4096, 2048), "moe down"),
    (f"{L}.4.mlp.experts.0.down_proj.weight_scale_inv", (32, 16), "moe down scale"),
    (f"{L}.4.mlp.shared_experts.up_proj.weight", (2048, 4096), "moe gate/up"),
    (f"{L}.4.mlp.shared_experts.down_proj.weight", (4096, 2048), "moe down"),
    # Router and the three dense MLP layers must be left alone
    (f"{L}.4.mlp.gate.weight", (288, 4096), None),
    (f"{L}.4.mlp.gate.e_score_correction_bias", (288,), None),
    (f"{L}.0.mlp.gate_proj.weight", (12288, 4096), None),
    (f"{L}.0.mlp.down_proj.weight", (4096, 12288), None),
    (f"{L}.0.mlp.down_proj.weight_scale_inv", (32, 96), None),
    # Shared / embedding / MTP glue
    ("lm_head.weight", (154880, 4096), None),
    ("model.language_model.embed_tokens.weight", (154880, 4096), None),
    (f"{L}.45.eh_proj.weight", (4096, 8192), None),
    (f"{L}.0.hc_attn_fn", (24, 16384), None),
    (f"{L}.0.input_layernorm.weight", (4096,), None),
    # Vision tower is not padded by the shim
    ("model.visual.blocks.0.attn.qkv.weight", (3072, 1024), None),
    ("model.visual.blocks.0.attn.proj.weight", (1024, 1024), None),
    ("model.visual.blocks.0.mlp.down_proj.weight", (1024, 4096), None),
    ("model.visual.merger.up_proj.weight", (10240, 4096), None),
    ("model.visual.merger.down_proj.weight", (4096, 10240), None),
]


def _match(rules, name, shape):
    for rule in rules:
        if rule.matches(name, shape):
            return rule.what
    return None


def test_rules_hit_exactly_the_intended_checkpoint_tensors():
    rules = pad.build_rules(pad.plan_for_tp(6))
    wrong = []
    for name, shape, expected in REAL_TENSORS:
        got = _match(rules, name, shape)
        if got != expected:
            wrong.append(f"{name} {shape}: expected {expected!r}, matched {got!r}")
    assert not wrong, "\n".join(wrong)


def test_mla_and_kda_o_proj_are_told_apart_by_shape_alone():
    # Both are `self_attn.o_proj.weight`. Only the extent distinguishes them.
    rules = pad.build_rules(pad.plan_for_tp(6))
    assert _match(rules, f"{L}.3.self_attn.o_proj.weight", (4096, 16384)) == "mla o_proj"
    assert _match(rules, f"{L}.0.self_attn.o_proj.weight", (4096, 8192)) == "kda o_proj"


def test_every_rule_is_exercised_by_the_fixture():
    rules = pad.build_rules(pad.plan_for_tp(6))
    covered = {_match(rules, n, s) for n, s, _ in REAL_TENSORS} - {None}
    missing = {r.what for r in rules} - covered
    assert not missing, f"rules never exercised by the fixture: {sorted(missing)}"


def test_output_side_rules_are_zero_or_ones_never_replicate():
    """The whole correctness argument rests on this split."""
    for rule in pad.build_rules(pad.plan_for_tp(6)):
        if rule.what in ("mla o_proj", "kda o_proj", "moe down"):
            assert rule.mode == pad.ZERO, rule.what
        if rule.what == "indexer weights_proj":
            assert rule.mode == pad.ZERO, "padded indexer heads must not move top-k"
        if rule.what.endswith("scale") and rule.mode != pad.REPLICATE:
            assert rule.mode == pad.ONES, rule.what


# ---------------------------------------------------------------------------
# sitecustomize hygiene — no torch needed
# ---------------------------------------------------------------------------


def test_sitecustomize_does_not_import_torch_or_vllm_at_module_level():
    """It runs at interpreter start in every process in the container."""
    with open(os.path.join(SHIM_DIR, "sitecustomize.py"), encoding="utf-8") as handle:
        source = handle.read()
    top_level = [
        line
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for line in top_level:
        assert "torch" not in line, line
        assert "vllm" not in line, line


def test_shim_module_has_no_top_level_torch_import():
    with open(os.path.join(SHIM_DIR, "glm53_tp_pad.py"), encoding="utf-8") as handle:
        source = handle.read()
    top_level = [
        line
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    assert not any("torch" in line for line in top_level), top_level


def test_cli_emits_shell_assignments():
    out = subprocess.run(
        [sys.executable, os.path.join(SHIM_DIR, "glm53_tp_pad.py"), "--tp", "6", "--shell"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "GLM53_PAD_ACTIVE=1" in out
    assert "GLM53_PAD_MLA_HEADS=66" in out
    assert "GLM53_PAD_MOE_INTERMEDIATE=2304" in out
    assert "GLM53_PAD_VOCAB=154944" in out

    out4 = subprocess.run(
        [sys.executable, os.path.join(SHIM_DIR, "glm53_tp_pad.py"), "--tp", "4", "--shell"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "GLM53_PAD_ACTIVE=0" in out4


# ---------------------------------------------------------------------------
# Tensor surgery — torch
# ---------------------------------------------------------------------------


def _pad(name, tensor, tp=6):
    padder = pad.Padder(pad.plan_for_tp(tp), strict=False)
    return padder.apply(name, tensor)


def test_padded_shapes_match_the_config_overlay():
    requires_torch()
    cases = [
        (f"{L}.3.self_attn.q_b_proj.weight", (16384, 1536), (16896, 1536)),
        (f"{L}.3.self_attn.q_b_proj.weight_scale_inv", (128, 12), (132, 12)),
        (f"{L}.3.self_attn.kv_b_proj.weight", (32768, 512), (33792, 512)),
        (f"{L}.3.self_attn.o_proj.weight", (4096, 16384), (4096, 16896)),
        (f"{L}.3.self_attn.o_proj.weight_scale_inv", (32, 128), (32, 132)),
        (f"{L}.3.self_attn.indexer.wq_b.weight", (4096, 1536), (4608, 1536)),
        (f"{L}.3.self_attn.indexer.weights_proj.weight", (32, 4096), (36, 4096)),
        (f"{L}.0.self_attn.q_proj.weight", (8192, 4096), (8448, 4096)),
        (f"{L}.0.self_attn.q_conv1d.weight", (8192, 1, 4), (8448, 1, 4)),
        (f"{L}.0.self_attn.b_proj.weight", (64, 4096), (66, 4096)),
        (f"{L}.0.self_attn.A_log", (64,), (66,)),
        (f"{L}.0.self_attn.dt_bias", (8192,), (8448,)),
        (f"{L}.0.self_attn.o_proj.weight", (4096, 8192), (4096, 8448)),
        (f"{L}.4.mlp.experts.0.gate_proj.weight", (2048, 4096), (2304, 4096)),
        (f"{L}.4.mlp.experts.0.gate_proj.weight_scale_inv", (16, 32), (18, 32)),
        (f"{L}.4.mlp.experts.0.down_proj.weight", (4096, 2048), (4096, 2304)),
        (f"{L}.4.mlp.experts.0.down_proj.weight_scale_inv", (32, 16), (32, 18)),
    ]
    for name, src_shape, want in cases:
        out = _pad(name, torch.zeros(src_shape, dtype=torch.bfloat16))
        assert tuple(out.shape) == want, f"{name}: {tuple(out.shape)} != {want}"
        # every padded extent has to shard cleanly
        for dim, size in enumerate(out.shape):
            if size in (want[dim],) and size != src_shape[dim]:
                assert size % 6 == 0, f"{name} dim{dim}={size} not divisible by 6"


def test_replicate_fill_cycles_from_index_zero():
    """Head H+i must get head i's weights, consistently across every per-head
    tensor, or the padded heads would be incoherent Frankenstein heads."""
    requires_torch()
    heads, dim = 64, 128
    src = torch.arange(heads * dim, dtype=torch.float32).reshape(heads * dim, 1)
    src = src.expand(heads * dim, 4).contiguous().to(torch.bfloat16)
    out = _pad(f"{L}.0.self_attn.q_proj.weight", src)
    assert tuple(out.shape) == (66 * dim, 4)
    # head 64 == head 0, head 65 == head 1
    assert torch.equal(out[64 * dim : 65 * dim], out[0:dim])
    assert torch.equal(out[65 * dim : 66 * dim], out[dim : 2 * dim])

    a_log = torch.arange(heads, dtype=torch.float32)
    out_a = _pad(f"{L}.0.self_attn.A_log", a_log)
    assert out_a[64] == a_log[0] and out_a[65] == a_log[1]


def test_output_side_padding_is_exactly_zero():
    requires_torch()
    src = torch.randn(4096, 8192, dtype=torch.bfloat16)
    out = _pad(f"{L}.0.self_attn.o_proj.weight", src)
    assert torch.equal(out[:, :8192], src)
    assert torch.count_nonzero(out[:, 8192:]) == 0


def test_zeroed_weight_blocks_get_scale_one_not_zero():
    """A zero weight_scale_inv would turn the dequant into 0 * inf on some paths."""
    requires_torch()
    src = torch.rand(32, 128, dtype=torch.float32) + 0.5
    out = _pad(f"{L}.3.self_attn.o_proj.weight_scale_inv", src)
    assert torch.equal(out[:, :128], src)
    assert torch.all(out[:, 128:] == 1.0)


def test_fp8_tensors_survive_the_uint8_round_trip():
    requires_torch()
    if not hasattr(torch, "float8_e4m3fn"):
        raise SkipTest("this torch has no float8_e4m3fn")
    src = (torch.randn(2048, 4096) * 0.1).to(torch.float8_e4m3fn)
    out = _pad(f"{L}.4.mlp.experts.0.gate_proj.weight", src)
    assert out.dtype == torch.float8_e4m3fn
    assert tuple(out.shape) == (2304, 4096)
    assert torch.equal(out[:2048].float(), src.float())
    assert torch.equal(out[2048:].float(), src[:256].float())

    down = (torch.randn(4096, 2048) * 0.1).to(torch.float8_e4m3fn)
    out_down = _pad(f"{L}.4.mlp.experts.0.down_proj.weight", down)
    assert torch.count_nonzero(out_down[:, 2048:].float()) == 0


def test_strict_mode_rejects_an_unknown_extent():
    """If the checkpoint changes shape under us we want a loud failure, not a
    silently unpadded tensor that dies inside a kernel 12 minutes later."""
    requires_torch()
    padder = pad.Padder(pad.plan_for_tp(6), strict=True)
    padder.apply(f"{L}.3.self_attn.q_b_proj.weight", torch.zeros(12345, 1536))
    assert padder.skipped
    try:
        padder.report()
    except RuntimeError as exc:
        assert "not by any known shape" in str(exc)
    else:
        raise AssertionError("strict mode should have raised")


def test_stream_wrapper_reports_and_passes_everything_through():
    requires_torch()
    stream = [
        (f"{L}.3.self_attn.q_b_proj.weight", torch.zeros(16384, 1536, dtype=torch.bfloat16)),
        (f"{L}.0.self_attn.A_log", torch.zeros(64)),
        (f"{L}.0.input_layernorm.weight", torch.zeros(4096, dtype=torch.bfloat16)),
    ]
    padder = pad.Padder(pad.plan_for_tp(6), strict=False)
    out = dict(padder.wrap(iter(stream)))
    assert out[f"{L}.3.self_attn.q_b_proj.weight"].shape[0] == 16896
    assert out[f"{L}.0.self_attn.A_log"].shape[0] == 66
    assert out[f"{L}.0.input_layernorm.weight"].shape[0] == 4096
    assert padder.counts["mla q_b"] == 1
    assert padder.seen == 3


# ---------------------------------------------------------------------------
# Numerical equivalence — the actual correctness claim
# ---------------------------------------------------------------------------


def _replicate_rows(weight, new_rows):
    """The REPLICATE fill, in plain torch, for building expected values."""
    old = weight.shape[0]
    out = torch.empty((new_rows,) + tuple(weight.shape[1:]), dtype=weight.dtype)
    out[:old] = weight
    written = old
    while written < new_rows:
        chunk = min(old, new_rows - written)
        out[written : written + chunk] = weight[:chunk]
        written += chunk
    return out


def _zero_cols(weight, new_cols):
    out = torch.zeros(weight.shape[0], new_cols, dtype=weight.dtype)
    out[:, : weight.shape[1]] = weight
    return out


def test_mla_attention_is_unchanged_by_head_padding():
    """64 heads vs 66 heads where the two extra heads copy head 0/1 and o_proj's
    matching columns are zero."""
    requires_torch()
    torch.manual_seed(0)
    T, H, D, hidden, latent = 7, 8, 16, 32, 24
    Hp = 12  # pad 8 -> 12, as 64 -> 66 does

    x = torch.randn(T, hidden, dtype=torch.float64)
    kv = torch.randn(T, latent, dtype=torch.float64)
    w_q = torch.randn(H * D, hidden, dtype=torch.float64)
    w_kv = torch.randn(H * 2 * D, latent, dtype=torch.float64)
    w_o = torch.randn(hidden, H * D, dtype=torch.float64)

    def forward(w_q, w_kv, w_o, heads):
        q = (x @ w_q.T).view(T, heads, D).transpose(0, 1)
        kvh = (kv @ w_kv.T).view(T, heads, 2 * D).transpose(0, 1)
        k, v = kvh[..., :D], kvh[..., D:]
        scores = (q @ k.transpose(-1, -2)) / D**0.5
        out = torch.softmax(scores, dim=-1) @ v
        return out.transpose(0, 1).reshape(T, heads * D) @ w_o.T

    base = forward(w_q, w_kv, w_o, H)
    padded = forward(
        _replicate_rows(w_q, Hp * D),
        _replicate_rows(w_kv, Hp * 2 * D),
        _zero_cols(w_o, Hp * D),
        Hp,
    )
    assert torch.allclose(base, padded, atol=1e-10), (base - padded).abs().max()


def test_moe_swiglu_is_unchanged_by_intermediate_padding():
    requires_torch()
    torch.manual_seed(0)
    T, hidden, inter, inter_p = 5, 16, 8, 12
    limit = 10.0
    x = torch.randn(T, hidden, dtype=torch.float64)
    w_gate = torch.randn(inter, hidden, dtype=torch.float64)
    w_up = torch.randn(inter, hidden, dtype=torch.float64)
    w_down = torch.randn(hidden, inter, dtype=torch.float64)

    def forward(w_gate, w_up, w_down):
        gate = (x @ w_gate.T).clamp(max=limit)
        up = x @ w_up.T
        return (torch.nn.functional.silu(gate) * up) @ w_down.T

    base = forward(w_gate, w_up, w_down)
    padded = forward(
        _replicate_rows(w_gate, inter_p),
        _replicate_rows(w_up, inter_p),
        _zero_cols(w_down, inter_p),
    )
    assert torch.allclose(base, padded, atol=1e-12), (base - padded).abs().max()


def test_kda_gated_recurrence_is_unchanged_by_head_padding():
    """A gated delta-rule style recurrence: padded heads run a real head's
    dynamics (finite, well scaled) and are cancelled by o_proj."""
    requires_torch()
    torch.manual_seed(0)
    T, H, D, hidden = 6, 4, 8, 16
    Hp = 6

    x = torch.randn(T, hidden, dtype=torch.float64)
    w_q = torch.randn(H * D, hidden, dtype=torch.float64)
    w_k = torch.randn(H * D, hidden, dtype=torch.float64)
    w_v = torch.randn(H * D, hidden, dtype=torch.float64)
    w_beta = torch.randn(H, hidden, dtype=torch.float64)
    a_log = torch.randn(H, dtype=torch.float64)
    dt_bias = torch.randn(H * D, dtype=torch.float64)
    w_o = torch.randn(hidden, H * D, dtype=torch.float64)

    def forward(w_q, w_k, w_v, w_beta, a_log, dt_bias, w_o, heads):
        q = (x @ w_q.T).view(T, heads, D)
        k = (x @ w_k.T).view(T, heads, D)
        v = (x @ w_v.T).view(T, heads, D)
        beta = torch.sigmoid(x @ w_beta.T)  # [T, heads]
        decay = torch.exp(-torch.exp(a_log))  # [heads]
        dt = torch.nn.functional.softplus(dt_bias).view(heads, D)
        state = torch.zeros(heads, D, D, dtype=torch.float64)
        outs = []
        for t in range(T):
            state = decay.view(heads, 1, 1) * state + beta[t].view(heads, 1, 1) * (
                k[t].unsqueeze(-1) @ v[t].unsqueeze(-2)
            )
            outs.append((q[t].unsqueeze(-2) @ state).squeeze(-2) * dt)
        return torch.stack(outs).reshape(T, heads * D) @ w_o.T

    base = forward(w_q, w_k, w_v, w_beta, a_log, dt_bias, w_o, H)
    padded = forward(
        _replicate_rows(w_q, Hp * D),
        _replicate_rows(w_k, Hp * D),
        _replicate_rows(w_v, Hp * D),
        _replicate_rows(w_beta, Hp),
        _replicate_rows(a_log.unsqueeze(1), Hp).squeeze(1),
        _replicate_rows(dt_bias.view(H * D, 1), Hp * D).squeeze(1),
        _zero_cols(w_o, Hp * D),
        Hp,
    )
    assert torch.allclose(base, padded, atol=1e-10), (base - padded).abs().max()


def test_indexer_topk_selection_is_unchanged_by_head_padding():
    """weights_proj is zeroed for padded heads, so they contribute no index
    logit and the selected token set is bit-identical."""
    requires_torch()
    torch.manual_seed(0)
    T, H, D, hidden, topk = 40, 8, 16, 32, 9
    Hp = 12

    x = torch.randn(T, hidden, dtype=torch.float64)
    w_q = torch.randn(H * D, hidden, dtype=torch.float64)
    w_k = torch.randn(D, hidden, dtype=torch.float64)  # MQA: one shared k head
    w_weights = torch.randn(H, hidden, dtype=torch.float64)

    def select(w_q, w_weights, heads):
        q = (x @ w_q.T).view(T, heads, D)
        k = x @ w_k.T
        per_head = torch.einsum("thd,sd->ths", q, k)
        combine = (x @ w_weights.T) * heads**-0.5
        logits = (per_head * combine.unsqueeze(-1)).sum(dim=1)
        return logits.topk(topk, dim=-1).indices.sort(dim=-1).values

    base = select(w_q, w_weights, H)
    padded = select(
        _replicate_rows(w_q, Hp * D),
        _zero_cols(w_weights.T, Hp).T,  # padded head rows are zero
        Hp,
    )
    assert torch.equal(base, padded)


def test_replicating_the_input_side_keeps_activation_blocks_nonzero():
    """Why the input side is replicated rather than zeroed.

    With a zeroed input side, the activation feeding down_proj contains whole
    all-zero 128-blocks; DeepGEMM's dynamic per-block FP8 quantisation divides
    by that block's amax. Replicating keeps every block well scaled.
    """
    requires_torch()
    torch.manual_seed(0)
    block, hidden = 128, 16
    inter, inter_p = 256, 384  # one padded 128-block
    x = torch.randn(4, hidden, dtype=torch.float64)
    w_gate = torch.randn(inter, hidden, dtype=torch.float64)
    w_up = torch.randn(inter, hidden, dtype=torch.float64)

    def amax_per_block(gate_w, up_w, width):
        act = torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)
        return act.view(x.shape[0], width // block, block).abs().amax(dim=-1)

    replicated = amax_per_block(
        _replicate_rows(w_gate, inter_p), _replicate_rows(w_up, inter_p), inter_p
    )
    assert (replicated > 0).all(), "replicate fill must not produce a dead block"

    zeroed = amax_per_block(
        _zero_cols(w_gate.T, inter_p).T, _zero_cols(w_up.T, inter_p).T, inter_p
    )
    assert (zeroed[:, -1] == 0).all(), "the rejected scheme does produce a dead block"


# ---------------------------------------------------------------------------
# Config overlay + injection — no torch needed
# ---------------------------------------------------------------------------

# The fields make_overlay.py reads, with the published values.
CONFIG_FIXTURE = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "model_type": "glm5_next",
    "text_config": {
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "index_n_heads": 32,
        "moe_intermediate_size": 2048,
        "intermediate_size": 12288,
        "n_routed_experts": 288,
        "vocab_size": 154880,
        "linear_attn_config": {"num_heads": 64, "head_dim": 128},
    },
    "vision_config": {"num_heads": 16, "hidden_size": 1024},
}


def _write_fake_model_dir(root):
    import json

    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(CONFIG_FIXTURE, handle)
    for name in ("model-00001-of-00062.safetensors", "tokenizer.json", "chat_template.jinja"):
        with open(os.path.join(root, name), "w", encoding="utf-8") as handle:
            handle.write("x")


def test_rewrite_config_produces_the_documented_values():
    """Covers the config maths even where symlinks are unavailable."""
    import copy

    sys.path.insert(0, SHIM_DIR)
    import make_overlay

    config = make_overlay.rewrite_config(
        copy.deepcopy(CONFIG_FIXTURE), pad.plan_for_tp(6)
    )
    text = config["text_config"]
    assert (text["num_attention_heads"], text["index_n_heads"]) == (66, 36)
    assert text["moe_intermediate_size"] == 2304
    assert text["linear_attn_config"]["num_heads"] == 66
    assert text["vocab_size"] == 154880
    assert len(config["_glm53_tp_pad"]["changes"]) == 5


def test_make_overlay_rewrites_config_and_symlinks_the_rest():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "model")
        dst = os.path.join(tmp, "model-tp6")
        _write_fake_model_dir(src)
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(SHIM_DIR, "make_overlay.py"),
                "--src", src, "--dst", dst, "--tp", "6",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 and "symlink" in (result.stderr + result.stdout).lower():
            raise SkipTest("this platform will not create symlinks unprivileged")
        assert result.returncode == 0, result.stderr

        with open(os.path.join(dst, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        text = config["text_config"]
        assert text["num_attention_heads"] == 66
        assert text["num_key_value_heads"] == 66
        assert text["index_n_heads"] == 36
        assert text["moe_intermediate_size"] == 2304
        assert text["linear_attn_config"]["num_heads"] == 66
        # vocab_size must NOT move: the lm_head on disk still has 154880 rows,
        # the padding happens inside vLLM's pad_vocab_size.
        assert text["vocab_size"] == 154880
        # untouched dims stay untouched
        assert text["intermediate_size"] == 12288
        assert config["vision_config"]["num_heads"] == 16
        assert config["_glm53_tp_pad"]["tp"] == 6

        for name in ("model-00001-of-00062.safetensors", "tokenizer.json"):
            link = os.path.join(dst, name)
            assert os.path.islink(link), f"{name} should be a symlink, not a copy"
            assert os.path.realpath(link) == os.path.realpath(os.path.join(src, name))


def test_make_overlay_refuses_a_checkpoint_it_does_not_recognise():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "model")
        _write_fake_model_dir(src)
        with open(os.path.join(src, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        config["text_config"]["num_attention_heads"] = 96  # a different model
        with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        result = subprocess.run(
            [
                sys.executable,
                os.path.join(SHIM_DIR, "make_overlay.py"),
                "--src", src, "--dst", os.path.join(tmp, "out"), "--tp", "6",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "checkpoint changed" in result.stdout + result.stderr


def test_sitecustomize_hook_patches_vocab_padding_on_import():
    """The meta-path post-import hook is the riskiest piece of the injection.

    Build a stand-in module tree with vLLM's exact dotted path, import it in a
    subprocess with the shim on PYTHONPATH, and check pad_vocab_size came back
    TP-aware.
    """
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        pkg = tmp
        for parts in (
            ("vllm",),
            ("vllm", "model_executor"),
            ("vllm", "model_executor", "layers"),
        ):
            path = os.path.join(pkg, *parts)
            os.makedirs(path, exist_ok=True)
            open(os.path.join(path, "__init__.py"), "w").close()
        target = os.path.join(pkg, "vllm", "model_executor", "layers", "vocab_parallel_embedding.py")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(
                textwrap.dedent(
                    """
                    def pad_vocab_size(vocab_size, pad_to=64):
                        return ((vocab_size + pad_to - 1) // pad_to) * pad_to
                    """
                )
            )

        script = textwrap.dedent(
            """
            import vllm.model_executor.layers.vocab_parallel_embedding as m
            print("RESULT", m.pad_vocab_size(154880, 64))
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([SHIM_DIR, pkg])
        env["GLM53_TP_PAD"] = "6"
        env.pop("GLM53_TP_PAD_GROUPS", None)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, result.stderr
        assert "RESULT 154944" in result.stdout, result.stdout + result.stderr
        assert "vocab padding now rounds to lcm(64, 6)=192" in result.stderr

        # ... and stays upstream-identical when the shim is off.
        env["GLM53_TP_PAD"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env
        )
        assert "RESULT 154880" in result.stdout, result.stdout + result.stderr


def test_sitecustomize_is_inert_without_the_env_var():
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        script = textwrap.dedent(
            """
            import sys
            print("METAPATH", any("PostImport" in type(f).__name__ for f in sys.meta_path))
            print("TORCH", "torch" in sys.modules)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = SHIM_DIR
        env.pop("GLM53_TP_PAD", None)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=tmp
        )
        assert result.returncode == 0, result.stderr
        assert "METAPATH False" in result.stdout
        assert "TORCH False" in result.stdout


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def _run_all() -> int:
    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except SkipTest as exc:
            print(f"SKIP {name}: {exc}")
            skipped += 1
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1
        else:
            print(f"ok   {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
