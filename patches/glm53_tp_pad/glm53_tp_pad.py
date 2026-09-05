"""Runtime tensor padding so GLM-5.3-Flash FP8 boots at a TP that does not
divide its head counts (TP=6 in particular).

Why this exists
---------------
GLM-5.3-Flash has 64 MLA heads, 64 KDA (linear-attention) heads, 32 sparse
indexer heads, a 2048-wide MoE expert intermediate and a 154880 vocab. None of
those divide by 6, so vLLM asserts inside ``divide()`` long before the first
forward pass. This module pads each of them up to a multiple of the tensor
parallel size *as the weights stream in*, so nothing on disk changes and the
sharding logic in vLLM stays untouched.

The padding rule
----------------
One rule, one exception:

  * column-parallel weights (the **output** dim is sharded) -> the padded slots
    get a *cyclic copy* of the real leading slots.
  * row-parallel weights (the **input** dim is sharded) -> the padded columns
    are **zeroed**.

Zeroing on the output side is what makes a padded head/expert-slot contribute
exactly nothing to the result. Replicating on the input side is deliberate and
not cosmetic: if the input side were zeroed too, the activations feeding
``o_proj`` / ``down_proj`` would contain entire all-zero 128-element blocks, and
DeepGEMM's dynamic per-block FP8 quantisation computes ``amax == 0`` there ->
0/0 -> NaN. Copying real weights keeps every activation block well scaled while
the output-side zeros still cancel the contribution.

The one exception is ``indexer.weights_proj`` (one row per indexer head, it is
the *combining* weight for the per-head index logits). Semantically that is an
output-side tensor, so padded heads must get **zero** there or they would
perturb the top-k token selection. That group is **off by default** — see below.

The indexer is not sharded
--------------------------
``scripts/glm53-tp-probe.sh`` settled this against the image's own glm5next
drop: the indexer's only parallel module is built with ``disable_tp=True``
(``models/glm5next/nvidia/attention.py:259-264``), and ``index_n_heads`` never
meets ``tp_size`` or ``divide()`` anywhere in that implementation. The 32 heads
are replicated on every rank, so TP=6 does not need them padded and padding them
would only put 4 dead heads through a top-k kernel that already has its own
head-count fixup path. ``indexer`` therefore stays out of ``DEFAULT_GROUPS``;
it is one env var away if a future build starts sharding it.

Because the replicate fill is a cyclic copy from index 0 and every affected
tensor is head-major, head ``H+i`` consistently receives head ``i``'s weights
across ``q_proj``/``A_log``/``dt_bias``/``b_proj``/... — the padded heads are
coherent duplicates of real heads, not a mix.

FP8 ``weight_scale_inv`` companions are padded the same way: replicated where
the weight is replicated, filled with **1.0** where the weight is zeroed (the
quantised value is 0 there, so the scale only has to be finite and non-zero).

Safety
------
Every rule carries the exact unpadded extent it expects on the padded dim, taken
from the published checkpoint. A tensor is only rewritten when its name matches
*and* its shape matches. That shape guard is also the dispatcher: MLA and KDA
both have a ``self_attn.o_proj.weight`` but they are 16384 and 8192 wide
respectively, and only MLA's is FP8.

Env vars
--------
``GLM53_TP_PAD``         tensor parallel size. Unset/1 -> shim does nothing.
``GLM53_TP_PAD_GROUPS``  comma list from {mla,kda,indexer,moe,vocab}.
                         Default ``mla,kda,moe,vocab`` (see above for indexer).
``GLM53_TP_PAD_DEBUG``   1 -> log every rewritten tensor.
``GLM53_TP_PAD_STRICT``  0 -> downgrade consistency errors to warnings.
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

# NOTE: torch is imported lazily. sitecustomize.py imports this module during
# interpreter start-up in *every* Python process in the container; pulling torch
# in there would cost seconds per subprocess for nothing.

__all__ = [
    "PadPlan",
    "plan_for_tp",
    "Padder",
    "patch_weight_loader",
    "patch_vocab_padding",
    "patch_encoder_dp",
    "install",
]

LOG_PREFIX = "[glm53-tp-pad]"

# ---------------------------------------------------------------------------
# Architecture constants, read off the published checkpoint. These are the
# *unpadded* extents; they double as the shape guard for every rule.
#   config.json ....... dealignai/GLM-5.3-Flash-UNCENSORED-FP8
#   shapes ............ safetensors headers of shards 1/2/32/46/62
# ---------------------------------------------------------------------------
MLA_HEADS = 64  # text_config.num_attention_heads
MLA_QK_HEAD_DIM = 256  # qk_nope_head_dim 256 + qk_rope_head_dim 0
MLA_KVB_HEAD_DIM = 512  # kv_b_proj packs qk_nope 256 + v 256 per head
IDX_HEADS = 32  # text_config.index_n_heads
IDX_HEAD_DIM = 128  # text_config.index_head_dim
KDA_HEADS = 64  # linear_attn_config.num_heads
KDA_HEAD_DIM = 128  # linear_attn_config.head_dim
MOE_INTERMEDIATE = 2048  # text_config.moe_intermediate_size
VOCAB_SIZE = 154880  # text_config.vocab_size
FP8_BLOCK = 128  # blockwise FP8 scale granularity

ALL_GROUPS = ("mla", "kda", "indexer", "moe", "vocab")

#: What actually gets padded unless told otherwise. ``indexer`` is excluded on
#: purpose: the image's glm5next builds the indexer's only parallel module with
#: ``disable_tp=True``, so its 32 heads live whole on every rank and never meet
#: a divisibility assert. Padding them would be pure dead work on a top-k kernel
#: that already carries its own head-count fixup. Re-enable with
#: ``GLM53_TP_PAD_GROUPS=mla,kda,indexer,moe,vocab`` if a build starts sharding
#: it — ``scripts/glm53-tp-probe.sh`` is what tells you.
DEFAULT_GROUPS = ("mla", "kda", "moe", "vocab")

_fp8_cache: tuple | None = None


def _fp8_dtypes() -> tuple:
    global _fp8_cache
    if _fp8_cache is None:
        import torch

        _fp8_cache = tuple(
            d
            for d in (
                getattr(torch, "float8_e4m3fn", None),
                getattr(torch, "float8_e5m2", None),
                getattr(torch, "float8_e4m3fnuz", None),
            )
            if d is not None
        )
    return _fp8_cache


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PadPlan:
    """Target extents for one tensor parallel size."""

    tp: int
    groups: frozenset
    mla_heads: int
    idx_heads: int
    kda_heads: int
    moe_intermediate: int
    vocab_pad_to: int

    @property
    def active(self) -> bool:
        """True when this TP actually needs something padded."""
        return (
            self.mla_heads != MLA_HEADS
            or self.idx_heads != IDX_HEADS
            or self.kda_heads != KDA_HEADS
            or self.moe_intermediate != MOE_INTERMEDIATE
            or _round_up(VOCAB_SIZE, self.vocab_pad_to) != _round_up(VOCAB_SIZE, 64)
        )

    def summary(self) -> str:
        bits = []
        if self.mla_heads != MLA_HEADS:
            bits.append(f"mla heads {MLA_HEADS}->{self.mla_heads}")
        if self.idx_heads != IDX_HEADS:
            bits.append(f"indexer heads {IDX_HEADS}->{self.idx_heads}")
        if self.kda_heads != KDA_HEADS:
            bits.append(f"kda heads {KDA_HEADS}->{self.kda_heads}")
        if self.moe_intermediate != MOE_INTERMEDIATE:
            bits.append(f"moe intermediate {MOE_INTERMEDIATE}->{self.moe_intermediate}")
        padded_vocab = _round_up(VOCAB_SIZE, self.vocab_pad_to)
        if padded_vocab != _round_up(VOCAB_SIZE, 64):
            bits.append(f"vocab {VOCAB_SIZE}->{padded_vocab}")
        return ", ".join(bits) if bits else "nothing to pad"


def plan_for_tp(tp: int, groups: Iterable[str] | None = None) -> PadPlan:
    """Compute the padded extents for ``tp`` ranks.

    Head counts round up to a multiple of ``tp``. The MoE intermediate rounds up
    to a multiple of ``tp * 128`` so every rank still owns whole blockwise-FP8
    128-blocks. Head dims are 256 (MLA) and 128 (KDA/indexer), all multiples of
    128, so the head padding is automatically block aligned.
    """
    if tp < 1:
        raise ValueError(f"tp must be >= 1, got {tp}")
    selected = frozenset(DEFAULT_GROUPS if groups is None else groups)
    unknown = selected - frozenset(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown pad groups: {sorted(unknown)}")

    plan = PadPlan(
        tp=tp,
        groups=selected,
        mla_heads=_round_up(MLA_HEADS, tp) if "mla" in selected else MLA_HEADS,
        idx_heads=_round_up(IDX_HEADS, tp) if "indexer" in selected else IDX_HEADS,
        kda_heads=_round_up(KDA_HEADS, tp) if "kda" in selected else KDA_HEADS,
        moe_intermediate=(
            _round_up(MOE_INTERMEDIATE, tp * FP8_BLOCK)
            if "moe" in selected
            else MOE_INTERMEDIATE
        ),
        vocab_pad_to=math.lcm(64, tp) if "vocab" in selected else 64,
    )
    _validate(plan)
    return plan


def _validate(plan: PadPlan) -> None:
    """Check the extents this plan is responsible for.

    Only enabled groups are checked. Switching a group off is an explicit claim
    that the dimension is not TP-sharded in this build (the indexer at 32 heads
    is the realistic candidate), so demanding divisibility there would be wrong.
    """
    tp = plan.tp
    checks = [
        ("mla", "mla q_b out", plan.mla_heads * MLA_QK_HEAD_DIM),
        ("mla", "mla kv_b out", plan.mla_heads * MLA_KVB_HEAD_DIM),
        ("indexer", "indexer wq_b out", plan.idx_heads * IDX_HEAD_DIM),
        ("indexer", "indexer heads", plan.idx_heads),
        ("kda", "kda qkv out", plan.kda_heads * KDA_HEAD_DIM),
        ("kda", "kda heads", plan.kda_heads),
        ("moe", "moe intermediate", plan.moe_intermediate),
        ("vocab", "vocab", _round_up(VOCAB_SIZE, plan.vocab_pad_to)),
    ]
    for group, what, size in checks:
        if group in plan.groups and size % tp:
            raise AssertionError(f"{what}={size} is not divisible by tp={tp}")

    # Blockwise FP8 shards must stay whole 128-blocks on every rank.
    for group, what, size in (
        ("mla", "mla q_b out", plan.mla_heads * MLA_QK_HEAD_DIM),
        ("mla", "mla o_proj in", plan.mla_heads * MLA_QK_HEAD_DIM),
        ("moe", "moe intermediate", plan.moe_intermediate),
    ):
        if group not in plan.groups:
            continue
        per_rank = size // tp
        if per_rank % FP8_BLOCK:
            raise AssertionError(
                f"{what} per rank = {per_rank} is not a multiple of {FP8_BLOCK}"
            )


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

REPLICATE = "replicate"
ZERO = "zero"
ONES = "ones"


@dataclass(frozen=True)
class Rule:
    group: str
    pattern: re.Pattern
    dim: int
    old: int
    new: int
    mode: str
    what: str

    def matches(self, name: str, shape) -> bool:
        return (
            self.pattern.search(name) is not None
            and len(shape) > self.dim
            and shape[self.dim] == self.old
        )


def _rx(tail: str) -> re.Pattern:
    # Match on the tail of the checkpoint name so an upstream rename of the
    # `model.language_model.` prefix cannot silently disable a rule.
    return re.compile(tail)


def build_rules(plan: PadPlan) -> list[Rule]:
    """Rules that actually change something, in match order."""
    rules: list[Rule] = []

    def add(group, tail, dim, old, new, mode, what):
        if old != new and group in plan.groups:
            rules.append(Rule(group, _rx(tail), dim, old, new, mode, what))

    # -- MLA full-attention layers (3, 7, ... 43) and the MTP layer (45) ------
    h, hp = MLA_HEADS, plan.mla_heads
    add(
        "mla", r"\.self_attn\.q_b_proj\.weight$", 0,
        h * MLA_QK_HEAD_DIM, hp * MLA_QK_HEAD_DIM, REPLICATE, "mla q_b",
    )
    add(
        "mla", r"\.self_attn\.q_b_proj\.weight_scale_inv$", 0,
        h * MLA_QK_HEAD_DIM // FP8_BLOCK, hp * MLA_QK_HEAD_DIM // FP8_BLOCK,
        REPLICATE, "mla q_b scale",
    )
    add(
        "mla", r"\.self_attn\.kv_b_proj\.weight$", 0,
        h * MLA_KVB_HEAD_DIM, hp * MLA_KVB_HEAD_DIM, REPLICATE, "mla kv_b",
    )
    add(
        "mla", r"\.self_attn\.o_proj\.weight$", 1,
        h * MLA_QK_HEAD_DIM, hp * MLA_QK_HEAD_DIM, ZERO, "mla o_proj",
    )
    add(
        "mla", r"\.self_attn\.o_proj\.weight_scale_inv$", 1,
        h * MLA_QK_HEAD_DIM // FP8_BLOCK, hp * MLA_QK_HEAD_DIM // FP8_BLOCK,
        ONES, "mla o_proj scale",
    )

    # -- sparse indexer, one per MLA layer -----------------------------------
    i, ip = IDX_HEADS, plan.idx_heads
    add(
        "indexer", r"\.self_attn\.indexer\.wq_b\.weight$", 0,
        i * IDX_HEAD_DIM, ip * IDX_HEAD_DIM, REPLICATE, "indexer wq_b",
    )
    # Output-side despite being column-parallel: this is the per-head combining
    # weight for the index logits. Zero => padded heads cannot move the top-k.
    add(
        "indexer", r"\.self_attn\.indexer\.weights_proj\.weight$", 0,
        i, ip, ZERO, "indexer weights_proj",
    )

    # -- KDA linear-attention layers (34 of them) ----------------------------
    k, kp = KDA_HEADS, plan.kda_heads
    kd = KDA_HEAD_DIM
    add(
        "kda", r"\.self_attn\.[qkv]_proj\.weight$", 0,
        k * kd, kp * kd, REPLICATE, "kda qkv_proj",
    )
    add(
        "kda", r"\.self_attn\.[qkv]_conv1d\.weight$", 0,
        k * kd, kp * kd, REPLICATE, "kda conv1d",
    )
    add(
        "kda", r"\.self_attn\.[fg]_b_proj\.weight$", 0,
        k * kd, kp * kd, REPLICATE, "kda gate b_proj",
    )
    add("kda", r"\.self_attn\.b_proj\.weight$", 0, k, kp, REPLICATE, "kda beta")
    add("kda", r"\.self_attn\.A_log$", 0, k, kp, REPLICATE, "kda A_log")
    add("kda", r"\.self_attn\.dt_bias$", 0, k * kd, kp * kd, REPLICATE, "kda dt_bias")
    add(
        "kda", r"\.self_attn\.o_proj\.weight$", 1,
        k * kd, kp * kd, ZERO, "kda o_proj",
    )

    # -- MoE experts and the shared expert -----------------------------------
    # `(experts\.\d+|shared_experts)` keeps the first three dense MLP layers
    # (12288 wide, already divisible) out of this.
    m, mp = MOE_INTERMEDIATE, plan.moe_intermediate
    moe_owner = r"\.mlp\.(?:experts\.\d+|shared_experts)\."
    add(
        "moe", moe_owner + r"(?:gate|up)_proj\.weight$", 0,
        m, mp, REPLICATE, "moe gate/up",
    )
    add(
        "moe", moe_owner + r"(?:gate|up)_proj\.weight_scale_inv$", 0,
        m // FP8_BLOCK, mp // FP8_BLOCK, REPLICATE, "moe gate/up scale",
    )
    add("moe", moe_owner + r"down_proj\.weight$", 1, m, mp, ZERO, "moe down")
    add(
        "moe", moe_owner + r"down_proj\.weight_scale_inv$", 1,
        m // FP8_BLOCK, mp // FP8_BLOCK, ONES, "moe down scale",
    )

    return rules


# ---------------------------------------------------------------------------
# Tensor surgery
# ---------------------------------------------------------------------------


def _tile_into(dst: torch.Tensor, src: torch.Tensor, dim: int) -> None:
    """Fill ``dst`` along ``dim`` by cycling through ``src`` from index 0."""
    span = src.size(dim)
    written = 0
    while written < dst.size(dim):
        chunk = min(span, dst.size(dim) - written)
        dst.narrow(dim, written, chunk).copy_(src.narrow(dim, 0, chunk))
        written += chunk


def pad_tensor(tensor: "torch.Tensor", rule: Rule) -> "torch.Tensor":
    """Return a new tensor grown from ``rule.old`` to ``rule.new`` on ``rule.dim``."""
    import torch

    tensor = tensor.contiguous()
    is_fp8 = tensor.dtype in _fp8_dtypes()
    # FP8 lacks kernel coverage for zero_()/fill_() on some builds; reinterpret
    # as uint8 and back. Everything here is pure data movement, so the bit
    # pattern is preserved and 0x00 is FP8 zero.
    src = tensor.view(torch.uint8) if is_fp8 else tensor

    shape = list(src.shape)
    shape[rule.dim] = rule.new
    out = torch.empty(shape, dtype=src.dtype, device=src.device)

    real = out.narrow(rule.dim, 0, rule.old)
    real.copy_(src)
    pad = out.narrow(rule.dim, rule.old, rule.new - rule.old)

    if rule.mode == REPLICATE:
        _tile_into(pad, real, rule.dim)
    elif rule.mode == ZERO:
        pad.zero_()
    elif rule.mode == ONES:
        if is_fp8:
            raise AssertionError(f"{rule.what}: ONES fill is only for f32 scales")
        pad.fill_(1.0)
    else:  # pragma: no cover - guarded by build_rules
        raise ValueError(f"unknown fill mode {rule.mode!r}")

    return out.view(tensor.dtype) if is_fp8 else out


# ---------------------------------------------------------------------------
# Padder
# ---------------------------------------------------------------------------


#: A weight stream shorter than this is a secondary source (e.g. a separately
#: hosted drafter), not the 76k-tensor main checkpoint. Only the main stream is
#: expected to exercise every rule, so only it gets the coverage check.
MAIN_STREAM_MIN_TENSORS = 1000


@dataclass
class Padder:
    plan: PadPlan
    debug: bool = False
    strict: bool = True
    rules: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    skipped: list = field(default_factory=list)
    seen: int = 0
    exhausted: bool = False

    def __post_init__(self):
        if not self.rules:
            self.rules = build_rules(self.plan)
        for rule in self.rules:
            self.counts.setdefault(rule.what, 0)
        # Every extent this architecture can legitimately present, regardless of
        # which groups are switched on. Used to tell "a group is off" apart from
        # "the checkpoint moved under us".
        self._known = build_rules(plan_for_tp(self.plan.tp, ALL_GROUPS))

    # -- one tensor ------------------------------------------------------
    def apply(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        self.seen += 1
        for rule in self.rules:
            if rule.matches(name, tensor.shape):
                out = pad_tensor(tensor, rule)
                self.counts[rule.what] += 1
                if self.debug:
                    _log(
                        f"{rule.what:22s} {name}  {tuple(tensor.shape)} -> "
                        f"{tuple(out.shape)}  [{rule.mode}]"
                    )
                return out
        self._check_unmatched(name, tensor)
        return tensor

    def _check_unmatched(self, name: str, tensor: torch.Tensor) -> None:
        """A name that looks like a padding target but carries an extent no rule
        knows about means the checkpoint or the vLLM naming moved under us.

        Checked against the *all groups* rule set: an extent that some disabled
        group would have handled (KDA's 8192-wide ``o_proj`` when ``kda`` is off,
        say) is expected, not a problem.
        """
        hit = None
        for rule in self._known:
            if rule.pattern.search(name) is None:
                continue
            if len(tensor.shape) <= rule.dim:
                continue
            if tensor.shape[rule.dim] in (rule.old, rule.new):
                return  # a known extent, just not one we are padding
            hit = rule
        if hit is not None:
            self.skipped.append((name, tuple(tensor.shape), hit.what))

    # -- stream ----------------------------------------------------------
    def wrap(self, weights: Iterable) -> Iterator:
        try:
            for name, tensor in weights:
                yield name, self.apply(name, tensor)
            self.exhausted = True
        finally:
            self.report()

    # -- reporting -------------------------------------------------------
    def report(self) -> None:
        if not self.rules:
            return
        total = sum(self.counts.values())
        _log(f"padded {total}/{self.seen} tensors for TP={self.plan.tp}")
        for what in sorted(self.counts):
            _log(f"    {what:22s} {self.counts[what]}")

        problems = []
        if self.exhausted and self.seen >= MAIN_STREAM_MIN_TENSORS:
            empty = [w for w, n in self.counts.items() if n == 0]
            if empty:
                problems.append(
                    "no tensor matched these rules (checkpoint or vLLM naming "
                    f"changed?): {', '.join(sorted(empty))}"
                )
        if self.skipped:
            preview = ", ".join(f"{n} {s}" for n, s, _ in self.skipped[:5])
            problems.append(
                f"{len(self.skipped)} tensors matched a rule by name but not by "
                f"any known shape: {preview}"
            )
        for problem in problems:
            if self.strict:
                raise RuntimeError(f"{LOG_PREFIX} {problem}")
            _log(f"WARNING: {problem}")


def _log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# vLLM patches
# ---------------------------------------------------------------------------

_state: dict = {
    "plan": None,
    "loader_patched": False,
    "vocab_patched": False,
    "encoder_dp_patched": False,
}


def _env_plan() -> PadPlan | None:
    raw = os.environ.get("GLM53_TP_PAD", "").strip()
    if not raw:
        return None
    try:
        tp = int(raw)
    except ValueError:
        _log(f"WARNING: GLM53_TP_PAD={raw!r} is not an int; shim disabled")
        return None
    if tp <= 1:
        return None
    groups_raw = os.environ.get("GLM53_TP_PAD_GROUPS", "").strip()
    groups = [g.strip() for g in groups_raw.split(",") if g.strip()] or None
    plan = plan_for_tp(tp, groups)
    return plan if plan.active else None


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip() not in ("", "0", "false", "no")


def patch_weight_loader(module) -> bool:
    """Wrap ``DefaultModelLoader``'s weight stream so tensors arrive padded.

    Hooking ``_get_weights_iterator`` puts the padding ahead of every per-param
    ``weight_loader``, so vLLM's column/row-parallel narrowing needs no changes
    and secondary weight sources (the MTP drafter) are covered too.
    """
    plan = _state.get("plan") or _env_plan()
    if plan is None:
        return False
    _state["plan"] = plan
    if _state["loader_patched"]:
        return True

    loader_cls = getattr(module, "DefaultModelLoader", None)
    if loader_cls is None:
        return False

    for seam in ("_get_weights_iterator", "get_all_weights"):
        original = getattr(loader_cls, seam, None)
        if original is None:
            continue

        def make(original):
            def wrapper(self, *args, **kwargs):
                padder = Padder(
                    plan,
                    debug=_flag("GLM53_TP_PAD_DEBUG", False),
                    strict=_flag("GLM53_TP_PAD_STRICT", True),
                )
                return padder.wrap(original(self, *args, **kwargs))

            wrapper.__name__ = seam
            wrapper.__wrapped__ = original
            return wrapper

        setattr(loader_cls, seam, make(original))
        _state["loader_patched"] = True
        _log(
            f"weight loader patched at {loader_cls.__name__}.{seam}; "
            f"TP={plan.tp} ({plan.summary()})"
        )
        return True

    _log(
        "WARNING: DefaultModelLoader has neither _get_weights_iterator nor "
        "get_all_weights; weights will NOT be padded"
    )
    return False


def patch_vocab_padding(module) -> bool:
    """Make ``pad_vocab_size`` round up to ``lcm(padding_size, tp)``.

    Upstream rounds to 64 only and ignores tp, so the subsequent
    ``vocab_range_from_global_vocab_size`` -> ``divide(154880, 6)`` asserts.
    """
    plan = _state.get("plan") or _env_plan()
    if plan is None:
        return False
    _state["plan"] = plan
    if _state["vocab_patched"]:
        return True

    original = getattr(module, "pad_vocab_size", None)
    if original is None:
        _log("WARNING: pad_vocab_size not found; vocab will NOT be padded for TP")
        return False

    tp = plan.tp

    def pad_vocab_size(vocab_size: int, pad_to: int = 64, *args, **kwargs) -> int:
        return _round_up(vocab_size, math.lcm(pad_to, tp))

    pad_vocab_size.__wrapped__ = original
    module.pad_vocab_size = pad_vocab_size

    # Other modules may hold `from ... import pad_vocab_size` references.
    rebound = 0
    for other in list(sys.modules.values()):
        if other is None or other is module:
            continue
        if getattr(other, "pad_vocab_size", None) is original:
            other.pad_vocab_size = pad_vocab_size
            rebound += 1

    _state["vocab_patched"] = True
    _log(
        f"vocab padding now rounds to lcm(64, {tp})={math.lcm(64, tp)} "
        f"-> {_round_up(VOCAB_SIZE, math.lcm(64, tp))}"
        + (f" ({rebound} extra import site(s) rebound)" if rebound else "")
    )
    return True


#: Classes carrying this marker let vLLM honour ``--mm-encoder-tp-mode data``.
ENCODER_DP_MARKER = "supports_encoder_tp_data"


def patch_encoder_dp(module) -> bool:
    """Declare encoder data-parallel support on the glm5next model class.

    The vision tower is 1024 wide with 16 heads, a 4096 MLP and a 10240 merger.
    None of those divide by 6 and none of them is padded here — the tower is not
    part of any pad group, so hosting the encoder whole on every rank is the only
    way through. The implementation already does exactly that when asked:
    ``use_data_parallel`` (set from ``mm_encoder_tp_mode == "data"``) forces the
    tower's ``tp_size`` to 1 and passes ``disable_tp=True`` to every vision
    linear. What it never does is set the ``supports_encoder_tp_data`` marker
    that vLLM checks before it grants the mode — without it the request is
    silently downgraded to ``weights`` and the tower shards anyway, straight into
    ``divide(16, 6)``.

    Setting the marker is a statement about the model, not a behaviour change:
    if vLLM's build has no such gate the attribute is simply unused.
    """
    plan = _state.get("plan") or _env_plan()
    if plan is None:
        return False
    _state["plan"] = plan
    if _state["encoder_dp_patched"]:
        return True

    marked = []
    for attr in dir(module):
        obj = getattr(module, attr, None)
        if not isinstance(obj, type):
            continue
        # Only classes this module actually defines, so we cannot brand an
        # imported base class that other models share.
        if getattr(obj, "__module__", None) != getattr(module, "__name__", None):
            continue
        if not attr.endswith("ForConditionalGeneration"):
            continue
        if getattr(obj, ENCODER_DP_MARKER, False):
            continue
        try:
            setattr(obj, ENCODER_DP_MARKER, True)
        except Exception as exc:  # pragma: no cover - exotic metaclass
            _log(f"WARNING: could not mark {attr}.{ENCODER_DP_MARKER}: {exc}")
            continue
        marked.append(attr)

    if not marked:
        return False
    _state["encoder_dp_patched"] = True
    _log(
        f"{ENCODER_DP_MARKER}=True on {', '.join(marked)} — the vision tower "
        "(1024/16/4096/10240, none divisible by "
        f"{plan.tp}) stays whole on every rank. Requires "
        "--mm-encoder-tp-mode data on the command line."
    )
    return True


#: module name -> patch function, consumed by sitecustomize.py.
HOOKS: dict[str, Callable] = {
    "vllm.model_executor.model_loader.default_loader": patch_weight_loader,
    # older layouts kept the loaders in one module
    "vllm.model_executor.model_loader.loader": patch_weight_loader,
    "vllm.model_executor.layers.vocab_parallel_embedding": patch_vocab_padding,
    # The glm5next drop lives outside vllm.model_executor.models in this image;
    # candidates that do not exist simply never fire.
    "vllm.models.glm5next.nvidia.model": patch_encoder_dp,
    "vllm.model_executor.models.glm5next.nvidia.model": patch_encoder_dp,
    "vllm.model_executor.models.glm5next": patch_encoder_dp,
}


def install() -> bool:
    """Patch whatever vLLM modules are already imported. Idempotent."""
    ok = False
    for name, fn in HOOKS.items():
        module = sys.modules.get(name)
        if module is not None:
            ok = fn(module) or ok
    return ok


# ---------------------------------------------------------------------------
# CLI — so the launch scripts and the container share one plan implementation
# instead of re-deriving 66/36/2304 in bash.
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="GLM-5.3 TP padding plan")
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--groups", default="")
    parser.add_argument(
        "--shell",
        action="store_true",
        help="emit eval-able shell assignments instead of prose",
    )
    args = parser.parse_args(argv)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    plan = plan_for_tp(args.tp, groups)

    if args.shell:
        import shlex

        print(f"GLM53_PAD_ACTIVE={1 if plan.active else 0}")
        print(f"GLM53_PAD_SUMMARY={shlex.quote(plan.summary())}")
        print(f"GLM53_PAD_GROUPS={shlex.quote(','.join(sorted(plan.groups)))}")
        print(f"GLM53_PAD_MLA_HEADS={plan.mla_heads}")
        print(f"GLM53_PAD_IDX_HEADS={plan.idx_heads}")
        print(f"GLM53_PAD_KDA_HEADS={plan.kda_heads}")
        print(f"GLM53_PAD_MOE_INTERMEDIATE={plan.moe_intermediate}")
        print(f"GLM53_PAD_VOCAB={_round_up(VOCAB_SIZE, plan.vocab_pad_to)}")
        return 0

    print(f"TP={plan.tp} active={plan.active} groups={','.join(sorted(plan.groups))}")
    print(f"  {plan.summary()}")
    for rule in build_rules(plan):
        print(
            f"  {rule.group:8s} {rule.what:22s} dim{rule.dim} "
            f"{rule.old} -> {rule.new}  [{rule.mode}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
