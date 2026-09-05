"""Read-only probe for the vLLM build inside the GLM-5.3 image.

The glm5next implementation in `vllm/vllm-openai:glm53-flash-arm64-cu130` is a
custom drop with a non-upstream layout (`vllm/models/glm5next/nvidia/ops/...`),
so the exact seams the padding shim hooks cannot be checked from a dev box. Run
this in the container before the first TP!=4 boot and read the report.

    python3 /opt/glm53/probe.py --tp 6

Touches nothing. Everything it prints is either a filesystem read or an
attribute lookup.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import sys

SECTION = "=" * 72

#: Kept in step with glm53_tp_pad.ENCODER_DP_MARKER without importing it here —
#: this section runs before the plan section adds the shim to sys.path.
ENCODER_DP_MARKER = "supports_encoder_tp_data"

# What we want to know about the custom implementation, and why.
GREPS = [
    ("tp-divide", r"\bdivide\s*\(", "every divisibility assert that TP=6 can trip"),
    ("tp-size", r"tensor_model_parallel_world_size|\btp_size\b", "TP-aware code"),
    ("mla-heads", r"num_attention_heads", "does it read the padded head count?"),
    ("idx-heads", r"index_n_heads", "is the indexer TP-sharded at all?"),
    ("kda-heads", r"linear_attn_config|kda_layers", "KDA head plumbing"),
    ("moe-inter", r"moe_intermediate_size", "MoE intermediate sharding"),
    ("col-parallel", r"ColumnParallelLinear|QKVParallelLinear|MergedColumnParallel", ""),
    ("row-parallel", r"RowParallelLinear", ""),
    ("disable-tp", r"disable_tp", "can a submodule opt out of TP?"),
    ("encoder-dp", r"encoder_tp_mode|supports_encoder_tp_data|use_data_parallel", "vision DP support"),
]


def _print_header(title: str) -> None:
    print(f"\n{SECTION}\n{title}\n{SECTION}")


def _vllm_root() -> str | None:
    """Locate the vllm package without executing its __init__."""
    try:
        spec = importlib.util.find_spec("vllm")
    except Exception as exc:
        print(f"  find_spec('vllm') failed: {exc}")
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _walk_py(root: str):
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def section_env() -> None:
    _print_header("environment")
    print(f"  python           {sys.version.split()[0]}")
    for name in ("torch", "vllm", "flashinfer"):
        try:
            module = importlib.import_module(name)
            print(f"  {name:16s} {getattr(module, '__version__', '?')}")
        except Exception as exc:
            print(f"  {name:16s} import failed: {type(exc).__name__}: {exc}")
    for var in ("GLM53_TP_PAD", "GLM53_TP_PAD_GROUPS", "PYTHONPATH"):
        print(f"  ${var:22s} {os.environ.get(var, '(unset)')}")


def section_model_files(root: str) -> list[str]:
    _print_header("glm5next implementation files")
    hits = []
    for path in _walk_py(root):
        rel = os.path.relpath(path, root)
        if re.search(r"glm5.?next|glm5_3|glm53", rel, re.IGNORECASE):
            hits.append(path)
    if not hits:
        print("  NONE FOUND — the model may be registered under a different name.")
        print("  Widen the search:  find <vllm-root> -name '*glm*'")
    for path in sorted(hits):
        print(f"  {os.path.relpath(path, root)}")
    return hits


def section_greps(root: str, files: list[str]) -> None:
    _print_header("divisibility / TP seams in the glm5next implementation")
    if not files:
        print("  (skipped: no implementation files located)")
        return
    for key, pattern, why in GREPS:
        regex = re.compile(pattern)
        matches = []
        for path in files:
            for lineno, line in enumerate(_read(path).splitlines(), 1):
                if regex.search(line):
                    matches.append(
                        f"    {os.path.relpath(path, root)}:{lineno}: {line.strip()[:110]}"
                    )
        suffix = f"  ({why})" if why else ""
        print(f"\n  [{key}] {len(matches)} hit(s){suffix}")
        for line in matches[:40]:
            print(line)
        if len(matches) > 40:
            print(f"    ... {len(matches) - 40} more")


def section_vision(root: str, files: list[str]) -> None:
    _print_header("vision tower — the one dimension set we could not pre-decide")
    print("  hidden 1024 / heads 16 / mlp 4096 / merger 10240: none divide by 6.")
    print("  If the tower uses parallel linears AND does not support encoder DP,")
    print("  --mm-encoder-tp-mode data silently falls back to 'weights' and TP=6")
    print("  will still fail here. Look for parallel linears in the files below.\n")
    vision_files = [f for f in files if re.search(r"vis|vit|image", f, re.IGNORECASE)]
    if not vision_files:
        vision_files = files
    parallel = re.compile(r"(Column|Row|QKV|MergedColumn)ParallelLinear")
    plain = re.compile(r"nn\.Linear")
    for path in vision_files:
        text = _read(path)
        rel = os.path.relpath(path, root)
        n_par = len(parallel.findall(text))
        n_plain = len(plain.findall(text))
        if n_par or n_plain:
            print(f"  {rel}: {n_par} parallel linear(s), {n_plain} nn.Linear")
    print("\n  encoder-DP capability:")
    declares = False
    honours = 0
    reads_mode = False
    for path in files:
        text = _read(path)
        rel = os.path.relpath(path, root)
        if ENCODER_DP_MARKER in text:
            declares = True
            print(f"    {rel}: declares {ENCODER_DP_MARKER}")
        if "mm_encoder_tp_mode" in text:
            reads_mode = True
            print(f"    {rel}: reads mm_encoder_tp_mode")
        honours += len(re.findall(r"disable_tp\s*=\s*use_data_parallel", text))
    if honours:
        print(f"    {honours} vision linear(s) built with disable_tp=use_data_parallel")
    if reads_mode and honours and not declares:
        print(
            "\n    VERDICT: the tower implements encoder DP but never declares the\n"
            f"    {ENCODER_DP_MARKER} marker. If vLLM core gates on that marker it will\n"
            "    downgrade --mm-encoder-tp-mode data to 'weights' and shard the tower\n"
            "    anyway. glm53_tp_pad.patch_encoder_dp sets the marker for exactly\n"
            "    this case; see the next section for whether the gate exists."
        )
    elif declares:
        print("\n    VERDICT: declares the marker itself; the shim's patch is a no-op.")
    elif not honours:
        print(
            "\n    VERDICT: no disable_tp=use_data_parallel found. The tower may not\n"
            "    support DP at all — TP=6 would need the vision dims padded too,\n"
            "    which this shim does not do. Stop and re-read multimodal.py."
        )


def section_seams() -> None:
    _print_header("shim attachment points")

    checks = [
        (
            "vllm.model_executor.model_loader.default_loader",
            ["DefaultModelLoader._get_weights_iterator", "DefaultModelLoader.get_all_weights"],
            True,
        ),
        (
            "vllm.model_executor.model_loader.loader",
            ["DefaultModelLoader._get_weights_iterator"],
            False,  # pre-0.9 layout; absent on any current build, and that is fine
        ),
        (
            "vllm.model_executor.layers.vocab_parallel_embedding",
            ["pad_vocab_size", "vocab_range_from_global_vocab_size"],
            True,
        ),
    ]
    for module_name, attrs, required in checks:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            verdict = "NOT IMPORTABLE" if required else "absent (optional fallback, fine)"
            print(f"  {module_name}: {verdict} ({type(exc).__name__})")
            continue
        print(f"  {module_name}: ok")
        for attr in attrs:
            obj = module
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            print(f"      {attr:48s} {'present' if obj is not None else 'MISSING'}")

    # Other modules that captured pad_vocab_size by value at import time.
    try:
        vpe = importlib.import_module("vllm.model_executor.layers.vocab_parallel_embedding")
        original = getattr(vpe, "pad_vocab_size", None)
    except Exception:
        original = None
    if original is not None:
        sites = [
            name
            for name, module in list(sys.modules.items())
            if module is not None
            and module is not vpe
            and getattr(module, "pad_vocab_size", None) is original
        ]
        print(f"\n  pad_vocab_size re-import sites currently loaded: {sites or 'none'}")
        print("  (the shim rebinds these too, but only those imported by then)")


def section_encoder_dp_gate(root: str) -> None:
    """Does vLLM core gate --mm-encoder-tp-mode=data on a model-class marker?

    The glm5next tower honours the mode correctly (tp_size=1 + disable_tp on
    every vision linear) but never declares ``supports_encoder_tp_data``. If the
    core checks for that marker, the request is downgraded to ``weights`` and the
    16-head tower shards into ``divide(16, 6)``. The shim sets the marker; this
    section is how you confirm the gate is the reason it has to.
    """
    _print_header("encoder-DP gate in vLLM core")
    wanted = re.compile(r"supports_encoder_tp_data|mm_encoder_tp_mode")
    scopes = ["config", "multimodal", "model_executor/models/interfaces.py", "v1/worker"]
    hits = 0
    for scope in scopes:
        target = os.path.join(root, scope)
        paths = [target] if os.path.isfile(target) else list(_walk_py(target)) if os.path.isdir(target) else []
        for path in paths:
            for lineno, line in enumerate(_read(path).splitlines(), 1):
                if wanted.search(line):
                    hits += 1
                    if hits <= 40:
                        print(f"    {os.path.relpath(path, root)}:{lineno}: {line.strip()[:110]}")
    if hits > 40:
        print(f"    ... {hits - 40} more")
    if not hits:
        print("  no gate found — the mode is probably passed straight through.")
    print(
        "\n  Read for: a line that resets mm_encoder_tp_mode to 'weights' when the\n"
        "  model class lacks supports_encoder_tp_data. If it exists, the shim's\n"
        "  patch_encoder_dp is load-bearing; if not, it is a harmless no-op."
    )


def section_sitecustomize() -> None:
    _print_header("sitecustomize collisions")
    found = []
    for entry in sys.path:
        if not entry:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            found.append(candidate)
    if not found:
        print("  none — our PYTHONPATH copy will be the only one")
    for path in found:
        print(f"  {path}")
    if len(found) > 1:
        print("  >1 found: sitecustomize.py chain-loads the others, verify the order")


def section_plan(tp: int) -> None:
    _print_header(f"padding plan for TP={tp}")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import glm53_tp_pad
    except Exception as exc:
        print(f"  glm53_tp_pad not importable: {exc}")
        return
    plan = glm53_tp_pad.plan_for_tp(tp)
    print(f"  active: {plan.active}")
    print(f"  groups: {','.join(sorted(plan.groups))} (default)")
    off = sorted(set(glm53_tp_pad.ALL_GROUPS) - plan.groups)
    if off:
        print(f"  off:    {','.join(off)} — not TP-sharded in this build")
    print(f"  {plan.summary()}")
    print("\n  rules that will fire:")
    for rule in glm53_tp_pad.build_rules(plan):
        print(
            f"    {rule.group:8s} {rule.what:22s} dim{rule.dim} "
            f"{rule.old} -> {rule.new}  [{rule.mode}]"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", type=int, default=6, help="tensor parallel size")
    args = parser.parse_args()

    section_env()
    root = _vllm_root()
    if root is None:
        print("\nvllm package not found; nothing else to probe.")
        return 1
    print(f"\nvllm package root: {root}")

    files = section_model_files(root)
    section_greps(root, files)
    section_vision(root, files)
    section_seams()
    section_encoder_dp_gate(root)
    section_sitecustomize()
    section_plan(args.tp)
    print(f"\n{SECTION}\nprobe complete — nothing was modified\n{SECTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
