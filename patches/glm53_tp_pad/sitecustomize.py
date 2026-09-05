"""Injected into every rank via ``PYTHONPATH=/opt/glm53``.

vLLM's mp executor starts each worker as a fresh interpreter, and ``site``
imports ``sitecustomize`` in every one of them — that is why the padding shim is
installed from here rather than from a wrapper around ``vllm serve``.

This file must stay cheap: it runs at interpreter start for *every* Python
process in the container, so it imports neither torch nor vllm. All it does is
register a post-import hook that calls into ``glm53_tp_pad`` once the relevant
vLLM module has finished executing.
"""

import os
import sys

_ENABLED = os.environ.get("GLM53_TP_PAD", "").strip() not in ("", "0", "1")
_PREFIX = "[glm53-tp-pad]"


def _chain_previous_sitecustomize():
    """Run the image's own sitecustomize, if it had one.

    We sit ahead of site-packages on sys.path, so importing us shadows it.
    """
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if not os.path.isfile(candidate):
            continue
        if os.path.dirname(os.path.abspath(candidate)) == here:
            continue
        spec = importlib.util.spec_from_file_location(
            "_glm53_chained_sitecustomize", candidate
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - image specific
            print(
                f"{_PREFIX} WARNING: chained sitecustomize {candidate} failed: {exc}",
                file=sys.stderr,
            )
        return


def _fire(module_name, module):
    import glm53_tp_pad

    hook = glm53_tp_pad.HOOKS.get(module_name)
    if hook is None:
        return
    try:
        hook(module)
    except Exception:
        # Failing here beats booting unpadded and dying 12 minutes later inside
        # a divide() assertion with no context.
        print(
            f"{_PREFIX} FATAL: patching {module_name} failed", file=sys.stderr
        )
        raise


class _PostImportHook:
    """Standard meta-path trick: wrap the real loader's exec_module so we run
    right after the target module finishes executing."""

    def __init__(self, targets):
        self.targets = set(targets)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.targets:
            return None
        import importlib.util

        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except (ImportError, AttributeError, ValueError):
            spec = None
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None

        original = getattr(spec.loader, "exec_module", None)
        if original is None:
            return None

        def exec_module(module, _original=original, _name=fullname):
            _original(module)
            _fire(_name, module)

        spec.loader.exec_module = exec_module
        return spec


def _install():
    try:
        import glm53_tp_pad
    except Exception as exc:  # pragma: no cover - mount problem
        print(
            f"{_PREFIX} FATAL: GLM53_TP_PAD is set but glm53_tp_pad is not "
            f"importable from PYTHONPATH ({exc})",
            file=sys.stderr,
        )
        raise
    sys.meta_path.insert(0, _PostImportHook(glm53_tp_pad.HOOKS))
    # Covers the case where vLLM was somehow already imported.
    glm53_tp_pad.install()


_chain_previous_sitecustomize()
if _ENABLED:
    _install()
