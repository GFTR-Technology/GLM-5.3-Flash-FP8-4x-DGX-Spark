#!/usr/bin/env bash
# IN-CONTAINER ENTRYPOINT — pick a RoCEv2 IPv4 GID, then exec vllm.
# Bind-mounted read-only from the host. Official image ENTRYPOINT is
# ["vllm","serve"]; we replace it with this wrapper and pass `vllm serve ...`.
set -eu

HCA="${NCCL_IB_HCA%%,*}"
if [ -n "${HCA:-}" ] && [ -d "/sys/class/infiniband/$HCA/ports/1" ]; then
  for i in $(seq 0 15); do
    t=$(cat "/sys/class/infiniband/$HCA/ports/1/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$HCA/ports/1/gids/$i" 2>/dev/null || true)
    case "$t" in
      *"RoCE v2"*)
        case "$g" in
          *"0000:0000:0000:0000:0000:ffff:"*)
            export NCCL_IB_GID_INDEX=$i
            echo "[glm53-entrypoint] HCA=$HCA NCCL_IB_GID_INDEX=$i gid=$g"
            break
            ;;
        esac
        ;;
    esac
  done
fi
if [ -z "${NCCL_IB_GID_INDEX:-}" ]; then
  echo "[glm53-entrypoint] WARNING: no RoCEv2 IPv4 GID for HCA=$HCA; NCCL will auto-select" >&2
fi

# TP that does not divide the head counts (TP=6): hand vLLM a symlink overlay of
# /model whose config.json carries the padded dims. The weights themselves are
# padded as they stream in, by the shim on PYTHONPATH. See docs/TP6.md.
if [ -n "${GLM53_TP_PAD:-}" ] && [ "${GLM53_TP_PAD}" != 1 ] && [ -n "${GLM53_MODEL_DIR:-}" ]; then
  echo "[glm53-entrypoint] building TP=${GLM53_TP_PAD} config overlay at ${GLM53_MODEL_DIR}"
  python3 /opt/glm53/make_overlay.py \
    --src "${GLM53_MODEL_SRC:-/model}" \
    --dst "${GLM53_MODEL_DIR}" \
    --tp "${GLM53_TP_PAD}"
fi

exec "$@"
