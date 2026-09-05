#!/usr/bin/env bash
# Read-only probe of the vLLM build inside the image, for a target TP.
#
# The glm5next implementation in this image is a custom drop; the padding shim's
# attachment points cannot be verified from a dev box. Run this once before the
# first TP!=4 boot and read the report — it tells you whether the indexer is
# TP-sharded at all, whether the vision tower can host whole weights per rank,
# and whether the loader seams the shim hooks still exist.
#
#   ./scripts/glm53-tp-probe.sh                 # head node, TP = node count
#   GLM53_PROBE_TP=6 ./scripts/glm53-tp-probe.sh
#   GLM53_PROBE_ALL=1 ./scripts/glm53-tp-probe.sh   # every node, not just head
#
# Starts a throwaway container with no GPU claim, so it is safe to run while an
# engine is live.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$HERE/lib.sh"
require_cluster

IMAGE="${GLM53_IMAGE:-$IMAGE}"
PATCH_HOST="${GLM53_PATCHES:-$REPO_ROOT/patches}"
TP="${GLM53_PROBE_TP:-$NNODES}"

if [ ! -f "$PATCH_HOST/glm53_tp_pad/probe.py" ]; then
  echo "Error: $PATCH_HOST/glm53_tp_pad/probe.py not found" >&2
  exit 1
fi

probe_one() {
  local ip="$1"
  echo
  echo "############ $ip ############"
  ssh_to "$ip" "docker run --rm \
      -v $PATCH_HOST/glm53_tp_pad:/opt/glm53:ro \
      --entrypoint python3 $IMAGE /opt/glm53/probe.py --tp $TP" \
    || echo "  probe failed on $ip"
}

if [ "${GLM53_PROBE_ALL:-0}" = 1 ]; then
  for ip in "${NODES[@]}"; do probe_one "$ip"; done
else
  probe_one "$HEAD_IP"
fi

echo
echo "Local plan for TP=$TP (no container needed):"
python3 "$PATCH_HOST/glm53_tp_pad/glm53_tp_pad.py" --tp "$TP"
