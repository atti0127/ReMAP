#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_URL=https://raw.githubusercontent.com/zqhang/AnomalyCLIP/main/checkpoints

download() {
  local relative_path=$1
  local checksum=$2
  local target=${REPO_ROOT}/checkpoints/${relative_path}
  mkdir -p "$(dirname "${target}")"
  if [[ ! -f "${target}" ]]; then
    curl --fail --location --retry 3 \
      "${BASE_URL}/${relative_path}" --output "${target}"
  fi
  printf '%s  %s\n' "${checksum}" "${target}" | sha256sum --check --status || {
    echo "checkpoint checksum failed: ${target}" >&2
    exit 1
  }
  echo "ready: ${target}"
}

download \
  9_12_4_multiscale/epoch_15.pth \
  94ce202da3e6486a864b904fdfed5057de75846c5834e446fd1d2fe7f97acb44
download \
  9_12_4_multiscale_visa/epoch_15.pth \
  415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4

