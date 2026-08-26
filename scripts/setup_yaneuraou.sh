#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YO_DIR="${ROOT_DIR}/external/YaneuraOu"
YO_REVISION="014036fd2e5012693f2fe28e7a7cbd168a6e5b90"
PATCH_FILE="${ROOT_DIR}/third_party/yaneuraou-posdiffusion.patch"

if [[ -e "${YO_DIR}" ]]; then
  echo "Refusing to overwrite existing path: ${YO_DIR}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/external"
git clone https://github.com/yaneurao/YaneuraOu.git "${YO_DIR}"
git -C "${YO_DIR}" checkout --detach "${YO_REVISION}"
git -C "${YO_DIR}" apply --check "${PATCH_FILE}"
git -C "${YO_DIR}" apply "${PATCH_FILE}"
"${ROOT_DIR}/scripts/build_yaneuraou_bindings.sh"
