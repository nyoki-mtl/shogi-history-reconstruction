#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YO_DIR="${ROOT_DIR}/external/YaneuraOu"
BUILD_DIR="${YO_DIR}/build"

mkdir -p "${BUILD_DIR}"

CXX="${CXX:-clang++}"
CXXFLAGS=(
  -O3
  -std=c++17
  -fPIC
  -DNDEBUG
  -DPOSDIFFUSION_MINIMAL
  -fno-exceptions
  -fno-rtti
  -Wno-unused-parameter
  "-I${YO_DIR}/source"
)

SOURCES=(
  "${YO_DIR}/posdiffusion_returnable.cpp"
  "${YO_DIR}/posdiffusion_types_min.cpp"
  "${YO_DIR}/source/bitboard.cpp"
  "${YO_DIR}/source/misc.cpp"
  "${YO_DIR}/source/memory.cpp"
  "${YO_DIR}/source/thread.cpp"
  "${YO_DIR}/source/position.cpp"
  "${YO_DIR}/source/movegen.cpp"
  "${YO_DIR}/source/usioption.cpp"
  "${YO_DIR}/posdiffusion_unit_test_min.cpp"
  "${YO_DIR}/posdiffusion_usi_min.cpp"
  "${YO_DIR}/source/book/book.cpp"
  "${YO_DIR}/source/book/apery_book.cpp"
  "${YO_DIR}/source/book/policybook.cpp"
  "${YO_DIR}/source/score.cpp"
  "${YO_DIR}/source/mate/mate.cpp"
  "${YO_DIR}/source/eval/evaluate.cpp"
)

"${CXX}" "${CXXFLAGS[@]}" -shared "${SOURCES[@]}" -o "${BUILD_DIR}/libyaneuraou_returnable.so"

echo "Built ${BUILD_DIR}/libyaneuraou_returnable.so"
