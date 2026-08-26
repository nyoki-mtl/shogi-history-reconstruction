"""ctypes bindings for YaneuraOu-based returnable checks."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
from pathlib import Path

import numpy as np

_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    root = Path(__file__).resolve().parents[3]
    lib_path = root / "external" / "YaneuraOu" / "build" / "libyaneuraou_returnable.so"
    if not lib_path.exists():
        raise FileNotFoundError(
            f"{lib_path} not found. Run `scripts/build_yaneuraou_bindings.sh` to build the library."
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.yaneuraou_returnable_init.argtypes = []
    lib.yaneuraou_returnable_init.restype = ctypes.c_int
    lib.yaneuraou_returnable_mask.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.yaneuraou_returnable_mask.restype = ctypes.c_int
    lib.yaneuraou_returnable_usi.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
    ]
    lib.yaneuraou_returnable_usi.restype = ctypes.c_int
    lib.yaneuraou_generate_predecessors.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.yaneuraou_generate_predecessors.restype = ctypes.c_int
    if lib.yaneuraou_returnable_init() != 0:
        raise RuntimeError("yaneuraou_returnable_init failed")
    _LIB = lib
    return lib


def returnable_mask(
    predicted_tokens: np.ndarray | Iterable[Iterable[int]],
    source_tokens: np.ndarray | Iterable[Iterable[int]],
    *,
    use_light_checks: bool = True,
) -> np.ndarray:
    pred = np.ascontiguousarray(predicted_tokens, dtype=np.int32)
    src = np.ascontiguousarray(source_tokens, dtype=np.int32)
    if pred.shape != src.shape:
        raise ValueError("predicted_tokens and source_tokens must have the same shape")
    if pred.ndim != 2 or pred.shape[1] != 96:
        raise ValueError("tokens must have shape (batch, 96)")

    out = np.zeros(pred.shape[0], dtype=np.uint8)
    lib = _load_lib()
    ret = lib.yaneuraou_returnable_mask(
        pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(pred.shape[0]),
        ctypes.c_int(1 if use_light_checks else 0),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    if ret != 0:
        raise RuntimeError("yaneuraou_returnable_mask failed")
    return out.astype(bool)


def returnable_moves(
    predicted_tokens: np.ndarray | Iterable[Iterable[int]],
    source_tokens: np.ndarray | Iterable[Iterable[int]],
    *,
    use_light_checks: bool = True,
    stride: int = 8,
) -> tuple[np.ndarray, list[str]]:
    pred = np.ascontiguousarray(predicted_tokens, dtype=np.int32)
    src = np.ascontiguousarray(source_tokens, dtype=np.int32)
    if pred.shape != src.shape:
        raise ValueError("predicted_tokens and source_tokens must have the same shape")
    if pred.ndim != 2 or pred.shape[1] != 96:
        raise ValueError("tokens must have shape (batch, 96)")
    if stride <= 0:
        raise ValueError("stride must be positive")

    lib = _load_lib()
    buf = (ctypes.c_char * (pred.shape[0] * stride))()
    ret = lib.yaneuraou_returnable_usi(
        pred.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(pred.shape[0]),
        ctypes.c_int(1 if use_light_checks else 0),
        ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)),
        ctypes.c_int(stride),
    )
    if ret != 0:
        raise RuntimeError("yaneuraou_returnable_usi failed")

    moves: list[str] = []
    for idx in range(pred.shape[0]):
        raw = bytes(buf[idx * stride : (idx + 1) * stride]).split(b"\0", 1)[0]
        moves.append(raw.decode("ascii"))
    mask = np.asarray([bool(mv) for mv in moves], dtype=bool)
    return mask, moves


def predecessor_candidates(
    source_tokens: np.ndarray | Iterable[int],
    *,
    max_candidates: int = 8192,
    stride: int = 8,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    src = np.ascontiguousarray(source_tokens, dtype=np.int32)
    if src.ndim != 1 or src.shape[0] != 96:
        raise ValueError("source_tokens must have shape (96,)")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    out_tokens = np.zeros((max_candidates, 96), dtype=np.int32)
    usi_buf = (ctypes.c_char * (max_candidates * stride))()
    out_kind = np.zeros(max_candidates, dtype=np.int32)
    out_from = np.zeros(max_candidates, dtype=np.int32)
    out_to = np.zeros(max_candidates, dtype=np.int32)
    out_moved = np.zeros(max_candidates, dtype=np.int32)
    out_captured = np.zeros(max_candidates, dtype=np.int32)
    out_promoted = np.zeros(max_candidates, dtype=np.uint8)
    out_count = ctypes.c_int(0)

    lib = _load_lib()
    ret = lib.yaneuraou_generate_predecessors(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(max_candidates),
        out_tokens.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.cast(usi_buf, ctypes.POINTER(ctypes.c_char)),
        ctypes.c_int(stride),
        out_kind.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_from.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_to.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_moved.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_captured.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        out_promoted.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.byref(out_count),
    )
    if ret == -2:
        raise RuntimeError(f"yaneuraou_generate_predecessors exceeded max_candidates={max_candidates}")
    if ret != 0:
        raise RuntimeError("yaneuraou_generate_predecessors failed")

    count = out_count.value
    moves: list[str] = []
    for idx in range(count):
        raw = bytes(usi_buf[idx * stride : (idx + 1) * stride]).split(b"\0", 1)[0]
        moves.append(raw.decode("ascii"))

    metadata = {
        "kind": out_kind[:count].copy(),
        "from_square": out_from[:count].copy(),
        "to_square": out_to[:count].copy(),
        "moved_piece": out_moved[:count].copy(),
        "captured_piece": out_captured[:count].copy(),
        "promoted": out_promoted[:count].astype(bool),
    }
    return out_tokens[:count].copy(), moves, metadata
