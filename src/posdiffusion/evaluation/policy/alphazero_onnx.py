"""Simplified AlphaZero ONNX model loader for policy guidance."""

from pathlib import Path

import numpy as np
import onnxruntime  # type: ignore[import-untyped]
from numpy.typing import NDArray
from rsshogi.core import Board
from rsshogi.policy import move_label

from .dlshogi_features import FEATURES1_NUM, FEATURES2_NUM, make_input_features


class AlphaZeroONNX:
    """Simplified ONNX-based AlphaZero model loader.

    This class loads and runs inference on dlshogi-format ONNX models.
    """

    def __init__(self, model_path: str | Path, device: str = "cpu", intra_op_num_threads: int = 4):
        """Initialize ONNX model.

        Args:
            model_path: Path to ONNX model file
            device: Device to run on ("cpu" or "cuda")
            intra_op_num_threads: CPU threads per inference session. The ONNX
                Runtime default spawns one spinning thread per core, which
                collapses under many concurrent evaluation processes; the small
                dlshogi net with small batches needs only a few threads.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Initialize ONNX session
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = intra_op_num_threads
        session_options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(str(model_path), sess_options=session_options, providers=providers)

        # Policy output size (27 move labels × 81 squares)
        self.policy_size = 27 * 81

    def get_policy_value(self, board: Board) -> tuple[NDArray[np.float32], float]:
        """Get policy and value for a board position.

        Args:
            board: rsshogi Board object

        Returns:
            Tuple of (policy_probs, value)
            - policy_probs: Probability distribution over moves [2187]
            - value: Position evaluation from current player's perspective
        """
        # Extract features
        x1 = np.empty((1, FEATURES1_NUM, 9, 9), dtype=np.float32)
        x2 = np.empty((1, FEATURES2_NUM, 9, 9), dtype=np.float32)
        make_input_features(board, x1[0], x2[0])

        # Run inference
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input("input1", x1)
        io_binding.bind_cpu_input("input2", x2)
        io_binding.bind_output("output_policy")
        io_binding.bind_output("output_value")

        self.session.run_with_iobinding(io_binding)
        policy_logits, values = io_binding.copy_outputs_to_cpu()

        # Convert logits to probabilities
        policy_probs = self._softmax(policy_logits[0])
        value = float(values[0])

        return policy_probs, value

    def get_policy_value_batch(self, boards: list[Board]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Get policies and values for multiple positions.

        Args:
            boards: List of rsshogi Board objects

        Returns:
            Tuple of (policy_probs_batch, values_batch)
            - policy_probs_batch: Shape [batch_size, 2187]
            - values_batch: Shape [batch_size]
        """
        batch_size = len(boards)

        # Extract features for all boards
        x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
        x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)

        for i, board in enumerate(boards):
            make_input_features(board, x1[i], x2[i])

        # Run inference
        io_binding = self.session.io_binding()
        io_binding.bind_cpu_input("input1", x1)
        io_binding.bind_cpu_input("input2", x2)
        io_binding.bind_output("output_policy")
        io_binding.bind_output("output_value")

        self.session.run_with_iobinding(io_binding)
        policy_logits, values = io_binding.copy_outputs_to_cpu()

        # Convert logits to probabilities
        policy_probs = np.array([self._softmax(logits) for logits in policy_logits])

        return policy_probs, values.flatten()

    def _softmax(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply softmax to logits."""
        exp_x = np.exp(x - np.max(x))
        result = (exp_x / exp_x.sum()).astype(np.float32)
        return result  # type: ignore[no-any-return]

    def get_legal_move_probs(self, board: Board) -> dict[str, float]:
        """Get probabilities for legal moves only.

        Args:
            board: rsshogi Board object

        Returns:
            Dictionary mapping USI move string to probability
        """
        policy_probs, _ = self.get_policy_value(board)

        # Extract probabilities for legal moves
        legal_move_probs: dict[str, float] = {}
        for move in board.legal_moves():
            # Convert rsshogi move to dlshogi policy index
            move_idx = move_label(move, board.turn)
            legal_move_probs[move.to_usi()] = float(policy_probs[move_idx])

        # Renormalize
        total = sum(legal_move_probs.values())
        if total > 0:
            for move_key in legal_move_probs:
                legal_move_probs[move_key] /= total

        return legal_move_probs
