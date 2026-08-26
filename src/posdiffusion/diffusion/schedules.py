"""Distance-based schedule for data-driven diffusion model."""

from typing import Any


class DistanceSchedule:
    """Distance-based schedule for curriculum learning in data-driven diffusion.

    Centralizes distance sampling configuration and provides a clean API
    for curriculum learning across training, evaluation, and inference.
    """

    def __init__(
        self,
        num_steps: int,
        distances: list[int] | None = None,
        weights: list[float] | None = None,
        curriculum_schedule: dict[int, list[float]] | None = None,
    ):
        """Initialize distance schedule.

        Args:
            num_steps: Maximum number of timesteps (distances)
            distances: Available distance values for sampling
            weights: Default sampling weights for each distance
            curriculum_schedule: Epoch-based weight schedule for curriculum learning
                Format: {epoch: [weight1, weight2, ...]}
        """
        self.num_steps = num_steps
        self.distances = distances or [1, 2, 5, 10, 20]
        self.weights = weights or [0.2] * len(self.distances)
        if curriculum_schedule is None:
            self.curriculum_schedule = {
                0: [0.7, 0.2, 0.1, 0.0, 0.0],  # epoch 0-1: short distances
                2: [0.3, 0.3, 0.2, 0.1, 0.1],  # epoch 2-3: medium distances
                4: [0.2, 0.2, 0.2, 0.2, 0.2],  # epoch 4+: uniform
            }
        else:
            self.curriculum_schedule = curriculum_schedule

        # Initialize current epoch
        self.current_epoch = 0

        # Validation
        if len(self.weights) != len(self.distances):
            raise ValueError("weights and distances must have same length")

        # Only validate if curriculum_schedule is not empty
        if self.curriculum_schedule:
            for epoch_weights in self.curriculum_schedule.values():
                if len(epoch_weights) != len(self.distances):
                    raise ValueError("All curriculum schedule weights must match distances length")

    def update_epoch(self, epoch: int) -> None:
        """Update the current epoch for curriculum learning.

        Args:
            epoch: Current training epoch
        """
        # Store current epoch for internal use
        self.current_epoch = epoch

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DistanceSchedule":
        """Create from configuration dictionary.

        Args:
            config: Configuration with keys: num_steps, distances, weights, curriculum_schedule

        Returns:
            DistanceSchedule instance
        """
        return cls(
            num_steps=config["num_steps"],
            distances=config.get("distances"),
            weights=config.get("weights"),
            curriculum_schedule=config.get("curriculum_schedule"),
        )

    def get_weights_for_epoch(self, epoch: int) -> list[float]:
        """Get sampling weights for specific epoch based on curriculum.

        Args:
            epoch: Current training epoch

        Returns:
            List of weights for each distance
        """
        # Find the most recent applicable epoch
        applicable_epochs = [e for e in sorted(self.curriculum_schedule.keys()) if e <= epoch]
        if applicable_epochs:
            return list(self.curriculum_schedule[applicable_epochs[-1]])
        return list(self.weights)

    def sample_distance(self, rng: None = None) -> int:
        """Sample a distance based on current epoch curriculum.

        Args:
            rng: Random number generator (unused, for API compatibility)

        Returns:
            Sampled distance value
        """
        import random

        weights = self.get_weights_for_epoch(self.current_epoch)
        return random.choices(self.distances, weights=weights, k=1)[0]

    def __len__(self) -> int:
        """Return number of timesteps."""
        return self.num_steps

    def __repr__(self) -> str:
        return (
            f"DistanceSchedule(num_steps={self.num_steps}, "
            f"distances={self.distances}, "
            f"curriculum_epochs={sorted(self.curriculum_schedule.keys())})"
        )
