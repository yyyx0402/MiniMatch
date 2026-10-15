"""Step-based stopping with denser early validation and exact checkpoint selection."""
from dataclasses import dataclass
import math


def validation_steps(config):
    maximum = config["max_optimizer_steps"]
    boundary = config["dense_validation_until"]
    early = config["dense_validation_interval"]
    later = config["later_validation_interval"]
    if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
           for x in (maximum, boundary, early, later)):
        raise ValueError("Validation steps and intervals must be positive integers")
    if boundary > maximum or boundary % early or boundary % later:
        raise ValueError("Validation boundary must align with both intervals and the budget")
    return sorted(set(range(early, boundary + 1, early))
                  | set(range(boundary + later, maximum + 1, later)) | {maximum})


@dataclass
class StepEarlyStopping:
    """Patience uses actual optimizer updates, independently of validation cadence.

    All-time raw best chooses the checkpoint. A separate significant-improvement
    reference governs patience; cumulative small gains can therefore reset it.
    The warmup period never consumes the non-improvement window.
    """
    baseline: float
    warmup_steps: int = 24
    minimum_training_steps: int = 120
    window_steps: int = 120
    minimum_improvement: float = .001

    def __post_init__(self):
        if not math.isfinite(self.baseline) or not 0 <= self.baseline <= 1:
            raise ValueError("Baseline Recall must be finite and in [0, 1]")
        if not 0 < self.minimum_improvement < 1:
            raise ValueError("Improvement threshold must be in (0, 1)")
        if (self.warmup_steps < 0 or self.window_steps <= 0
                or self.minimum_training_steps < self.warmup_steps):
            raise ValueError("Invalid warmup, minimum training or stopping window")
        self.best = float(self.baseline)
        self.best_step = 0
        self.reference = float(self.baseline)
        self.last_significant_step = self.warmup_steps
        self.last_validation_step = 0
        self.stopped = False

    def update(self, score, step):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Validation Recall must be finite and in [0, 1]")
        if not isinstance(step, int) or isinstance(step, bool) or step <= self.last_validation_step:
            raise ValueError("Validation steps must strictly increase")
        if self.stopped:
            raise ValueError("Cannot update a stopped trial")
        best_changed = score > self.best
        if best_changed:
            self.best, self.best_step = score, step
        significant = score > self.reference + self.minimum_improvement
        if significant:
            self.reference = score
            self.last_significant_step = max(step, self.warmup_steps)
        self.last_validation_step = step
        gap = max(0, step - self.last_significant_step)
        self.stopped = step >= self.minimum_training_steps and gap >= self.window_steps
        return {"checkpoint_best_changed": best_changed, "significant_improvement": significant,
                "reference_recall_at_100": self.reference,
                "last_significant_improvement_step": self.last_significant_step,
                "steps_without_significant_improvement": gap, "stop": self.stopped}

    def selection(self):
        eligible = self.best_step > 0 and self.best > self.baseline + self.minimum_improvement
        return {"source": "finetuned" if eligible else "pretrained",
                "step": self.best_step if eligible else 0,
                "recall_at_100": self.best if eligible else float(self.baseline),
                "best_observed_step": self.best_step, "best_observed_recall_at_100": self.best}
