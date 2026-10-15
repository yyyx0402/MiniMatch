"""Restored 10/30-step validation with fixed 120-update patience."""
from .dense_v3 import StepEarlyStopping, validation_steps as staged_steps


def validation_steps(config):
    expected = {"warmup_steps": 30, "dense_validation_until": 120, "dense_validation_interval": 10,
                "later_validation_interval": 30, "minimum_training_steps": 120,
                "early_stopping_window_steps": 120, "early_stopping_starts_at": "after_warmup"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Use restored early 10-step / later 30-step checks and 120-update patience")
    if config["max_optimizer_steps"] < 150 or config["max_optimizer_steps"] % 30:
        raise ValueError("Budget must allow a complete post-warmup patience window")
    return staged_steps(config)


class StagedRecallStopping(StepEarlyStopping):
    def __init__(self, baseline, warmup_steps=30, minimum_training_steps=120,
                 window_steps=120, minimum_improvement=.001):
        if (warmup_steps, minimum_training_steps, window_steps) != (30, 120, 120):
            raise ValueError("Use 30 warmup steps and a fixed 120-update patience window")
        super().__init__(baseline, warmup_steps=30, minimum_training_steps=120,
                         window_steps=120, minimum_improvement=minimum_improvement)

    def update(self, score, step):
        if step % (10 if step <= 120 else 30):
            raise ValueError("Validation must follow the fixed early 10 / later 30 cadence")
        result = super().update(score, step)
        result["patience_reset_reason"] = "significant_improvement" if result["significant_improvement"] else None
        if step == self.warmup_steps:
            self.reference = self.best
            self.last_significant_step = step
            result.update(reference_recall_at_100=self.reference,
                          last_significant_improvement_step=step,
                          steps_without_significant_improvement=0,
                          patience_reset_reason="warmup_completed")
        result["early_stopping_active"] = step > self.warmup_steps
        result["stopping_allowed"] = step >= self.minimum_training_steps
        return result
