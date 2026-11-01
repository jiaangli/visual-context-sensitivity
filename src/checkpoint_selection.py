"""Project-specific checkpoint selection policies."""

from __future__ import annotations

from typing import Any

from transformers.trainer_utils import SaveStrategy


class JointGainPairTieBreakMixin:
    """Select by joint gain first, then pair accuracy on exact ties.

    Hugging Face Trainer only compares ``metric_for_best_model``.  That keeps
    the first checkpoint when two epochs have the same joint gain, even if a
    later checkpoint has a better pair accuracy.  This mixin makes the
    ``eval_joint_gain`` policy lexicographic while preserving Trainer's normal
    behavior for every other selection metric.
    """

    _best_joint_gain_pair: tuple[int | None, float] | None = None

    @staticmethod
    def _metric_key(metric_name: str) -> str:
        return metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"

    @staticmethod
    def _missing_metric_error(metric_key: str, metrics: dict[str, float]) -> KeyError:
        return KeyError(
            "The `metric_for_best_model` training argument uses lexicographic "
            f"joint-gain selection, but '{metric_key}' is not found in the "
            "evaluation metrics. The available evaluation metrics are: "
            f"{list(metrics.keys())}."
        )

    def _previous_best_pair_accuracy(self, pair_key: str) -> float:
        best_step = self.state.best_global_step
        cached = getattr(self, "_best_joint_gain_pair", None)
        if cached is not None and cached[0] == best_step:
            return cached[1]

        # TrainerState serializes log_history, so this also restores the
        # secondary metric when training resumes from a checkpoint.
        for row in reversed(self.state.log_history):
            if row.get("step") == best_step and pair_key in row:
                pair_accuracy = float(row[pair_key])
                self._best_joint_gain_pair = (best_step, pair_accuracy)
                return pair_accuracy

        # Older checkpoints did not persist the secondary selection metric.
        # Let the first exact-primary tie establish it instead of silently
        # retaining a checkpoint whose pair accuracy is unknown.
        return float("-inf")

    def _determine_best_metric(
        self,
        metrics: dict[str, float],
        trial: Any,
    ) -> bool:
        metric_name = self.args.metric_for_best_model
        if metric_name is None or self._metric_key(metric_name) != "eval_joint_gain":
            return super()._determine_best_metric(metrics, trial)

        joint_key = "eval_joint_gain"
        pair_key = "eval_pair_accuracy"
        if joint_key not in metrics:
            raise self._missing_metric_error(joint_key, metrics)
        if pair_key not in metrics:
            raise self._missing_metric_error(pair_key, metrics)

        joint_gain = float(metrics[joint_key])
        pair_accuracy = float(metrics[pair_key])
        previous_joint_gain = self.state.best_metric

        if previous_joint_gain is None:
            is_new_best = True
        else:
            previous_joint_gain = float(previous_joint_gain)
            if self.args.greater_is_better:
                primary_is_better = joint_gain > previous_joint_gain
            else:
                primary_is_better = joint_gain < previous_joint_gain
            primary_is_tied = joint_gain == previous_joint_gain
            is_new_best = primary_is_better or (
                primary_is_tied and pair_accuracy > self._previous_best_pair_accuracy(pair_key)
            )

        if not is_new_best:
            return False

        self.state.best_metric = joint_gain
        if self.args.save_strategy in {
            SaveStrategy.STEPS,
            SaveStrategy.EPOCH,
            SaveStrategy.BEST,
        }:
            self.state.best_global_step = self.state.global_step
        self._best_joint_gain_pair = (
            self.state.best_global_step,
            pair_accuracy,
        )
        return True
