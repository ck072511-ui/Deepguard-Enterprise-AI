"""
DeepGuard — training/evaluators/metrics.py

Model performance metrics computation using scikit-learn.
"""

from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


class MetricsEvaluator:
    """Computes comprehensive binary classification performance metrics.

    Calculates metrics based on ground truth labels and predicted positive class
    probabilities.
    """

    @staticmethod
    def evaluate(
        y_true: list[int] | np.ndarray,
        y_pred_probs: list[float] | np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Compute performance metrics.

        Args:
            y_true:       Ground truth labels (0 = REAL, 1 = FAKE).
            y_pred_probs: Predicted probabilities for the FAKE class.
            threshold:    Classification probability threshold (default: 0.5).

        Returns:
            Dictionary containing accuracy, precision, recall, f1_score,
            auc_roc, confusion_matrix breakdowns, and roc_curve coordinates.
        """
        y_true_arr = np.array(y_true, dtype=np.int64)
        y_probs_arr = np.array(y_pred_probs, dtype=np.float32)

        # Convert probabilities to binary predictions
        y_pred_arr = (y_probs_arr >= threshold).astype(np.int64)

        # Basic scalar metrics
        accuracy = accuracy_score(y_true_arr, y_pred_arr)

        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_true_arr, y_pred_arr, average="binary", zero_division=0
        )

        try:
            auc_roc = roc_auc_score(y_true_arr, y_probs_arr)
        except ValueError:
            # Fallback if only 1 class is present in the labels set
            auc_roc = 0.5

        # Confusion Matrix
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # ROC Curve coordinates
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_probs_arr)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "auc_roc": float(auc_roc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "raw": cm.tolist(),
            },
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
            },
        }
