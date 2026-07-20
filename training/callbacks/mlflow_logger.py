"""
DeepGuard — training/callbacks/mlflow_logger.py

MLflow experiment tracking callback module.
"""

import logging
from pathlib import Path
from typing import Any
import mlflow
from training.callbacks.base import Callback

logger = logging.getLogger(__name__)


class MLflowCallback(Callback):
    """Callback for tracking training runs and metrics in MLflow.

    Args:
        experiment_name: MLflow experiment title.
        run_name:        Individual training run tag name.
        params:          Dictionary of hyper-parameters to log on training start.
    """

    def __init__(
        self,
        experiment_name: str = "deepguard-experiments",
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.params = params or {}
        self._active_run: Any = None

    def on_train_begin(self, trainer: Any) -> None:
        try:
            mlflow.set_experiment(self.experiment_name)
            self._active_run = mlflow.start_run(run_name=self.run_name)
            logger.info(
                "Started MLflow run '%s' under experiment '%s'.",
                self.run_name or "default",
                self.experiment_name,
            )
            if self.params:
                # Clean up parameters for standard scalar representation
                flat_params = self._flatten_dict(self.params)
                mlflow.log_params(flat_params)
        except Exception as exc:
            logger.warning("Failed to start MLflow run: %s", exc)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, float]) -> None:
        if self._active_run is None:
            return
        try:
            mlflow.log_metrics(metrics, step=epoch + 1)
        except Exception as exc:
            logger.warning("Failed to log metrics to MLflow at epoch %d: %s", epoch + 1, exc)

    def on_train_end(self, trainer: Any) -> None:
        if self._active_run is None:
            return
        try:
            # Log final best model weight file as an artifact
            best_model_path = Path(trainer.output_dir) / "best_model.pt"
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), artifact_path="model_weights")

            mlflow.end_run()
            logger.info("MLflow logging completed successfully.")
        except Exception as exc:
            logger.warning("Failed to upload artifacts or close MLflow run: %s", exc)
            # Guarantee run termination
            try:
                mlflow.end_run()
            except Exception:
                pass
        finally:
            self._active_run = None

    @staticmethod
    def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        """Utility method to flatten nested configuration dictionaries."""
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowCallback._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
