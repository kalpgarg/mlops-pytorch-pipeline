import csv
import json
from pathlib import Path


class MetricsLogger:
    """Logs training metrics to CSV and JSON for experiment tracking."""

    def __init__(self, log_dir: str, experiment_name: str = "default"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.json_path = self.log_dir / f"{experiment_name}_summary.json"
        self.metrics_history: list[dict] = []
        self._csv_initialized = False

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        lr: float | None = None,
    ) -> None:
        """Log metrics for a single epoch."""
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }
        if lr is not None:
            entry["learning_rate"] = lr
        self.metrics_history.append(entry)
        self._write_csv(entry)

    def _write_csv(self, entry: dict) -> None:
        """Append a single row to the CSV file."""
        file_exists = self.csv_path.exists() and self._csv_initialized
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
                self._csv_initialized = True
            writer.writerow(entry)

    def save_summary(self, best_val_loss: float, best_val_acc: float, total_epochs: int) -> None:
        """Save experiment summary as JSON."""
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": total_epochs,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_accuracy": round(best_val_acc, 4),
            "metrics_csv": str(self.csv_path),
        }
        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)

    def get_best_epoch(self) -> dict | None:
        """Return the epoch with the lowest validation loss."""
        if not self.metrics_history:
            return None
        return min(self.metrics_history, key=lambda x: x["val_loss"])


def get_device() -> str:
    """Detect the best available device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
