import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import MetricsLogger, get_device


def test_metrics_logger_creates_files():
    """Test that MetricsLogger creates CSV and JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, experiment_name="test_exp")
        logger.log_epoch(
            epoch=1,
            train_loss=1.5,
            train_acc=0.45,
            val_loss=1.8,
            val_acc=0.40,
            lr=0.001,
        )
        assert logger.csv_path.exists(), "CSV file not created"

        logger.save_summary(best_val_loss=1.8, best_val_acc=0.40, total_epochs=1)
        assert logger.json_path.exists(), "JSON summary not created"


def test_metrics_logger_best_epoch():
    """Test that get_best_epoch returns the epoch with lowest val_loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, experiment_name="test_best")
        logger.log_epoch(1, 1.5, 0.45, 1.8, 0.40)
        logger.log_epoch(2, 1.2, 0.55, 1.3, 0.50)
        logger.log_epoch(3, 1.0, 0.60, 1.5, 0.48)

        best = logger.get_best_epoch()
        assert best is not None
        assert best["epoch"] == 2, f"Expected epoch 2, got {best['epoch']}"


def test_metrics_logger_empty():
    """Test that get_best_epoch returns None when no metrics logged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(log_dir=tmpdir, experiment_name="empty")
        assert logger.get_best_epoch() is None


def test_get_device_returns_string():
    """Test that get_device returns a valid device string."""
    device = get_device()
    assert device in ("cpu", "cuda", "mps"), f"Unexpected device: {device}"
