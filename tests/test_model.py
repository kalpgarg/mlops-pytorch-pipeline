import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import get_model


def test_resnet18_output_shape():
    """Test that ResNet-18 model outputs correct shape for CIFAR-10."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 10), f"Expected (1, 10), got {output.shape}"


def test_resnet18_batch():
    """Test that model handles batch input correctly."""
    model = get_model(architecture="resnet18", num_classes=10)
    model.eval()
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (8, 10), f"Expected (8, 10), got {output.shape}"


def test_invalid_architecture():
    """Test that invalid architecture raises ValueError."""
    try:
        get_model(architecture="invalid_model", num_classes=10)
        assert False, "Expected ValueError"
    except ValueError:
        pass
