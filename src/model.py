import torch.nn as nn
from torchvision import models


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Factory function to create a model for CIFAR-10 classification.

    Args:
        architecture: Model architecture name (currently supports 'resnet18').
        num_classes: Number of output classes.

    Returns:
        A PyTorch model ready for CIFAR-10 (32x32 input).
    """
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        # Adapt for CIFAR-10 32x32 images (original ResNet uses 7x7 conv + maxpool for 224x224)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.bn1 = nn.BatchNorm2d(64)
        model.maxpool = nn.Identity()
        # Replace final FC layer for num_classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
