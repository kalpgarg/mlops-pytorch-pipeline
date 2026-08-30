from torch import nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Lightweight CNN for CIFAR-10 (~62K params). Trains fast on CPU."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(architecture: str = "simplecnn", num_classes: int = 10) -> nn.Module:
    """Factory function to create a model for CIFAR-10 classification.

    Args:
        architecture: Model architecture name ('simplecnn' or 'resnet18').
        num_classes: Number of output classes.

    Returns:
        A PyTorch model ready for CIFAR-10 (32x32 input).
    """
    if architecture == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif architecture == "resnet18":
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
