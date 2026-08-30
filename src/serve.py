import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import get_model

app = FastAPI(title="CIFAR-10 Model Serving API")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Global model reference
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inference_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def load_model():
    global model
    checkpoint_path = os.environ.get(
        "MODEL_CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt"
    )

    if not Path(checkpoint_path).exists():
        print(f"Warning: checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = get_model(architecture="simplecnn", num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/health")
async def health():
    if model is not None:
        return {"status": "healthy"}
    return JSONResponse(status_code=503, content={"status": "model not loaded"})


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503, content={"error": "Model not loaded"}
        )

    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = get_inference_transform()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    results = {
        CIFAR10_CLASSES[i]: round(probabilities[i].item(), 4)
        for i in range(len(CIFAR10_CLASSES))
    }

    predicted_class = CIFAR10_CLASSES[probabilities.argmax().item()]

    return {
        "predicted_class": predicted_class,
        "probabilities": results,
    }
