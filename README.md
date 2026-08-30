# MLOps PyTorch Pipeline

A production-style ML pipeline for training and serving a PyTorch CIFAR-10 image classifier using Docker and Kubernetes.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GitHub Repository                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │ src/     │  │ configs/ │  │ docker/  │  │ k8s/   │  │
│  │ model.py │  │ training │  │ Train    │  │ Job    │  │
│  │ train.py │  │ _config  │  │ Serve    │  │ Deploy │  │
│  │ dataset  │  │ .yaml    │  │ Dockerf. │  │ Svc    │  │
│  │ serve.py │  │          │  │          │  │ HPA    │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
└─────────────────────────────────────────────────────────┘
        │                            │              │
        ▼                            ▼              ▼
  ┌───────────┐              ┌─────────────┐  ┌──────────┐
  │  CI/CD    │              │   Docker     │  │ K8s      │
  │  (Lint +  │              │  Training &  │  │ Training │
  │   Test)   │              │  Serving     │  │ Job +    │
  └───────────┘              │  Images      │  │ Serving  │
                             └─────────────┘  │ Deploy   │
                                    │         └──────────┘
                                    ▼              │
                             ┌─────────────┐       │
                             │  Trained     │◄──────┘
                             │  Model       │
                             │  Checkpoint  │
                             └─────────────┘
                                    │
                                    ▼
                             ┌─────────────┐
                             │  Serving API │
                             │  /predict    │
                             │  /health     │
                             └─────────────┘
```

## Project Structure

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/workflows/ci.yml      # CI: lint + test
├── src/
│   ├── model.py                   # ResNet-18 model for CIFAR-10
│   ├── dataset.py                 # CIFAR-10 data loading
│   ├── train.py                   # Training loop with early stopping
│   └── serve.py                   # FastAPI serving API
├── configs/
│   └── training_config.yaml       # Training hyperparameters
├── docker/
│   ├── Dockerfile.train           # Multi-stage training image
│   └── Dockerfile.serve           # Serving image (non-root)
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
├── requirements/
│   ├── train.txt                  # Training dependencies
│   └── serve.txt                  # Inference dependencies
└── tests/
    └── test_model.py
```

## Setup

### Prerequisites

- Python 3.10+
- Docker Desktop
- kubectl CLI
- A Kubernetes cluster (Minikube, kind, or cloud-managed)

### Local Development

```bash
# Install training dependencies
pip install -r requirements/train.txt

# Run training locally
cd src && python train.py

# Run tests
pip install pytest
pytest tests/ -v
```

### Docker

```bash
# Build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# Run training with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# Build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# Run serving
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# Test prediction endpoint
curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

### Kubernetes Deployment

```bash
# Apply namespace and config
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Run training job
kubectl apply -f k8s/training-job.yaml

# Deploy serving (after training completes)
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# Verify
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# Test locally via port-forward
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Model

- **Architecture:** SimpleCNN (3-layer CNN, ~62K params) — lightweight, trains fast on CPU. ResNet-18 also available via config.
- **Dataset:** CIFAR-10 (10 classes)
- **Training:** Adam optimizer, CrossEntropy loss, ReduceLROnPlateau scheduler, early stopping
- **Serving:** FastAPI with POST /predict and GET /health endpoints
- **Metrics logging:** CSV + JSON experiment tracking via MetricsLogger

## Training Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1     | 1.4822    | 45.37%    | 1.1506   | 58.19%  |
| 2     | 1.1437    | 59.10%    | 1.0006   | 64.81%  |
| 3     | 1.0303    | 63.30%    | 0.8897   | 69.10%  |
| 4     | 0.9509    | 66.49%    | 0.8027   | 71.65%  |
| 5     | 0.9009    | 68.04%    | 0.7486   | 73.55%  |
