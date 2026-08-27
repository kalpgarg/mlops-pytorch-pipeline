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

- **Architecture:** ResNet-18 (adapted for CIFAR-10 32x32 images)
- **Dataset:** CIFAR-10 (10 classes)
- **Training:** Adam optimizer, CrossEntropy loss, early stopping
- **Serving:** FastAPI with POST /predict and GET /health endpoints
