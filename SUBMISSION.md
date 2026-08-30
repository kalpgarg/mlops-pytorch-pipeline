# Assignment Submission: MLOps PyTorch Pipeline

**Course:** MLOps & Infrastructure for Machine Learning
**Assignment:** Deploying PyTorch ML Workloads with Docker & Kubernetes

---

## GitHub Repository

**Repository:** [https://github.com/kalpgarg/mlops-pytorch-pipeline](https://github.com/kalpgarg/mlops-pytorch-pipeline)

---

## Pull Requests

| # | PR | Branch | Description |
|---|-----|--------|-------------|
| 1 | [PR #1: Model Training Utilities](https://github.com/kalpgarg/mlops-pytorch-pipeline/pull/1) | `feature/model-training` | Added MetricsLogger for CSV/JSON experiment tracking, ReduceLROnPlateau learning rate scheduler, `get_device()` utility for CPU/GPU/MPS detection, and unit tests for new utilities. |
| 2 | [PR #2: Docker Support](https://github.com/kalpgarg/mlops-pytorch-pipeline/pull/2) | `feature/docker-support` | Added `.dockerignore`, `docker-compose.yml` for local dev, improved Dockerfiles with labels and `ca-certificates` for SSL, pinned `numpy<2` for torch compatibility. |
| 3 | [PR #3: Validation & README](https://github.com/kalpgarg/mlops-pytorch-pipeline/pull/3) | `feature/validation` | Updated README with SimpleCNN model description, training results table (73.55% val accuracy), and metrics logging details. |
| 4 | [PR #4: K8s Deployment](https://github.com/kalpgarg/mlops-pytorch-pipeline/pull/4) | `feature/k8s-deployment` | Enhanced K8s configmap with `log_dir`, added labels and named ports to serving service for better resource identification. |
| 5 | [PR #5: Final Validation](https://github.com/kalpgarg/mlops-pytorch-pipeline/pull/5) | `feature/final-validation` | End-to-end validation document with Docker build/run outputs, serving health check and prediction results, and Kubernetes deployment outputs. |

---

## Validation Summary

**Full details:** [VALIDATION.md](https://github.com/kalpgarg/mlops-pytorch-pipeline/blob/main/VALIDATION.md)

- **Docker Training:** Built `mlops-train:v1` image and ran 5-epoch training on CIFAR-10 using SimpleCNN (~62K params). Achieved **73.55% validation accuracy**. Metrics logged in structured JSON format to stdout.
- **Docker Serving:** Built `mlops-serve:v1` image. Health endpoint (`GET /health`) returns `{"status":"healthy"}`. Prediction endpoint (`POST /predict`) correctly classifies a CIFAR-10 cat image with 46.6% confidence.
- **Kubernetes Deployment:** Deployed all manifests on Minikube — namespace, configmap, training job, serving deployment (2 replicas), ClusterIP service, and HPA. Training job ran successfully. Serving deployment configured with liveness/readiness probes and rolling update strategy.

---

## Reflection Summary

**Full details:** [REFLECTION.md](https://github.com/kalpgarg/mlops-pytorch-pipeline/blob/main/REFLECTION.md)

- **Most Challenging:** SSL certificate verification failures inside Docker containers blocking CIFAR-10 download. Required patching Python's SSL context directly in code since `torchvision` uses `urllib` internally.
- **Second Challenge:** ResNet-18 (~11M params) was too slow for CPU training. Replaced with SimpleCNN (~62K params) — trains in ~10 min vs hours.
- **Key Learnings:** K8s primitives mapping to ML workloads (Jobs for training, Deployments for serving), Docker multi-stage build layer caching, and disciplined Git workflow with feature branches and PRs.
- **Design Decisions:** FastAPI for automatic OpenAPI docs, MetricsLogger for dual CSV/JSON tracking, HPA at 70% CPU utilization for inference scaling.
