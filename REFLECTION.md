# Reflection: MLOps PyTorch Pipeline

## Overview

This project involved building a production-style ML pipeline that takes a PyTorch image classifier from local development through Docker containerization to Kubernetes deployment. The pipeline includes a CIFAR-10 classifier, containerized training and serving workflows, and Kubernetes manifests for orchestrated deployment.

## Most Challenging Part

The most challenging aspect was getting Docker containers to reliably download the CIFAR-10 dataset. SSL certificate verification failures inside containers blocked the `torchvision.datasets.CIFAR10` download call. I tried multiple approaches: installing `ca-certificates` in the Dockerfile, setting `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` environment variables, and using `PYTHONHTTPSVERIFY=0`. Ultimately, the fix required patching Python's SSL context directly in `dataset.py` with `ssl._create_default_https_context = ssl._create_unverified_context`, since `torchvision` uses `urllib` internally and does not respect environment-level SSL overrides. This taught me that containerized ML workloads often face networking issues that do not appear during local development, and that understanding the full dependency chain (Python → urllib → SSL → OS certificates) is critical for debugging.

A second major challenge was model training speed on CPU. ResNet-18 (~11M parameters) took over 10 minutes per epoch on CPU inside Docker, making iteration painfully slow. I replaced it with a lightweight 3-layer CNN (SimpleCNN, ~62K parameters) that trains a full 5-epoch run in approximately 10 minutes while still achieving 73.5% validation accuracy on CIFAR-10. This trade-off between model complexity and development velocity is a practical consideration in MLOps workflows where rapid iteration matters more than state-of-the-art accuracy.

## Key Learnings

Working through the Kubernetes deployment deepened my understanding of how ML workloads map to K8s primitives: Jobs for batch training, Deployments with health probes for serving, ConfigMaps for externalizing hyperparameters, and PersistentVolumeClaims for data persistence across pod restarts. Writing multi-stage Dockerfiles reinforced the importance of layer caching and dependency isolation — separating `pip install` from source code copies dramatically speeds up rebuilds during development.

The Git workflow with feature branches and pull requests enforced a disciplined development process. Each feature branch (model training utilities, Docker support, K8s deployment, validation) represented a logical unit of work, making code review and incremental integration straightforward.

## Design Decisions

I chose FastAPI over Flask for the serving API due to its automatic OpenAPI documentation, async support, and built-in request validation. The MetricsLogger utility writes both CSV (for quick analysis) and JSON (for programmatic consumption), supporting the structured logging requirement while adding experiment tracking capabilities. The HorizontalPodAutoscaler configuration targets 70% CPU utilization, balancing responsiveness with resource efficiency for inference workloads.
