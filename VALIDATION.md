# End-to-End Validation

## Part C: Docker Containerization

### Docker Build - Training Image

```bash
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

Build completes successfully using multi-stage Dockerfile with python:3.11-slim base.

### Docker Run - Training

```bash
$ docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  mlops-train:v1
```

```
{"event": "device_selected", "device": "cpu"}
Files already downloaded and verified
Files already downloaded and verified
{"epoch": 1, "train_loss": 1.4822, "train_accuracy": 0.4537, "val_loss": 1.1506, "val_accuracy": 0.5819, "lr": 0.001}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 2, "train_loss": 1.1437, "train_accuracy": 0.591, "val_loss": 1.0006, "val_accuracy": 0.6481, "lr": 0.001}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 3, "train_loss": 1.0303, "train_accuracy": 0.633, "val_loss": 0.8897, "val_accuracy": 0.691, "lr": 0.001}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 4, "train_loss": 0.9509, "train_accuracy": 0.6649, "val_loss": 0.8027, "val_accuracy": 0.7165, "lr": 0.001}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 5, "train_loss": 0.9009, "train_accuracy": 0.6804, "val_loss": 0.7486, "val_accuracy": 0.7355, "lr": 0.001}
{"event": "checkpoint_saved", "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 0.7486, "best_val_accuracy": 0.7355}
```

Training completes 5 epochs with **73.55% validation accuracy**.

### Docker Build - Serving Image

```bash
$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

Build completes successfully. Image runs as non-root user, exposes port 8080, includes HEALTHCHECK.

### Docker Run - Serving + Health Check

```bash
$ docker run --rm -d -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  --name mlops-serve mlops-serve:v1

$ curl http://localhost:8080/health
{"status":"healthy"}
```

### Test Prediction Endpoint

```bash
$ curl -X POST http://localhost:8080/predict \
  -F "image=@test_image.png"
```

```json
{
    "predicted_class": "cat",
    "probabilities": {
        "airplane": 0.008,
        "automobile": 0.036,
        "bird": 0.0066,
        "cat": 0.4662,
        "deer": 0.0017,
        "dog": 0.1756,
        "frog": 0.0954,
        "horse": 0.0019,
        "ship": 0.1505,
        "truck": 0.058
    }
}
```

Model correctly predicts **"cat"** (46.6% confidence) on a real CIFAR-10 test image.

---

## Part F: Kubernetes Validation (Minikube)

### Apply All Manifests

```bash
$ kubectl apply -f k8s/namespace.yaml
namespace/ml-training created

$ kubectl apply -f k8s/configmap.yaml
configmap/training-config created

$ kubectl apply -f k8s/training-job.yaml
job.batch/pytorch-training-job created

$ kubectl apply -f k8s/serving-deployment.yaml
deployment.apps/model-serving created

$ kubectl apply -f k8s/serving-service.yaml
service/model-serving created

$ kubectl apply -f k8s/hpa.yaml
horizontalpodautoscaler.autoscaling/model-serving-hpa created
```

### Verify Pods Running

```bash
$ kubectl get pods -n ml-training
NAME                              READY   STATUS    RESTARTS   AGE
model-serving-76cb4c7cf9-jx9hf   0/1     Running   0          7m
model-serving-76cb4c7cf9-nvjcv   0/1     Running   0          7m
pytorch-training-job-sw5s6       1/1     Running   0          9m
```

### All Resources

```bash
$ kubectl get all -n ml-training
NAME                                 READY   STATUS    RESTARTS   AGE
pod/model-serving-76cb4c7cf9-jx9hf   0/1     Running   0          7m
pod/model-serving-76cb4c7cf9-nvjcv   0/1     Running   0          7m
pod/pytorch-training-job-sw5s6       1/1     Running   0          9m

NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
service/model-serving   ClusterIP   10.98.254.218   <none>        80/TCP    7m

NAME                            READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/model-serving   0/2     2            0           7m

NAME                                       DESIRED   CURRENT   READY   AGE
replicaset.apps/model-serving-76cb4c7cf9   2         2         0       7m

NAME                                                    REFERENCE                  TARGETS              MINPODS   MAXPODS   REPLICAS   AGE
horizontalpodautoscaler.autoscaling/model-serving-hpa   Deployment/model-serving   cpu: <unknown>/70%   2         10        2          7m

NAME                             STATUS    COMPLETIONS   DURATION   AGE
job.batch/pytorch-training-job   Running   0/1           9m         9m
```

### Describe Deployment

```bash
$ kubectl describe deployment model-serving -n ml-training
Name:                   model-serving
Namespace:              ml-training
Replicas:               2 desired | 2 updated | 2 total
StrategyType:           RollingUpdate
RollingUpdateStrategy:  0 max unavailable, 1 max surge
Pod Template:
  Containers:
   serving:
    Image:      mlops-serve:v1
    Port:       8080/TCP
    Limits:     cpu: 500m, memory: 1Gi
    Requests:   cpu: 250m, memory: 512Mi
    Liveness:   http-get http://:8080/health delay=0s period=10s failureThreshold=3
    Readiness:  http-get http://:8080/health delay=15s period=5s
    Mounts:     /app/checkpoints from checkpoint-volume (ro)
Events:
  Normal  ScalingReplicaSet  Scaled up replica set model-serving-76cb4c7cf9 from 0 to 1
  Normal  ScalingReplicaSet  Scaled up replica set model-serving-76cb4c7cf9 from 1 to 2
```

### Port-Forward + Predict (via Docker, since K8s serving pods lack checkpoint in emptyDir)

The serving pods on Minikube use `emptyDir` volumes (no PVC provisioner) and thus lack the trained checkpoint. The Docker-based serving validation above demonstrates the full predict workflow end-to-end.

```bash
$ kubectl port-forward svc/model-serving 8080:80 -n ml-training
$ curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## Notes

- **Model:** SimpleCNN (3-layer CNN, ~62K params) — lightweight for CPU training
- **Dataset:** CIFAR-10 (10 classes, 50K train / 10K test images)
- **Validation accuracy:** 73.55% after 5 epochs on CPU
- **K8s manifests** in the repo use PersistentVolumeClaims and GPU resources for production environments. Minikube deployment uses emptyDir and reduced resources for local testing.
