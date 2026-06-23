# Deploying llm-inference

## Local stack (Docker Compose)

From the repo root:

```bash
cd docker && docker compose up
```

- vLLM — http://localhost:8000 (OpenAI API + `/metrics` + `/health`)
- Prometheus — http://localhost:9090
- Grafana — http://localhost:3000 (anonymous access on; admin/admin to edit)

Set `MODEL_CONFIG` (and `HF_TOKEN` for gated models) in a `.env` file next to the compose file.

## Kubernetes

```bash
# 1. Build & push the image to a registry your cluster can pull from
docker build -t <registry>/llm-inference:0.1.0 -f docker/Dockerfile .
docker push <registry>/llm-inference:0.1.0
#    then update image: in deploy/k8s/deployment.yaml

# 2. (Gated models) create the HF token secret
kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN

# 3. Apply
kubectl apply -f deploy/k8s/
```

Manifests:

| File | Purpose |
| --- | --- |
| `deployment.yaml` | GPU `Deployment` + HF-cache PVC + large `/dev/shm`; probes wired to `/health` |
| `service.yaml` | `ClusterIP` fronting the pods on port 80 → 8000 |
| `hpa.yaml` | Autoscale on `vllm_num_requests_waiting` (needs prometheus-adapter) |

### Scaling notes

- **Bigger models:** set `tensor_parallel_size` in the model config and raise `nvidia.com/gpu`
  to match (one GPU per shard). NCCL needs the large `/dev/shm` already provisioned here.
- **Autoscaling signal:** queue depth (`num_requests_waiting`) reflects saturation far better
  than CPU/GPU utilization for token-generation workloads.
- **Cold starts:** weight download + CUDA-graph capture can take minutes; the `startupProbe`
  allows up to ~10 min. Pre-warm with `scripts/download_model.py` baked into the image or an
  init container to shorten this.
