# vLLM ships an official CUDA image with the engine + OpenAI server preinstalled.
# Pin a version for reproducibility; bump deliberately.
FROM vllm/vllm-openai:v0.6.6

# Repo helpers (config loader, serve.sh). Weights are NOT baked in — they download at
# startup into the mounted HF cache (or pre-warm with scripts/download_model.py).
WORKDIR /app
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY scripts/serve.sh /app/scripts/serve.sh
RUN chmod +x /app/scripts/serve.sh

ENV PYTHONPATH=/app/src \
    MODEL_CONFIG=qwen3-0.6b \
    VLLM_HOST=0.0.0.0 \
    VLLM_PORT=8000

EXPOSE 8000

# The base image's entrypoint is `vllm serve ...`; override it to drive from a config file.
ENTRYPOINT ["/app/scripts/serve.sh"]
