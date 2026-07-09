# echoSentinel v2 — offline CPU inference image (PS-12 submission format).
#
# Build (from the repo root, weights/panns_pcen.pt must exist):
#   docker build -f docker/Dockerfile -t echosentinel .
#
# Run fully offline over a folder of .wav files:
#   docker run --rm --network none \
#     -v /path/to/wavs:/data/in:ro -v /path/to/out:/data/out \
#     echosentinel
#
# Output: /data/out/results.json in the competition JSON format.

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image GPU-independent and much smaller.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio && \
    pip install --no-cache-dir \
        "numpy>=1.24,<2.1" "pandas>=2.0,<2.4" "scipy>=1.10" \
        librosa soundfile soxr omegaconf tqdm "pyannote.metrics>=3.2"

# Code, configs, and baked model weights — no network needed at runtime.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY predict.py ./
COPY weights/panns_pcen.pt ./weights/panns_pcen.pt
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["python", "predict.py", \
            "--input_dir", "/data/in", \
            "--output_json", "/data/out/results.json", \
            "--weights", "weights/panns_pcen.pt"]
