FROM python:3.11-slim

# Audio system libraries: libsndfile for soundfile (wav/flac/ogg), ffmpeg for
# librosa's mp3 fallback. Installed as root before we drop to the app user.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as user ID 1000. Everything below runs
# as this user so the app can write to its working tree (out/webapp/…) at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

RUN pip install --no-cache-dir --upgrade pip

# CPU-only torch keeps the image small; then the rest of the runtime deps.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio && \
    pip install --no-cache-dir \
        "numpy>=1.24,<2.1" "pandas>=2.0,<2.4" "scipy>=1.10" \
        librosa soundfile soxr omegaconf tqdm \
        "pyannote.metrics>=3.2" \
        fastapi "uvicorn>=0.29" python-multipart pillow \
        gdown

COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
COPY --chown=user configs ./configs
COPY --chown=user web ./web
COPY --chown=user predict.py ./

# Writable dirs owned by the app user.
RUN mkdir -p weights out

# Model weights (293 MB) pulled from Google Drive at build time.
RUN gdown "https://drive.google.com/uc?id=19IU8-RbiKg4C-yBqHY_wGhA-UVn_Pm7B" \
        -O weights/panns_pcen.pt

RUN pip install --no-cache-dir .

ENV ECHOSENTINEL_ROOT=/home/user/app
ENV ECHOSENTINEL_HOST=0.0.0.0
ENV ECHOSENTINEL_PORT=8080

EXPOSE 8080

ENTRYPOINT ["python","-m","echosentinel.server"]
