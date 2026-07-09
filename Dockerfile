FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch torchaudio && \
    pip install --no-cache-dir \
        "numpy>=1.24,<2.1" "pandas>=2.0,<2.4" "scipy>=1.10" \
        librosa soundfile soxr omegaconf tqdm \
        "pyannote.metrics>=3.2" \
        fastapi "uvicorn>=0.29" python-multipart pillow \
        gdown

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY web ./web
COPY predict.py ./

RUN mkdir -p weights

RUN gdown "https://drive.google.com/uc?id=19IU8-RbiKg4C-yBqHY_wGhA-UVn_Pm7B" \
    -O weights/panns_pcen.pt

RUN pip install --no-cache-dir .

ENV ECHOSENTINEL_HOST=0.0.0.0
ENV ECHOSENTINEL_PORT=8080

EXPOSE 8080

ENTRYPOINT ["python","-m","echosentinel.server"]