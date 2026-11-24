FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BURNMAP_MODEL_PATH=/app/models/phase3_nir_swir.pt \
    BURNMAP_RESULTS_DIR=/app/outputs \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    GDAL_HTTP_MULTIPLEX=YES GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES \
    AWS_NO_SIGN_REQUEST=YES

WORKDIR /app
# CPU-only torch keeps the image ~1 GB instead of ~5 GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY web ./web
# Weights are not in git (root .gitignore excludes models/ and *.pt); put the checkpoint
# in ./models before building. See README "Get the weights".
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "burnmap.api:app", "--host", "0.0.0.0", "--port", "8000"]
