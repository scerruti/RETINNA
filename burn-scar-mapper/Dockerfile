# One image, two Lambda functions (see deploy/cloudformation.yaml):
#   API:    CMD burnmap.lambda_api.handler   (default below; behind a function URL)
#   worker: CMD burnmap.worker.handler       (set by the stack's ImageConfig)
# Built for arm64 (Graviton Lambda: cheaper, and native on Apple Silicon).
#
# Local run without AWS (uvicorn instead of the Lambda runtime):
#   docker run -p 8000:8000 --entrypoint uvicorn burn-scar-mapper burnmap.api:app --host 0.0.0.0
FROM public.ecr.aws/lambda/python:3.12

# Checkpoint file in ./models to bake into the image (weights are not in git).
ARG MODEL_FILE=phase3_nir_swir.pt

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BURNMAP_MODEL_PATH=/opt/burnmap/models/${MODEL_FILE} \
    BURNMAP_WEB_DIR=/opt/burnmap/web \
    BURNMAP_RESULTS_DIR=/tmp/outputs \
    HOME=/tmp \
    GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR \
    GDAL_HTTP_MULTIPLEX=YES GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES \
    GDAL_CACHEMAX=512 \
    AWS_NO_SIGN_REQUEST=YES

# CPU-only torch keeps the image ~1 GB instead of ~5 GB
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml /tmp/build/
COPY src /tmp/build/src
RUN pip install --no-cache-dir "/tmp/build[aws]" && rm -rf /tmp/build
COPY web /opt/burnmap/web
# Weights are not in git (root .gitignore excludes models/ and *.pt); put the checkpoint
# in ./models before building. See README "Get the weights".
COPY models/${MODEL_FILE} /opt/burnmap/models/${MODEL_FILE}

CMD ["burnmap.lambda_api.handler"]
