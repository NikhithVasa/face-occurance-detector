# InsightFace buffalo_l + onnxruntime-gpu (CUDA 12 / cuDNN 9).
# The base image provides Python, CUDA 12.4 and cuDNN 9, which match the
# onnxruntime-gpu CUDA 12 build. This project does not use PyTorch at runtime;
# the image is chosen because it bundles a known-good CUDA + cuDNN stack that
# RunPod hosts support.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MODULE_LOADING=LAZY

# Help onnxruntime-gpu locate the CUDA / cuDNN runtime shipped in the base image.
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    build-essential \
    cmake \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies. cython + numpy are installed first so insightface can
# build from source on Python versions without a prebuilt wheel.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --upgrade cython numpy \
    && python -m pip install -r /app/requirements.txt \
    && python -m pip install runpod \
    # insightface declares a dependency on the CPU `onnxruntime`, which gets
    # installed alongside `onnxruntime-gpu`. Both write to the same module
    # directory, so the CPU build silently clobbers the GPU one (leaving only
    # CPU/Azure providers). Remove both and reinstall ONLY the GPU build last so
    # CUDAExecutionProvider is the provider that ends up resident.
    && python -m pip uninstall -y onnxruntime onnxruntime-gpu \
    && python -m pip install onnxruntime-gpu==1.22.0

# In this base image the CUDA runtime libraries (cuBLAS, cuDNN, cuFFT, ...) are
# provided as pip `nvidia-*` wheels under site-packages/nvidia/*/lib, which is
# NOT on the default loader path. Without this, onnxruntime-gpu fails to dlopen
# libonnxruntime_providers_cuda.so (missing libcublasLt.so.12 / libcudnn) and
# silently falls back to CPUExecutionProvider at runtime. Register those
# directories with ldconfig so the CUDA provider can be loaded.
RUN python - <<'PY' > /etc/ld.so.conf.d/nvidia-pip-libs.conf
import glob, os
import nvidia
base = os.path.dirname(nvidia.__file__)
for path in sorted(glob.glob(os.path.join(base, "*", "lib"))):
    print(path)
PY
RUN ldconfig

# Fail the build early if the CUDA execution provider is not compiled into
# onnxruntime. Actual GPU availability is verified at runtime on the RunPod host.
RUN python - <<'PY'
import onnxruntime as ort
print('onnxruntime:', ort.__version__)
print('device:', ort.get_device())
print('providers:', ort.get_available_providers())
assert 'CUDAExecutionProvider' in ort.get_available_providers(), \
    'CUDAExecutionProvider missing from the onnxruntime build'
PY

# Pre-download buffalo_l so the runtime needs no network access. CPU is used
# only for this build-time download; runtime enforces CUDAExecutionProvider.
RUN python - <<'PY'
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('InsightFace buffalo_l downloaded')
PY

# Install the application package and the serverless handler.
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src
COPY handler.py /app/handler.py
RUN python -m pip install --no-deps .

# Default entrypoint: RunPod serverless worker.
# For a RunPod Pod, override the command to use the CLI instead, e.g.:
#   python -m face_occurrence_detector.cli \
#     --video /workspace/video.mp4 \
#     --targets /workspace/front.jpg /workspace/side.jpg \
#     --output /workspace/result.json
CMD ["python", "-u", "/app/handler.py"]
