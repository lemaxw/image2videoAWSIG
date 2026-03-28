FROM python:3.10-slim

WORKDIR /opt

ARG COMFYUI_REF=4c4be1bba5ae714c6f455a49757bd7fc2e32c577
ARG TORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    curl \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/ComfyUI \
    && git -C /opt/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git -C /opt/ComfyUI fetch --depth 1 origin ${COMFYUI_REF} \
    && git -C /opt/ComfyUI checkout --detach FETCH_HEAD

WORKDIR /opt/ComfyUI
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall \
         torch==${TORCH_VERSION} \
         torchvision==${TORCHVISION_VERSION} \
         torchaudio==${TORCHAUDIO_VERSION}

# Required by workflow templates that use VHS_VideoCombine.
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite \
    && if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt ]; then \
         pip install --no-cache-dir -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt; \
       fi

# AnimateDiff Evolved
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git /opt/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved \
    && if [ -f /opt/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/requirements.txt ]; then \
         pip install --no-cache-dir -r /opt/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/requirements.txt; \
       fi
              
EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--output-directory", "/data/outputs/comfy", "--cpu"]
