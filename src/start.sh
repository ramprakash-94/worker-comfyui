#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Force the native synchronous CUDA allocator.
# A100/H100 default to cudaMallocAsync which conflicts with torch.compile's
# cudagraphs backend (checkPoolLiveAllocations error). Native allocator avoids this.
export PYTORCH_CUDA_ALLOC_CONF="backend:native"

# Verify the network volume is mounted and ComfyUI is present on it
COMFYUI_PATH="/runpod-volume/runpod-slim/ComfyUI"
if [ ! -f "${COMFYUI_PATH}/main.py" ]; then
    echo "worker-comfyui: ERROR - ComfyUI not found at ${COMFYUI_PATH}/main.py" >&2
    echo "worker-comfyui: Make sure a network volume is attached to this endpoint and ComfyUI is installed on it." >&2
    echo "worker-comfyui: Contents of /runpod-volume (if mounted):" >&2
    ls /runpod-volume 2>&1 >&2 || echo "worker-comfyui: /runpod-volume is not mounted" >&2
    exit 1
fi

# Ensure ComfyUI-Manager runs in offline mode (targets the network volume's ComfyUI)
COMFYUI_MANAGER_CONFIG="${COMFYUI_PATH}/user/default/ComfyUI-Manager/config.ini" \
  comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

echo "Symlinking files from Network Volume"
rm -rf /workspace && \
  mkdir -p /workspace && \
  ln -s /runpod-volume/runpod-slim /workspace/ComfyUI

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u "${COMFYUI_PATH}/main.py" --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout --highvram 2>&1 &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u "${COMFYUI_PATH}/main.py" --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout --highvram 2>&1 &

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
