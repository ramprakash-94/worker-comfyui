#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Verify the network volume is mounted and ComfyUI is present on it
COMFYUI_PATH="/runpod-volume/runpod-slim/ComfyUI"
if [ ! -f "${COMFYUI_PATH}/main.py" ]; then
    echo "worker-comfyui: ERROR - ComfyUI not found at ${COMFYUI_PATH}/main.py" >&2
    echo "worker-comfyui: Make sure a network volume is attached to this endpoint and ComfyUI is installed on it." >&2
    echo "worker-comfyui: Contents of /runpod-volume (if mounted):" >&2
    ls /runpod-volume 2>&1 >&2 || echo "worker-comfyui: /runpod-volume is not mounted" >&2
    exit 1
fi

# Patch WanVideoWrapper: add 'eager' as a valid torch.compile backend.
# The node ships with only ['inductor', 'cudagraphs']:
#   - inductor needs triton (not installed)
#   - cudagraphs calls checkPoolLiveAllocations which cudaMallocAsync (A100 default) doesn't support
# 'eager' is a no-op compile backend (safe on all GPUs, no graph capture, no triton).
# Setting PYTORCH_CUDA_ALLOC_CONF=backend:native causes a PyTorch assertion on A100 — don't use it.
python3 - <<'PYEOF'
import os, sys
wan_dir = "/runpod-volume/runpod-slim/ComfyUI/custom_nodes/WanVideoWrapper"
if not os.path.isdir(wan_dir):
    print("worker-comfyui: WanVideoWrapper not found, skipping eager-backend patch")
    sys.exit(0)
patched = False
for root, _, files in os.walk(wan_dir):
    for fname in files:
        if not fname.endswith(".py"):
            continue
        path = os.path.join(root, fname)
        with open(path) as fh:
            text = fh.read()
        if "WanVideoTorchCompileSettings" not in text:
            continue
        new_text = text.replace(
            '["inductor", "cudagraphs"]',
            '["inductor", "cudagraphs", "eager"]',
        )
        if new_text != text:
            with open(path, "w") as fh:
                fh.write(new_text)
            print(f"worker-comfyui: Patched {path} — added 'eager' to torch.compile backend list")
            patched = True
if not patched:
    print("worker-comfyui: WanVideoWrapper eager-backend patch: already applied or pattern not found")
PYEOF

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
