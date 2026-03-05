#!/bin/bash

set -e

# ── GPU driver warm-up ──────────────────────────────────────────────
# nvidia-smi talks directly to the kernel driver and must succeed
# before PyTorch or ComfyUI can use CUDA. Retry up to 30 s to handle
# the race condition where the device files are not yet ready.
echo "Waiting for NVIDIA driver..."
max_gpu_wait=30
gpu_wait=0
while [ $gpu_wait -lt $max_gpu_wait ]; do
    if nvidia-smi > /dev/null 2>&1; then
        echo "NVIDIA driver ready"
        nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
        break
    fi
    echo "  GPU not available yet ($gpu_wait/$max_gpu_wait)..."
    sleep 2
    gpu_wait=$((gpu_wait + 2))
done

if [ $gpu_wait -ge $max_gpu_wait ]; then
    echo "WARNING: nvidia-smi never succeeded — CUDA may not work"
fi

# ── Detect GPU compute capability via nvidia-smi (no PyTorch needed) ─
echo "Detecting GPU compute capability..."
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo "0")
echo "GPU Compute Capability code: $GPU_CC"

SAGE_FLAG=""
if [ "$GPU_CC" -ge "100" ]; then
    echo "Blackwell+ GPU detected (CC >= 10.0) — enabling SageAttention"
    SAGE_FLAG="--use-sage-attention"
else
    echo "Non-Blackwell GPU detected (CC $GPU_CC) — SageAttention disabled to avoid FP8 kernel crash"
fi

if [ "${FORCE_SAGE_ATTENTION}" = "1" ]; then
    SAGE_FLAG="--use-sage-attention"
    echo "FORCE_SAGE_ATTENTION=1 — SageAttention force-enabled"
elif [ "${FORCE_SAGE_ATTENTION}" = "0" ]; then
    SAGE_FLAG=""
    echo "FORCE_SAGE_ATTENTION=0 — SageAttention force-disabled"
fi

# ── Warm up CUDA for PyTorch ────────────────────────────────────────
# Running a trivial CUDA op before ComfyUI prevents the "CUDA unknown
# error" race condition that kills torch._C._cuda_init().
echo "Warming up PyTorch CUDA..."
for attempt in 1 2 3 4 5; do
    if python -c "import torch; torch.cuda.init(); print(f'PyTorch CUDA OK — {torch.cuda.get_device_name(0)}')" 2>/dev/null; then
        break
    fi
    echo "  CUDA init attempt $attempt failed, retrying in 3s..."
    sleep 3
done

# ── Start ComfyUI ───────────────────────────────────────────────────
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen $SAGE_FLAG &

echo "Waiting for ComfyUI to be ready..."
max_wait=180
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

echo "Starting the handler..."
exec python handler.py
