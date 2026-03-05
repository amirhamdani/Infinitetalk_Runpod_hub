#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Detect GPU compute capability and decide whether sage attention is safe
echo "Detecting GPU compute capability..."
GPU_CC=$(python -c "import torch; cc=torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "0")
echo "GPU Compute Capability code: $GPU_CC"

SAGE_FLAG=""
if [ "$GPU_CC" -ge "100" ]; then
    echo "Blackwell+ GPU detected (CC >= 10.0) — enabling SageAttention"
    SAGE_FLAG="--use-sage-attention"
else
    echo "Non-Blackwell GPU detected (CC $GPU_CC) — SageAttention disabled to avoid FP8 kernel crash"
fi

# Allow explicit override via environment variable
if [ "${FORCE_SAGE_ATTENTION}" = "1" ]; then
    SAGE_FLAG="--use-sage-attention"
    echo "FORCE_SAGE_ATTENTION=1 — SageAttention force-enabled"
elif [ "${FORCE_SAGE_ATTENTION}" = "0" ]; then
    SAGE_FLAG=""
    echo "FORCE_SAGE_ATTENTION=0 — SageAttention force-disabled"
fi

# Start ComfyUI in the background
echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen $SAGE_FLAG &

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120  # 최대 2분 대기
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

# Start the handler in the foreground
# 이 스크립트가 컨테이너의 메인 프로세스가 됩니다.
echo "Starting the handler..."
exec python handler.py