#!/bin/bash
# build_plugin.sh
# Build the JamMa SelectiveScan TRT plugin as a shared library.
#
# Prerequisites:
#   - TensorRT 10.x installed (e.g., at /usr/local/TensorRT-10.16.0.72)
#   - PyTorch with CUDA (for libtorch)
#   - mamba_ssm installed (provides selective_scan_cuda kernel)
#   - CMake >= 3.18, CUDA toolkit, GCC >= 9
#
# Usage:
#   conda activate jamma
#   bash build_plugin.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# --- Detect TensorRT ---
TRT_ROOT=""
for candidate in /usr/local/TensorRT-*; do
    if [ -f "${candidate}/include/NvInfer.h" ]; then
        TRT_ROOT="${candidate}"
        break
    fi
done
if [ -z "${TRT_ROOT}" ]; then
    echo "ERROR: TensorRT not found. Set TRT_ROOT environment variable."
    exit 1
fi
echo "TensorRT: ${TRT_ROOT}"

# --- Detect PyTorch ---
TORCH_ROOT=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || echo "")
if [ -z "${TORCH_ROOT}" ]; then
    echo "ERROR: PyTorch not found. Activate conda environment first."
    exit 1
fi
echo "PyTorch:  ${TORCH_ROOT}"

# --- Configure & Build ---
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_ROOT="${TRT_ROOT}" \
    -DTORCH_ROOT="${TORCH_ROOT}" \
    -DCMAKE_CUDA_ARCHITECTURES="native"

make -j$(nproc)

echo ""
echo "============================================"
echo "  Plugin built: ${BUILD_DIR}/libjam_plugin.so"
echo "============================================"
echo ""
echo "Next: python build_trt_engine.py"
