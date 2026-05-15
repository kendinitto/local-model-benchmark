#!/bin/bash
#
# Build llama.cpp with optimized settings for benchmarking
#
# Usage: ./build_llama.sh [backend]
#   backend: cuda, metal, vulkan, cpu (default: auto-detect)
#

set -e

BACKEND="${1:-auto}"

echo "========================================="
echo "llama.cpp Build Script"
echo "========================================="

# Clone or update llama.cpp
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git
else
    echo "Updating llama.cpp..."
    cd llama.cpp && git pull && cd ..
fi

cd llama.cpp
mkdir -p build
cd build

# Detect backend if auto
if [ "$BACKEND" = "auto" ]; then
    if command -v nvidia-smil &> /dev/null || command -v nvidia-smi &> /dev/null; then
        BACKEND="cuda"
        echo "Detected NVIDIA GPU — building with CUDA"
    elif [ "$(uname)" = "Darwin" ]; then
        BACKEND="metal"
        echo "Detected macOS — building with Metal"
    else
        BACKEND="cpu"
        echo "No GPU detected — building for CPU only"
    fi
fi

# CMake configuration
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple"

case "$BACKEND" in
    cuda)
        CMAKE_ARGS="-DGGML_CUDA=ON -DLLAMA_CUDA_F16=ON"
        ;;
    metal)
        CMAKE_ARGS="-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON"
        ;;
    vulkan)
        CMAKE_ARGS="-DGGML_VULKAN=ON"
        ;;
    cpu)
        CMAKE_ARGS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_F16C=ON -DGGML_FMA=ON -DGGML_LTO=ON"
        ;;
esac

echo ""
echo "Building with backend: $BACKEND"
echo "CMake args: $CMAKE_ARGS"
echo ""

cmake .. $CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "========================================="
echo "Build complete!"
echo "Binary: $(pwd)/llama-cli"
echo "========================================="

# Verify build
if [ -f "llama-cli" ]; then
    echo ""
    echo "Running version check..."
    ./llama-cli --version || true
else
    echo "ERROR: llama-cli not found after build!"
    exit 1
fi
