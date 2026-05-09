#!/bin/bash

#set -e

export EMSDK_IMAGE_TAG="4.0.20"

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $CURRENT_PATH

export D_UID=$UID
export D_GID=$GID
export WLLAMA_ENABLE_SERVER_CONTEXT_POC=${WLLAMA_ENABLE_SERVER_CONTEXT_POC:-OFF}
export WLLAMA_GGML_WEBGPU=${WLLAMA_GGML_WEBGPU:-ON}
export WLLAMA_LLAMA_CPP_DIR=${WLLAMA_LLAMA_CPP_DIR:-/source/llama.cpp}
export WLLAMA_LLAMA_CPP_DIR_HOST=${WLLAMA_LLAMA_CPP_DIR_HOST:-../llama.cpp}

if [[ $(uname -m) == "arm64" ]]; then
  echo "Running on ARM64 processor"
  export DOCKER_DEFAULT_PLATFORM="linux/arm64"
  export EMSDK_IMAGE_TAG="${EMSDK_IMAGE_TAG}-arm64"
fi

docker compose up llamacpp-wasm-builder --exit-code-from llamacpp-wasm-builder
