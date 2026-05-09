#!/bin/bash

set -e

CURRENT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# change to the llama.cpp directory. Allow the server-context POC to use the
# fllama-pinned llama.cpp checkout without requiring it to be copied into
# wllama/llama.cpp.
LLAMA_CPP_DIR="${WLLAMA_LLAMA_CPP_DIR_HOST:-$CURRENT_PATH/../llama.cpp}"
cd "$LLAMA_CPP_DIR"
BUILD_NUMBER="$(git rev-list --count HEAD 2>/dev/null || echo 0)"
SHORT_HASH="$(git rev-parse --short=7 HEAD 2>/dev/null || basename "$LLAMA_CPP_DIR")"

# change to the root of the project
cd $CURRENT_PATH
cd ..

echo "// This file is auto-generated" > ./src/workers-code/generated.ts
echo "// To re-generate it, run: npm run build:worker" >> ./src/workers-code/generated.ts
echo "" >> ./src/workers-code/generated.ts
echo "export const LIBLLAMA_VERSION = 'b${BUILD_NUMBER}-${SHORT_HASH}';" >> ./src/workers-code/generated.ts
echo "" >> ./src/workers-code/generated.ts

process_file() {
  local file="$1"
  local content
  content=$(node -e "console.log(JSON.stringify(require('fs').readFileSync('$file', 'utf8').toString()))")
  echo "export const $2 = $content;" >> ./src/workers-code/generated.ts
  echo "" >> ./src/workers-code/generated.ts
}

process_file ./src/workers-code/llama-cpp.js  LLAMA_CPP_WORKER_CODE
process_file ./src/workers-code/opfs-utils.js OPFS_UTILS_WORKER_CODE

# emscripten
process_file ./src/jspi-single-thread/wllama.js        WLLAMA_JSPI_SINGLE_THREAD_CODE
process_file ./src/asyncify-single-thread/wllama.js    WLLAMA_ASYNCIFY_SINGLE_THREAD_CODE
process_file ./src/asyncify-multi-thread/wllama.js     WLLAMA_ASYNCIFY_MULTI_THREAD_CODE

# build CDN paths
node ./scripts/generate_wasm_from_cdn.js
