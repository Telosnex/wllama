import { test, expect } from 'vitest';
import { Wllama } from './wllama';

const CONFIG_PATHS = {
  'jspi/single-thread/wllama.wasm': '/src/jspi-single-thread/wllama.wasm',
  'asyncify/single-thread/wllama.wasm':
    '/src/asyncify-single-thread/wllama.wasm',
  'asyncify/multi-thread/wllama.wasm': '/src/asyncify-multi-thread/wllama.wasm',
};

const TINY_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf';
const TOOL_MODEL =
  'https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf';

const pocTest =
  import.meta.env.VITE_WLLAMA_SERVER_CONTEXT_POC_TEST === '1'
    ? test.sequential
    : test.sequential.skip;

const webgpuPocTest =
  import.meta.env.VITE_WLLAMA_SERVER_CONTEXT_POC_TEST === '1' &&
  import.meta.env.VITE_WLLAMA_SERVER_CONTEXT_POC_WEBGPU_TEST === '1'
    ? test.sequential
    : test.sequential.skip;

function parseServerChunkObjects(chunks: string[]) {
  return chunks.flatMap((chunk) => {
    const parsed = JSON.parse(chunk);
    return Array.isArray(parsed) ? parsed : [parsed];
  });
}

async function expectTinyCompletion(wllama: Wllama) {
  const result = await wllama._serverContextPoc(
    JSON.stringify({
      messages: [{ role: 'user', content: 'Say hi.' }],
    }),
    { nPredict: 8 }
  );

  expect(result.prompt).toContain('Say hi.');
  expect(result.chunks.length).toBeGreaterThan(0);
  expect(result.chunks.some((chunk) => chunk.includes('chat.completion'))).toBe(
    true
  );
}

pocTest('server_context POC loads without normal wllama loadModel()', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  try {
    await wllama._loadServerContextPocFromUrl(TINY_MODEL, {
      useWebGPU: false,
      useOpfs: false,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 8,
    });

    await expectTinyCompletion(wllama);
    await wllama._unloadServerContextPoc();
  } finally {
    await wllama.exit();
  }
}, 120_000);

pocTest('server_context POC returns OpenAI tool_call chunks', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  try {
    await wllama._loadServerContextPocFromUrl(TOOL_MODEL, {
      useWebGPU: false,
      useOpfs: false,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 128,
    });

    const result = await wllama._serverContextPoc(
      JSON.stringify({
        messages: [{ role: 'user', content: 'Call ping.' }],
        tools: [
          {
            type: 'function',
            function: {
              name: 'ping',
              parameters: { type: 'object', properties: {} },
            },
          },
        ],
        tool_choice: 'required',
      }),
      { nPredict: 128, temp: 0, topP: 1 }
    );

    const chunkObjects = parseServerChunkObjects(result.chunks);
    const choices = chunkObjects.flatMap((chunk) => chunk.choices ?? []);
    const serializedChoices = JSON.stringify(choices);

    expect(result.chatFormat).toBe('Generic');
    expect(result.prompt).toContain('Respond in JSON format');
    expect(serializedChoices).toContain('tool_calls');
    expect(serializedChoices).toContain('ping');
    expect(
      choices.some((choice) => choice.finish_reason === 'tool_calls')
    ).toBe(true);
    await wllama._unloadServerContextPoc();
  } finally {
    await wllama.exit();
  }
}, 120_000);

webgpuPocTest('server_context POC loads cached model with WebGPU', async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) {
    console.warn('Skipping server_context WebGPU POC test: no adapter');
    return;
  }

  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  try {
    const model = await wllama.modelManager.getModelOrDownload(TINY_MODEL);
    expect(model.size).toBeGreaterThan(0);

    await wllama._loadServerContextPocFromUrl(TINY_MODEL, {
      useWebGPU: true,
      useOpfs: false,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 8,
    });

    await expectTinyCompletion(wllama);
    await wllama._unloadServerContextPoc();
  } finally {
    await wllama.exit();
  }
}, 120_000);

pocTest('server_context POC loads cached model directly from OPFS', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  const origFetch = window.fetch;
  try {
    // First make sure the model exists in the OPFS cache. Then force offline so
    // the actual POC load must use the cached Model path.
    const model = await wllama.modelManager.getModelOrDownload(TINY_MODEL);
    expect(model.size).toBeGreaterThan(0);

    window.fetch = () => Promise.reject(new Error('offline'));

    await wllama._loadServerContextPocFromUrl(TINY_MODEL, {
      useWebGPU: false,
      useOpfs: true,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 8,
    });

    await expectTinyCompletion(wllama);
    await wllama._unloadServerContextPoc();
  } finally {
    window.fetch = origFetch;
    await wllama.exit();
  }
}, 120_000);
