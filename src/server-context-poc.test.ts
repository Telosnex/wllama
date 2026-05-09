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

async function expectTinyCompletion(wllama: Wllama) {
  const result = await wllama.createServerChatCompletion(
    {
      messages: [{ role: 'user', content: 'Say hi.' }],
    },
    { nPredict: 8 }
  );

  expect(result.debug.prompt).toContain('Say hi.');
  expect(result.rawChunks.length).toBeGreaterThan(0);
  expect(
    result.rawChunks.some((chunk) => chunk.includes('chat.completion'))
  ).toBe(true);
}

async function expectTinyStreamingCompletion(wllama: Wllama) {
  const stream = await wllama.createServerChatCompletionStream(
    {
      messages: [{ role: 'user', content: 'Say hi streamed.' }],
    },
    { nPredict: 8 }
  );

  const chunks = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }

  expect(chunks.length).toBeGreaterThan(0);
  expect(chunks.some((chunk) => chunk.rawChunk.includes('chat.completion'))).toBe(
    true
  );
}

pocTest('server_context POC loads without normal wllama loadModel()', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  try {
    await wllama.loadServerModelFromUrl(TINY_MODEL, {
      useWebGPU: false,
      useOpfs: false,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 8,
    });

    await expectTinyCompletion(wllama);
    await expectTinyStreamingCompletion(wllama);
    await wllama.unloadServerModel();
  } finally {
    await wllama.exit();
  }
}, 120_000);

pocTest('server_context POC returns OpenAI tool_call chunks', async () => {
  const wllama = new Wllama(CONFIG_PATHS, {
    allowOffline: true,
  });

  try {
    await wllama.loadServerModelFromUrl(TOOL_MODEL, {
      useWebGPU: false,
      useOpfs: false,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 128,
    });

    const result = await wllama.createServerChatCompletion(
      {
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
      },
      { nPredict: 128, sampling: { temp: 0, top_p: 1 } }
    );

    const choices = result.chunks.flatMap(
      (chunk) => (chunk as { choices?: unknown[] }).choices ?? []
    );
    const serializedChoices = JSON.stringify(choices);

    expect(result.debug.chatFormat).toBe('Generic');
    expect(result.debug.prompt).toContain('Respond in JSON format');
    expect(serializedChoices).toContain('tool_calls');
    expect(serializedChoices).toContain('ping');
    expect(
      choices.some(
        (choice) =>
          (choice as { finish_reason?: string }).finish_reason === 'tool_calls'
      )
    ).toBe(true);
    await wllama.unloadServerModel();
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

    await wllama.loadServerModelFromUrl(TINY_MODEL, {
      useWebGPU: true,
      useOpfs: false,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 8,
    });

    await expectTinyCompletion(wllama);
    await wllama.unloadServerModel();
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

    await wllama.loadServerModelFromUrl(TINY_MODEL, {
      useWebGPU: false,
      useOpfs: true,
      n_ctx: 512,
      n_threads: 1,
      nPredict: 8,
    });

    await expectTinyCompletion(wllama);
    await wllama.unloadServerModel();
  } finally {
    window.fetch = origFetch;
    await wllama.exit();
  }
}, 120_000);
