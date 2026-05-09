#include "glue.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten/threading.h>
#endif

#define PARSE_REQ(msg_typename) \
  msg_typename req;             \
  glue_inbuf inbuf(req_raw);    \
  req.handler.deserialize(inbuf);

struct app_t;

#if WLLAMA_ENABLE_SERVER_CONTEXT_POC
#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"
#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

namespace {

struct server_context_poc_state
{
  std::unique_ptr<server_context> srv;
  std::thread loop_thread;
  std::string model_path;
  int32_t n_predict = 64;
  bool use_webgpu = false;

  bool loaded() const
  {
    return srv != nullptr;
  }

  void unload()
  {
    if (srv) {
      srv->terminate();
    }
    if (loop_thread.joinable()) {
      loop_thread.join();
    }
    srv.reset();
    model_path.clear();
    n_predict = 64;
    use_webgpu = false;
  }

  ~server_context_poc_state()
  {
    unload();
  }
};

static server_context_poc_state g_server_context_poc;

static common_params make_common_params(
    const std::string &model_path,
    bool has_use_webgpu,
    bool use_webgpu,
    bool has_n_ctx,
    int32_t n_ctx,
    bool has_n_batch,
    int32_t n_batch,
    bool has_n_ubatch,
    int32_t n_ubatch,
    bool has_n_threads,
    int32_t n_threads,
    bool has_n_gpu_layers,
    int32_t n_gpu_layers,
    bool has_n_predict,
    int32_t n_predict)
{
  common_params params;
  params.model.path = model_path;
  params.n_ctx = has_n_ctx ? n_ctx : 1024;
  params.n_batch = has_n_batch ? n_batch : std::min<int32_t>(params.n_ctx, 2048);
  params.n_ubatch = has_n_ubatch ? n_ubatch : std::min<int32_t>(params.n_batch, 512);
  // Flash attention AUTO currently gets enabled for the tiny CPU/browser smoke
  // model and then stalls after prompt processing under Emscripten pthreads.
  // Keep the server-context POC conservative until the browser runtime is
  // validated end-to-end.
  params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  params.n_parallel = 1;
  params.n_predict = has_n_predict ? n_predict : 64;
  params.cpuparams.n_threads = has_n_threads ? n_threads : 1;
  params.use_jinja = true;
  params.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
  params.cache_ram_mib = 0;
  // The POC may mount cached GGUFs through a custom OPFS-backed MEMFS node.
  // Force stdio/fread instead of mmap so reads flow through the JS-backed file
  // ops rather than assuming a normal contiguous memory mapping.
  params.use_mmap = false;
  params.use_mlock = false;
  params.n_gpu_layers = has_n_gpu_layers ? n_gpu_layers : (has_use_webgpu && use_webgpu ? 999 : 0);

  return params;
}

static void select_backend_device(common_params &params, bool has_use_webgpu, bool use_webgpu)
{
  if (!has_use_webgpu) {
    return;
  }

  ggml_backend_dev_t dev = use_webgpu
      ? ggml_backend_dev_by_name("WebGPU")
      : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (dev == nullptr) {
    throw std::runtime_error(use_webgpu ? "WebGPU backend not available" : "CPU backend not available");
  }
  params.devices.push_back(dev);
  params.devices.push_back(nullptr);
}

static void load_server_context(common_params &&params, bool has_use_webgpu, bool use_webgpu)
{
  ggml_backend_load_all();
  select_backend_device(params, has_use_webgpu, use_webgpu);

  g_server_context_poc.unload();
  auto next = std::make_unique<server_context>();
  if (!next->load_model(params)) {
    throw std::runtime_error("server_context::load_model failed");
  }

  g_server_context_poc.model_path = params.model.path;
  g_server_context_poc.n_predict = params.n_predict;
  g_server_context_poc.use_webgpu = use_webgpu;
  g_server_context_poc.srv = std::move(next);
#if defined(__EMSCRIPTEN__)
  // Emscripten only allows std::thread in pthread-enabled builds. The
  // asyncify-single-thread artifact is still useful for clear diagnostics, but
  // it cannot host server_context's blocking start_loop thread.
  if (!emscripten_has_threading_support()) {
    throw std::runtime_error(
        "server_context_poc requires the asyncify/multi-thread wasm artifact");
  }
#endif
  if (!use_webgpu) {
    g_server_context_poc.loop_thread = std::thread([] {
      g_server_context_poc.srv->start_loop();
    });
  }
}

static void apply_request_json(
    server_context &srv,
    const std::string &request_json,
    const std::string &fallback_prompt,
    const std::string &explicit_jinja_template,
    std::string &prompt,
    bool &is_oai,
    common_chat_parser_params &parser_params,
    std::string &chat_format,
    std::string &reasoning_format,
    common_chat_params &chat_params)
{
  prompt = fallback_prompt;
  is_oai = false;
  chat_format.clear();
  reasoning_format.clear();
  chat_params = common_chat_params();

  if (request_json.empty()) {
    return;
  }

  auto body = nlohmann::ordered_json::parse(request_json);
  if (!body.contains("messages") || !body["messages"].is_array()) {
    return;
  }

  is_oai = true;

  std::string jinja_tmpl = explicit_jinja_template;
  if (jinja_tmpl.empty() && body.contains("jinja_template") && body["jinja_template"].is_string()) {
    jinja_tmpl = body["jinja_template"].get<std::string>();
    body.erase("jinja_template");
  }

  auto *lctx = srv.get_llama_context();
  auto *model = llama_get_model(lctx);
  auto tmpls = common_chat_templates_init(model, jinja_tmpl);

  try {
    std::map<std::string, std::string> empty;
    common_chat_format_example(tmpls.get(), true, empty);
  } catch (...) {
    tmpls = common_chat_templates_init(model, "chatml");
  }

  common_chat_templates_inputs inputs;
  inputs.use_jinja = true;
  inputs.add_generation_prompt = true;
  inputs.messages = common_chat_msgs_parse_oaicompat(body["messages"]);
  inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
  inputs.enable_thinking = true;

  if (body.contains("reasoning_format") && body["reasoning_format"].is_string()) {
    inputs.reasoning_format = common_reasoning_format_from_name(
        body["reasoning_format"].get<std::string>());
  }

  if (body.contains("tools")) {
    inputs.tools = common_chat_tools_parse_oaicompat(body["tools"]);
    inputs.tool_choice =
        body.contains("tool_choice") && body["tool_choice"].is_string()
            ? common_chat_tool_choice_parse_oaicompat(body["tool_choice"].get<std::string>())
            : COMMON_CHAT_TOOL_CHOICE_AUTO;
  }

  auto result = common_chat_templates_apply(tmpls.get(), inputs);
  chat_params = result;
  prompt = result.prompt;
  parser_params = common_chat_parser_params(result);
  parser_params.reasoning_format = inputs.reasoning_format;
  parser_params.reasoning_in_content =
      (inputs.reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY);
  if (!result.parser.empty()) {
    parser_params.parser.load(result.parser);
  }

  chat_format = common_chat_format_name(result.format);
  reasoning_format = common_reasoning_format_name(inputs.reasoning_format);
}

static glue_msg_server_context_poc_res run_completion(
    const std::string &request_json,
    const std::string &fallback_prompt,
    const std::string &jinja_template,
    const std::string &oaicompat_model,
    bool has_n_predict,
    int32_t n_predict,
    bool has_temp,
    float temp,
    bool has_top_p,
    float top_p,
    bool has_penalty_freq,
    float penalty_freq,
    bool has_penalty_repeat,
    float penalty_repeat)
{
  glue_msg_server_context_poc_res res;
  res.success.value = false;

  if (!g_server_context_poc.loaded()) {
    res.message.value = "server_context_poc is not loaded";
    return res;
  }

  std::string prompt = fallback_prompt;
  bool is_oai = false;
  common_chat_parser_params parser_params;
  std::string chat_format;
  std::string reasoning_format;
  common_chat_params chat_params;
  apply_request_json(
      *g_server_context_poc.srv,
      request_json,
      prompt,
      jinja_template,
      prompt,
      is_oai,
      parser_params,
      chat_format,
      reasoning_format,
      chat_params);

  auto reader = g_server_context_poc.srv->get_response_reader();

  server_task task(SERVER_TASK_TYPE_COMPLETION);
  task.id = reader.get_new_id();
  task.index = 0;
  task.cli = true;
  task.cli_prompt = prompt;

  task.params.stream = true;
  task.params.cache_prompt = true;
  task.params.n_predict = has_n_predict ? n_predict : g_server_context_poc.n_predict;
  if (has_temp) {
    task.params.sampling.temp = temp;
  }
  if (has_top_p) {
    task.params.sampling.top_p = top_p;
  }
  if (has_penalty_freq) {
    task.params.sampling.penalty_freq = penalty_freq;
  }
  if (has_penalty_repeat) {
    task.params.sampling.penalty_repeat = penalty_repeat;
  }
  if (is_oai) {
    task.params.sampling.grammar = chat_params.grammar;
    task.params.sampling.grammar_lazy = chat_params.grammar_lazy;
    task.params.sampling.grammar_triggers = chat_params.grammar_triggers;
    task.params.antiprompt = chat_params.additional_stops;
    auto *lctx = g_server_context_poc.srv->get_llama_context();
    auto *model = llama_get_model(lctx);
    auto *vocab = llama_model_get_vocab(model);
    for (const auto &preserved_token : chat_params.preserved_tokens) {
      auto ids = common_tokenize(vocab, preserved_token, false, true);
      if (ids.size() == 1) {
        task.params.sampling.preserved_tokens.insert(ids[0]);
      }
    }
  }
  std::random_device rd;
  task.params.sampling.seed = rd();

  if (is_oai) {
    task.params.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
    task.params.oaicompat_model = oaicompat_model.empty()
        ? g_server_context_poc.model_path
        : oaicompat_model;
    task.params.oaicompat_cmpl_id = gen_chatcmplid();
    task.params.chat_parser_params = parser_params;
  }

  fprintf(stderr,
      "wllama server_context_poc: posting task id=%d n_predict=%d prompt_bytes=%zu stream=%d\n",
      task.id,
      task.params.n_predict,
      prompt.size(),
      task.params.stream ? 1 : 0);
  reader.post_task(std::move(task));

  const int64_t t_start_ms = ggml_time_ms();
  constexpr int64_t max_wait_ms = 30000;
  bool timed_out = false;
  auto should_stop = [t_start_ms, &timed_out] {
    timed_out = ggml_time_ms() - t_start_ms > max_wait_ms;
    if (timed_out) {
      fprintf(stderr,
          "wllama server_context_poc: timed out waiting for server response after %lld ms\n",
          (long long) max_wait_ms);
    }
    return timed_out;
  };
  std::vector<std::string> chunks;
  auto collect_chunks = [&] {
    while (reader.has_next()) {
      server_task_result_ptr task_res = reader.next(should_stop);
      if (!task_res) {
        break;
      }
      auto chunk_json = task_res->to_json().dump();
      fprintf(stderr,
          "wllama server_context_poc: received chunk id=%d stop=%d error=%d bytes=%zu\n",
          task_res->id,
          task_res->is_stop() ? 1 : 0,
          task_res->is_error() ? 1 : 0,
          chunk_json.size());
      chunks.push_back(std::move(chunk_json));
      if (task_res->is_error()) {
        break;
      }
    }
  };

  if (g_server_context_poc.use_webgpu) {
    // WebGPU JS objects live in the worker that initialized the device. Running
    // server_context::start_loop() on a pthread makes emdawn look up an empty
    // JS object table and crash at queue.writeBuffer. For the WebGPU POC, run
    // the server loop on the current worker while a helper pthread waits for
    // response chunks and terminates the loop once the streamed completion ends.
    std::thread response_thread([&] {
      collect_chunks();
      g_server_context_poc.srv->terminate();
    });
    g_server_context_poc.srv->start_loop();
    if (response_thread.joinable()) {
      response_thread.join();
    }
  } else {
    collect_chunks();
  }

  res.success.value = !timed_out;
  if (timed_out) {
    res.message.value = "server_context_poc timed out waiting for completion";
  }
  res.prompt.value = prompt;
  res.chat_format.value = chat_format;
  res.reasoning_format.value = reasoning_format;
  res.chunks.arr = std::move(chunks);
  return res;
}

} // namespace
#endif // WLLAMA_ENABLE_SERVER_CONTEXT_POC

glue_msg_server_context_poc_load_res action_server_context_poc_load(app_t &app, const char *req_raw)
{
  (void) app;
  PARSE_REQ(glue_msg_server_context_poc_load_req);
  glue_msg_server_context_poc_load_res res;
  res.success.value = false;

#if !WLLAMA_ENABLE_SERVER_CONTEXT_POC
  res.message.value = "wllama was built with WLLAMA_ENABLE_SERVER_CONTEXT_POC=OFF";
  return res;
#else
  try {
    auto params = make_common_params(
        req.model_path.value,
        req.use_webgpu.not_null(), req.use_webgpu.not_null() ? req.use_webgpu.value : false,
        req.n_ctx.not_null(), req.n_ctx.not_null() ? req.n_ctx.value : 0,
        req.n_batch.not_null(), req.n_batch.not_null() ? req.n_batch.value : 0,
        req.n_ubatch.not_null(), req.n_ubatch.not_null() ? req.n_ubatch.value : 0,
        req.n_threads.not_null(), req.n_threads.not_null() ? req.n_threads.value : 0,
        req.n_gpu_layers.not_null(), req.n_gpu_layers.not_null() ? req.n_gpu_layers.value : 0,
        req.n_predict.not_null(), req.n_predict.not_null() ? req.n_predict.value : 0);
    load_server_context(
        std::move(params),
        req.use_webgpu.not_null(),
        req.use_webgpu.not_null() ? req.use_webgpu.value : false);
    res.success.value = true;
    return res;
  } catch (const std::exception &e) {
    res.success.value = false;
    res.message.value = e.what();
    return res;
  }
#endif
}

glue_msg_server_context_poc_res action_server_context_poc_completion(app_t &app, const char *req_raw)
{
  (void) app;
  PARSE_REQ(glue_msg_server_context_poc_completion_req);
  glue_msg_server_context_poc_res res;
  res.success.value = false;

#if !WLLAMA_ENABLE_SERVER_CONTEXT_POC
  res.message.value = "wllama was built with WLLAMA_ENABLE_SERVER_CONTEXT_POC=OFF";
  return res;
#else
  try {
    return run_completion(
        req.request_json.value,
        req.prompt.not_null() ? req.prompt.value : "",
        req.jinja_template.not_null() ? req.jinja_template.value : "",
        req.oaicompat_model.not_null() ? req.oaicompat_model.value : "",
        req.n_predict.not_null(), req.n_predict.not_null() ? req.n_predict.value : 0,
        req.temp.not_null(), req.temp.not_null() ? req.temp.value : 0.0f,
        req.top_p.not_null(), req.top_p.not_null() ? req.top_p.value : 0.0f,
        req.penalty_freq.not_null(), req.penalty_freq.not_null() ? req.penalty_freq.value : 0.0f,
        req.penalty_repeat.not_null(), req.penalty_repeat.not_null() ? req.penalty_repeat.value : 0.0f);
  } catch (const std::exception &e) {
    res.success.value = false;
    res.message.value = e.what();
    return res;
  }
#endif
}

glue_msg_server_context_poc_unload_res action_server_context_poc_unload(app_t &app, const char *req_raw)
{
  (void) app;
  PARSE_REQ(glue_msg_server_context_poc_unload_req);
  glue_msg_server_context_poc_unload_res res;
  res.success.value = false;

#if !WLLAMA_ENABLE_SERVER_CONTEXT_POC
  res.message.value = "wllama was built with WLLAMA_ENABLE_SERVER_CONTEXT_POC=OFF";
  return res;
#else
  g_server_context_poc.unload();
  res.success.value = true;
  return res;
#endif
}

glue_msg_server_context_poc_res action_server_context_poc(app_t &app, const char *req_raw)
{
  (void) app;
  PARSE_REQ(glue_msg_server_context_poc_req);
  glue_msg_server_context_poc_res res;
  res.success.value = false;

#if !WLLAMA_ENABLE_SERVER_CONTEXT_POC
  res.message.value = "wllama was built with WLLAMA_ENABLE_SERVER_CONTEXT_POC=OFF";
  return res;
#else
  try {
    auto params = make_common_params(
        req.model_path.value,
        req.use_webgpu.not_null(), req.use_webgpu.not_null() ? req.use_webgpu.value : false,
        req.n_ctx.not_null(), req.n_ctx.not_null() ? req.n_ctx.value : 0,
        req.n_batch.not_null(), req.n_batch.not_null() ? req.n_batch.value : 0,
        req.n_ubatch.not_null(), req.n_ubatch.not_null() ? req.n_ubatch.value : 0,
        req.n_threads.not_null(), req.n_threads.not_null() ? req.n_threads.value : 0,
        req.n_gpu_layers.not_null(), req.n_gpu_layers.not_null() ? req.n_gpu_layers.value : 0,
        req.n_predict.not_null(), req.n_predict.not_null() ? req.n_predict.value : 0);
    load_server_context(
        std::move(params),
        req.use_webgpu.not_null(),
        req.use_webgpu.not_null() ? req.use_webgpu.value : false);
    auto completion = run_completion(
        req.request_json.value,
        req.prompt.not_null() ? req.prompt.value : "",
        req.jinja_template.not_null() ? req.jinja_template.value : "",
        req.model_path.value,
        req.n_predict.not_null(), req.n_predict.not_null() ? req.n_predict.value : 0,
        req.temp.not_null(), req.temp.not_null() ? req.temp.value : 0.0f,
        req.top_p.not_null(), req.top_p.not_null() ? req.top_p.value : 0.0f,
        req.penalty_freq.not_null(), req.penalty_freq.not_null() ? req.penalty_freq.value : 0.0f,
        req.penalty_repeat.not_null(), req.penalty_repeat.not_null() ? req.penalty_repeat.value : 0.0f);
    g_server_context_poc.unload();
    return completion;
  } catch (const std::exception &e) {
    g_server_context_poc.unload();
    res.success.value = false;
    res.message.value = e.what();
    return res;
  }
#endif
}
