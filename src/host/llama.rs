use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Result, bail};
use encoding_rs::{Decoder, UTF_8};
use koharu_llama::context::LlamaContext;
use koharu_llama::context::params::LlamaContextParams;
use koharu_llama::llama_backend::LlamaBackend;
use koharu_llama::llama_batch::LlamaBatch;
use koharu_llama::model::params::LlamaModelParams;
use koharu_llama::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use koharu_llama::sampling::LlamaSampler;
use koharu_llama::token::LlamaToken;
use koharu_runtime::package::llama_cpp::LlamaCpp;
use koharu_runtime::package::{Package, PreloadablePackage};
use pd_host_function::pd_host_function;

use crate::{CallOutcome, Value, VmResult};

use super::{host_error, native, return_int, return_value};

static NEXT_HANDLE: AtomicI64 = AtomicI64::new(1);
static BACKENDS: LazyLock<Mutex<HashMap<i64, LlamaBackend>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static MODEL_PARAMS: LazyLock<Mutex<HashMap<i64, ModelParamsResource>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static MODELS: LazyLock<Mutex<HashMap<i64, Arc<LlamaModel>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CONTEXT_PARAMS: LazyLock<Mutex<HashMap<i64, LlamaContextParams>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CONTEXTS: LazyLock<Mutex<HashMap<i64, ContextResource>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static TOKEN_LISTS: LazyLock<Mutex<HashMap<i64, Vec<LlamaToken>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static BATCHES: LazyLock<Mutex<HashMap<i64, BatchResource>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SAMPLERS: LazyLock<Mutex<HashMap<i64, SamplerResource>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static DECODERS: LazyLock<Mutex<HashMap<i64, Decoder>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CHAT_TEMPLATES: LazyLock<Mutex<HashMap<i64, LlamaChatTemplate>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static CHAT_MESSAGES: LazyLock<Mutex<HashMap<i64, Vec<LlamaChatMessage>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

struct ContextResource {
    context: LlamaContext<'static>,
    _model: Arc<LlamaModel>,
}

struct ModelParamsResource(LlamaModelParams);

struct BatchResource(LlamaBatch<'static>);

struct SamplerResource {
    no_perf: bool,
    pending: Vec<LlamaSampler>,
    sampler: Option<LlamaSampler>,
}

// llama.cpp objects are used only while their registry mutex is held. The
// model Arc also outlives every context that borrows it.
unsafe impl Send for ContextResource {}
unsafe impl Send for ModelParamsResource {}
unsafe impl Send for BatchResource {}
unsafe impl Send for SamplerResource {}

fn next_handle() -> i64 {
    NEXT_HANDLE.fetch_add(1, Ordering::Relaxed)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::backend_init")]
pub(super) fn llama_backend_init_impl(kind: &str) -> VmResult<CallOutcome> {
    let directory = if Path::new(kind).is_dir() {
        let directory = Path::new(kind).to_path_buf();
        native::preload_directory(&directory, LLAMA_LIBRARY_PRELOAD_ORDER)
            .map_err(llama_host_error)?;
        directory
    } else {
        let package = select_package(kind).map_err(llama_host_error)?;
        let directory = native::block_on(package.resolve()).map_err(llama_host_error)?;
        native::block_on(package.preload()).map_err(llama_host_error)?;
        directory
    };
    LlamaBackend::load_all_backends_from_path(&directory).map_err(llama_host_error)?;
    let backend = LlamaBackend::init().map_err(llama_host_error)?;
    let handle = next_handle();
    BACKENDS
        .lock()
        .map_err(|_| registry_error("backend"))?
        .insert(handle, backend);
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::backend_supports_gpu_offload")]
pub(super) fn llama_backend_supports_gpu_offload_impl(handle: i64) -> VmResult<CallOutcome> {
    let backends = BACKENDS.lock().map_err(|_| registry_error("backend"))?;
    let backend = get(&backends, handle, "backend")?;
    return_value(Value::Bool(backend.supports_gpu_offload()))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::backend_list_devices")]
pub(super) fn llama_backend_list_devices_impl(handle: i64) -> VmResult<CallOutcome> {
    let backends = BACKENDS.lock().map_err(|_| registry_error("backend"))?;
    get(&backends, handle, "backend")?;
    let output = koharu_llama::list_llama_ggml_backend_devices()
        .into_iter()
        .map(|device| {
            format!(
                "{}\t{}\t{}\t{}\t{}",
                device.index, device.name, device.backend, device.description, device.memory_free
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    return_value(Value::String(output.into()))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::backend_free")]
pub(super) fn llama_backend_free_impl(handle: i64) -> VmResult<CallOutcome> {
    BACKENDS
        .lock()
        .map_err(|_| registry_error("backend"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("backend", handle))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_params_init")]
pub(super) fn llama_model_params_init_impl(backend_handle: i64) -> VmResult<CallOutcome> {
    ensure_backend(backend_handle)?;
    let handle = next_handle();
    MODEL_PARAMS
        .lock()
        .map_err(|_| registry_error("model params"))?
        .insert(handle, ModelParamsResource(LlamaModelParams::default()));
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_params_set_gpu_layers")]
pub(super) fn llama_model_params_set_gpu_layers_impl(
    handle: i64,
    layers: i64,
) -> VmResult<CallOutcome> {
    let layers = u32::try_from(layers)
        .map_err(|_| host_error("n_gpu_layers must be a non-negative uint32"))?;
    update_model_params(handle, |params| params.with_n_gpu_layers(layers))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_params_set_main_gpu")]
pub(super) fn llama_model_params_set_main_gpu_impl(
    handle: i64,
    main_gpu: i64,
) -> VmResult<CallOutcome> {
    let main_gpu = checked_i32(main_gpu, "main_gpu")?;
    update_model_params(handle, |params| params.with_main_gpu(main_gpu))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_params_set_memory")]
pub(super) fn llama_model_params_set_memory_impl(
    handle: i64,
    use_mmap: bool,
    use_mlock: bool,
) -> VmResult<CallOutcome> {
    update_model_params(handle, |params| {
        params.with_use_mmap(use_mmap).with_use_mlock(use_mlock)
    })?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_load")]
pub(super) fn llama_model_load_impl(
    backend_handle: i64,
    params_handle: i64,
    path: &str,
) -> VmResult<CallOutcome> {
    let backends = BACKENDS.lock().map_err(|_| registry_error("backend"))?;
    let backend = get(&backends, backend_handle, "backend")?;
    let params = MODEL_PARAMS
        .lock()
        .map_err(|_| registry_error("model params"))?
        .remove(&params_handle)
        .ok_or_else(|| unknown_handle("model params", params_handle))?
        .0;
    let model =
        LlamaModel::load_from_file(backend, Path::new(path), &params).map_err(llama_host_error)?;
    let handle = next_handle();
    MODELS
        .lock()
        .map_err(|_| registry_error("model"))?
        .insert(handle, Arc::new(model));
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_free")]
pub(super) fn llama_model_free_impl(handle: i64) -> VmResult<CallOutcome> {
    MODELS
        .lock()
        .map_err(|_| registry_error("model"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("model", handle))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_n_ctx_train")]
pub(super) fn llama_model_n_ctx_train_impl(handle: i64) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    return_int(i64::from(get(&models, handle, "model")?.n_ctx_train()))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_n_vocab")]
pub(super) fn llama_model_n_vocab_impl(handle: i64) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    return_int(i64::from(get(&models, handle, "model")?.n_vocab()))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_tokenize")]
pub(super) fn llama_model_tokenize_impl(
    model_handle: i64,
    text: &str,
    add_bos: bool,
) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    let model = get(&models, model_handle, "model")?;
    let add_bos = if add_bos {
        AddBos::Always
    } else {
        AddBos::Never
    };
    let tokens = model
        .str_to_token(text, add_bos)
        .map_err(llama_host_error)?;
    let handle = next_handle();
    TOKEN_LISTS
        .lock()
        .map_err(|_| registry_error("tokens"))?
        .insert(handle, tokens);
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::model_is_eog")]
pub(super) fn llama_model_is_eog_impl(model_handle: i64, token: i64) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    let model = get(&models, model_handle, "model")?;
    return_value(Value::Bool(
        model.is_eog_token(LlamaToken::new(checked_i32(token, "token")?)),
    ))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::chat_template")]
pub(super) fn llama_chat_template_impl(model_handle: i64, name: &str) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    let model = get(&models, model_handle, "model")?;
    let template = model
        .chat_template((!name.is_empty()).then_some(name))
        .map_err(llama_host_error)?;
    let handle = next_handle();
    CHAT_TEMPLATES
        .lock()
        .map_err(|_| registry_error("chat template"))?
        .insert(handle, template);
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::chat_messages_init")]
pub(super) fn llama_chat_messages_init_impl() -> VmResult<CallOutcome> {
    let handle = next_handle();
    CHAT_MESSAGES
        .lock()
        .map_err(|_| registry_error("chat messages"))?
        .insert(handle, Vec::new());
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::chat_messages_add")]
pub(super) fn llama_chat_messages_add_impl(
    handle: i64,
    role: &str,
    content: &str,
) -> VmResult<CallOutcome> {
    let mut messages = CHAT_MESSAGES
        .lock()
        .map_err(|_| registry_error("chat messages"))?;
    let messages = get_mut(&mut messages, handle, "chat messages")?;
    messages.push(
        LlamaChatMessage::new(role.to_owned(), content.to_owned()).map_err(llama_host_error)?,
    );
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::apply_chat_template")]
pub(super) fn llama_apply_chat_template_impl(
    model_handle: i64,
    template_handle: i64,
    messages_handle: i64,
    add_assistant: bool,
) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    let model = get(&models, model_handle, "model")?;
    let templates = CHAT_TEMPLATES
        .lock()
        .map_err(|_| registry_error("chat template"))?;
    let template = get(&templates, template_handle, "chat template")?;
    let messages = CHAT_MESSAGES
        .lock()
        .map_err(|_| registry_error("chat messages"))?;
    let messages = get(&messages, messages_handle, "chat messages")?;
    let prompt = model
        .apply_chat_template(template, messages, add_assistant)
        .map_err(llama_host_error)?;
    return_value(Value::String(prompt.into()))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::chat_free")]
pub(super) fn llama_chat_free_impl(handle: i64) -> VmResult<CallOutcome> {
    let removed_template = CHAT_TEMPLATES
        .lock()
        .map_err(|_| registry_error("chat template"))?
        .remove(&handle)
        .is_some();
    let removed_messages = CHAT_MESSAGES
        .lock()
        .map_err(|_| registry_error("chat messages"))?
        .remove(&handle)
        .is_some();
    if !removed_template && !removed_messages {
        return Err(unknown_handle("chat resource", handle));
    }
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::tokens_len")]
pub(super) fn llama_tokens_len_impl(handle: i64) -> VmResult<CallOutcome> {
    let lists = TOKEN_LISTS.lock().map_err(|_| registry_error("tokens"))?;
    let len = get(&lists, handle, "tokens")?.len();
    return_int(i64::try_from(len).map_err(|_| host_error("token count exceeds int64"))?)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::tokens_get")]
pub(super) fn llama_tokens_get_impl(handle: i64, index: i64) -> VmResult<CallOutcome> {
    let lists = TOKEN_LISTS.lock().map_err(|_| registry_error("tokens"))?;
    let list = get(&lists, handle, "tokens")?;
    let index =
        usize::try_from(index).map_err(|_| host_error("token index must be non-negative"))?;
    let token = list
        .get(index)
        .ok_or_else(|| host_error(format!("token index {index} is out of range")))?;
    return_int(i64::from(token.0))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::tokens_free")]
pub(super) fn llama_tokens_free_impl(handle: i64) -> VmResult<CallOutcome> {
    TOKEN_LISTS
        .lock()
        .map_err(|_| registry_error("tokens"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("tokens", handle))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_params_init")]
pub(super) fn llama_context_params_init_impl(backend_handle: i64) -> VmResult<CallOutcome> {
    ensure_backend(backend_handle)?;
    let handle = next_handle();
    CONTEXT_PARAMS
        .lock()
        .map_err(|_| registry_error("context params"))?
        .insert(handle, LlamaContextParams::default());
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_params_set_sizes")]
pub(super) fn llama_context_params_set_sizes_impl(
    handle: i64,
    n_ctx: i64,
    n_batch: i64,
    n_ubatch: i64,
) -> VmResult<CallOutcome> {
    let n_ctx = checked_u32(n_ctx, "n_ctx")?;
    let n_batch = checked_u32(n_batch, "n_batch")?;
    let n_ubatch = checked_u32(n_ubatch, "n_ubatch")?;
    let mut params = CONTEXT_PARAMS
        .lock()
        .map_err(|_| registry_error("context params"))?;
    let current = get_mut(&mut params, handle, "context params")?.clone();
    *get_mut(&mut params, handle, "context params")? = current
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_ubatch);
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_params_set_threads")]
pub(super) fn llama_context_params_set_threads_impl(
    handle: i64,
    n_threads: i64,
    n_threads_batch: i64,
) -> VmResult<CallOutcome> {
    let n_threads = checked_i32(n_threads, "n_threads")?;
    let n_threads_batch = checked_i32(n_threads_batch, "n_threads_batch")?;
    let mut params = CONTEXT_PARAMS
        .lock()
        .map_err(|_| registry_error("context params"))?;
    let current = get_mut(&mut params, handle, "context params")?.clone();
    *get_mut(&mut params, handle, "context params")? = current
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads_batch);
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_new")]
pub(super) fn llama_context_new_impl(
    model_handle: i64,
    backend_handle: i64,
    params_handle: i64,
) -> VmResult<CallOutcome> {
    let model = MODELS
        .lock()
        .map_err(|_| registry_error("model"))?
        .get(&model_handle)
        .cloned()
        .ok_or_else(|| unknown_handle("model", model_handle))?;
    let backends = BACKENDS.lock().map_err(|_| registry_error("backend"))?;
    let backend = get(&backends, backend_handle, "backend")?;
    let params = CONTEXT_PARAMS
        .lock()
        .map_err(|_| registry_error("context params"))?
        .remove(&params_handle)
        .ok_or_else(|| unknown_handle("context params", params_handle))?;

    // Arc keeps the allocation fixed until the context is dropped. The
    // resource field order drops the context before its model Arc.
    let model_ref: &'static LlamaModel = unsafe { &*Arc::as_ptr(&model) };
    let context = model_ref
        .new_context(backend, params)
        .map_err(llama_host_error)?;
    let handle = next_handle();
    CONTEXTS
        .lock()
        .map_err(|_| registry_error("context"))?
        .insert(
            handle,
            ContextResource {
                context,
                _model: model,
            },
        );
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_n_ctx")]
pub(super) fn llama_context_n_ctx_impl(handle: i64) -> VmResult<CallOutcome> {
    let contexts = CONTEXTS.lock().map_err(|_| registry_error("context"))?;
    return_int(i64::from(
        get(&contexts, handle, "context")?.context.n_ctx(),
    ))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_decode")]
pub(super) fn llama_context_decode_impl(
    context_handle: i64,
    batch_handle: i64,
) -> VmResult<CallOutcome> {
    let mut contexts = CONTEXTS.lock().map_err(|_| registry_error("context"))?;
    let context = &mut get_mut(&mut contexts, context_handle, "context")?.context;
    let mut batches = BATCHES.lock().map_err(|_| registry_error("batch"))?;
    let batch = &mut get_mut(&mut batches, batch_handle, "batch")?.0;
    context.decode(batch).map_err(llama_host_error)?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::context_free")]
pub(super) fn llama_context_free_impl(handle: i64) -> VmResult<CallOutcome> {
    CONTEXTS
        .lock()
        .map_err(|_| registry_error("context"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("context", handle))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::batch_init")]
pub(super) fn llama_batch_init_impl(capacity: i64, n_seq_max: i64) -> VmResult<CallOutcome> {
    let capacity =
        usize::try_from(capacity).map_err(|_| host_error("batch capacity must be non-negative"))?;
    let n_seq_max = checked_i32(n_seq_max, "n_seq_max")?;
    let handle = next_handle();
    BATCHES
        .lock()
        .map_err(|_| registry_error("batch"))?
        .insert(handle, BatchResource(LlamaBatch::new(capacity, n_seq_max)));
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::batch_add")]
pub(super) fn llama_batch_add_impl(
    handle: i64,
    token: i64,
    position: i64,
    sequence: i64,
    logits: bool,
) -> VmResult<CallOutcome> {
    let mut batches = BATCHES.lock().map_err(|_| registry_error("batch"))?;
    get_mut(&mut batches, handle, "batch")?
        .0
        .add(
            LlamaToken::new(checked_i32(token, "token")?),
            checked_i32(position, "position")?,
            &[checked_i32(sequence, "sequence")?],
            logits,
        )
        .map_err(llama_host_error)?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::batch_add_sequence")]
pub(super) fn llama_batch_add_sequence_impl(
    batch_handle: i64,
    tokens_handle: i64,
    sequence: i64,
    logits_all: bool,
) -> VmResult<CallOutcome> {
    let lists = TOKEN_LISTS.lock().map_err(|_| registry_error("tokens"))?;
    let tokens = get(&lists, tokens_handle, "tokens")?;
    let mut batches = BATCHES.lock().map_err(|_| registry_error("batch"))?;
    get_mut(&mut batches, batch_handle, "batch")?
        .0
        .add_sequence(tokens, checked_i32(sequence, "sequence")?, logits_all)
        .map_err(llama_host_error)?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::batch_clear")]
pub(super) fn llama_batch_clear_impl(handle: i64) -> VmResult<CallOutcome> {
    let mut batches = BATCHES.lock().map_err(|_| registry_error("batch"))?;
    get_mut(&mut batches, handle, "batch")?.0.clear();
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::batch_free")]
pub(super) fn llama_batch_free_impl(handle: i64) -> VmResult<CallOutcome> {
    BATCHES
        .lock()
        .map_err(|_| registry_error("batch"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("batch", handle))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_chain_init")]
pub(super) fn llama_sampler_chain_init_impl(no_perf: bool) -> VmResult<CallOutcome> {
    let handle = next_handle();
    SAMPLERS
        .lock()
        .map_err(|_| registry_error("sampler"))?
        .insert(
            handle,
            SamplerResource {
                no_perf,
                pending: Vec::new(),
                sampler: None,
            },
        );
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_add_top_k")]
pub(super) fn llama_sampler_add_top_k_impl(handle: i64, k: i64) -> VmResult<CallOutcome> {
    add_sampler(handle, LlamaSampler::top_k(checked_i32(k, "top_k")?))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_add_top_p")]
pub(super) fn llama_sampler_add_top_p_impl(
    handle: i64,
    p: f64,
    min_keep: i64,
) -> VmResult<CallOutcome> {
    add_sampler(
        handle,
        LlamaSampler::top_p(p as f32, checked_usize(min_keep, "min_keep")?),
    )?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_add_min_p")]
pub(super) fn llama_sampler_add_min_p_impl(
    handle: i64,
    p: f64,
    min_keep: i64,
) -> VmResult<CallOutcome> {
    add_sampler(
        handle,
        LlamaSampler::min_p(p as f32, checked_usize(min_keep, "min_keep")?),
    )?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_add_temp")]
pub(super) fn llama_sampler_add_temp_impl(handle: i64, temperature: f64) -> VmResult<CallOutcome> {
    add_sampler(handle, LlamaSampler::temp(temperature as f32))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_add_dist")]
pub(super) fn llama_sampler_add_dist_impl(handle: i64, seed: i64) -> VmResult<CallOutcome> {
    add_sampler(handle, LlamaSampler::dist(checked_u32(seed, "seed")?))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_add_greedy")]
pub(super) fn llama_sampler_add_greedy_impl(handle: i64) -> VmResult<CallOutcome> {
    add_sampler(handle, LlamaSampler::greedy())?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_chain_build")]
pub(super) fn llama_sampler_chain_build_impl(handle: i64) -> VmResult<CallOutcome> {
    let mut samplers = SAMPLERS.lock().map_err(|_| registry_error("sampler"))?;
    let resource = get_mut(&mut samplers, handle, "sampler")?;
    if resource.sampler.is_some() {
        return Err(host_error("sampler chain is already built"));
    }
    if resource.pending.is_empty() {
        return Err(host_error("sampler chain has no components"));
    }
    let pending = std::mem::take(&mut resource.pending);
    resource.sampler = Some(LlamaSampler::chain(pending, resource.no_perf));
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_sample")]
pub(super) fn llama_sampler_sample_impl(
    sampler_handle: i64,
    context_handle: i64,
    index: i64,
) -> VmResult<CallOutcome> {
    let contexts = CONTEXTS.lock().map_err(|_| registry_error("context"))?;
    let context = &get(&contexts, context_handle, "context")?.context;
    let mut samplers = SAMPLERS.lock().map_err(|_| registry_error("sampler"))?;
    let sampler = ready_sampler(get_mut(&mut samplers, sampler_handle, "sampler")?)?;
    let token = sampler.sample(context, checked_i32(index, "index")?);
    return_int(i64::from(token.0))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_accept")]
pub(super) fn llama_sampler_accept_impl(sampler_handle: i64, token: i64) -> VmResult<CallOutcome> {
    let mut samplers = SAMPLERS.lock().map_err(|_| registry_error("sampler"))?;
    let sampler = ready_sampler(get_mut(&mut samplers, sampler_handle, "sampler")?)?;
    sampler
        .try_accept(LlamaToken::new(checked_i32(token, "token")?))
        .map_err(llama_host_error)?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::sampler_free")]
pub(super) fn llama_sampler_free_impl(handle: i64) -> VmResult<CallOutcome> {
    SAMPLERS
        .lock()
        .map_err(|_| registry_error("sampler"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("sampler", handle))?;
    return_value(Value::Bool(true))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::decoder_init")]
pub(super) fn llama_decoder_init_impl() -> VmResult<CallOutcome> {
    let handle = next_handle();
    DECODERS
        .lock()
        .map_err(|_| registry_error("decoder"))?
        .insert(handle, UTF_8.new_decoder());
    return_int(handle)
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::decoder_push")]
pub(super) fn llama_decoder_push_impl(
    decoder_handle: i64,
    model_handle: i64,
    token: i64,
    special: bool,
) -> VmResult<CallOutcome> {
    let models = MODELS.lock().map_err(|_| registry_error("model"))?;
    let model = get(&models, model_handle, "model")?;
    let mut decoders = DECODERS.lock().map_err(|_| registry_error("decoder"))?;
    let decoder = get_mut(&mut decoders, decoder_handle, "decoder")?;
    let piece = model
        .token_to_piece(
            LlamaToken::new(checked_i32(token, "token")?),
            decoder,
            special,
            None,
        )
        .map_err(llama_host_error)?;
    return_value(Value::String(piece.into()))
}

/// Exposes the corresponding koharu-llama operation to RustScript.
#[pd_host_function(name = "flint::llama::decoder_free")]
pub(super) fn llama_decoder_free_impl(handle: i64) -> VmResult<CallOutcome> {
    DECODERS
        .lock()
        .map_err(|_| registry_error("decoder"))?
        .remove(&handle)
        .ok_or_else(|| unknown_handle("decoder", handle))?;
    return_value(Value::Bool(true))
}

fn update_model_params(
    handle: i64,
    update: impl FnOnce(LlamaModelParams) -> LlamaModelParams,
) -> VmResult<()> {
    let mut params = MODEL_PARAMS
        .lock()
        .map_err(|_| registry_error("model params"))?;
    let params = &mut get_mut(&mut params, handle, "model params")?.0;
    let current = std::mem::take(params);
    *params = update(current);
    Ok(())
}

fn add_sampler(handle: i64, sampler: LlamaSampler) -> VmResult<()> {
    let mut samplers = SAMPLERS.lock().map_err(|_| registry_error("sampler"))?;
    let resource = get_mut(&mut samplers, handle, "sampler")?;
    if resource.sampler.is_some() {
        return Err(host_error(
            "cannot add a component after sampler chain build",
        ));
    }
    resource.pending.push(sampler);
    Ok(())
}

fn ready_sampler(resource: &mut SamplerResource) -> VmResult<&mut LlamaSampler> {
    resource
        .sampler
        .as_mut()
        .ok_or_else(|| host_error("sampler chain has not been built"))
}

fn ensure_backend(handle: i64) -> VmResult<()> {
    let backends = BACKENDS.lock().map_err(|_| registry_error("backend"))?;
    get(&backends, handle, "backend")?;
    Ok(())
}

fn select_package(kind: &str) -> Result<LlamaCpp> {
    let kind = kind.to_ascii_lowercase();
    if kind.is_empty() || kind == "auto" || kind.starts_with("cuda") {
        return Ok(LlamaCpp::for_current_target());
    }
    if kind == "cpu" {
        if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            return Ok(LlamaCpp::WindowsX64Cpu);
        }
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            return Ok(LlamaCpp::LinuxX64Cpu);
        }
        if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            return Ok(LlamaCpp::LinuxArm64Cpu);
        }
    }
    if kind.starts_with("vulkan") {
        if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
            return Ok(LlamaCpp::WindowsX64Vulkan);
        }
        if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
            return Ok(LlamaCpp::LinuxX64Vulkan);
        }
        if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
            return Ok(LlamaCpp::LinuxArm64Vulkan);
        }
    }
    bail!("unsupported llama.cpp backend '{kind}' for this target")
}

const LLAMA_LIBRARY_PRELOAD_ORDER: &[&str] = &[
    "cudart64_13.dll",
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "libomp140.x86_64.dll",
    "ggml-base.dll",
    "ggml.dll",
    "ggml-cpu-x64.dll",
    "ggml-cuda.dll",
    "ggml-vulkan.dll",
    "ggml-rpc.dll",
    "llama.dll",
    "llama-common.dll",
    "mtmd.dll",
];

fn get<'a, T>(map: &'a HashMap<i64, T>, handle: i64, kind: &str) -> VmResult<&'a T> {
    map.get(&handle).ok_or_else(|| unknown_handle(kind, handle))
}

fn get_mut<'a, T>(map: &'a mut HashMap<i64, T>, handle: i64, kind: &str) -> VmResult<&'a mut T> {
    map.get_mut(&handle)
        .ok_or_else(|| unknown_handle(kind, handle))
}

fn checked_i32(value: i64, name: &str) -> VmResult<i32> {
    i32::try_from(value).map_err(|_| host_error(format!("{name} is out of range for int32")))
}

fn checked_u32(value: i64, name: &str) -> VmResult<u32> {
    u32::try_from(value).map_err(|_| host_error(format!("{name} is out of range for uint32")))
}

fn checked_usize(value: i64, name: &str) -> VmResult<usize> {
    usize::try_from(value).map_err(|_| host_error(format!("{name} must be non-negative")))
}

fn llama_host_error(error: impl std::fmt::Display) -> vm::VmError {
    host_error(format!("llama error: {error}"))
}

fn registry_error(kind: &str) -> vm::VmError {
    host_error(format!("llama {kind} registry is poisoned"))
}

fn unknown_handle(kind: &str, handle: i64) -> vm::VmError {
    host_error(format!("unknown llama {kind} handle {handle}"))
}
