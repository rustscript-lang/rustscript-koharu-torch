mod cache;
mod ggml;
mod llama;
mod native;
mod pair;
mod runtime;
mod sd;

use std::cell::{Cell, UnsafeCell};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use image::imageops::FilterType;
use koharu_torch::{Device, Kind, Tensor};
use tokenizers::Tokenizer;
use vm::{
    CallOutcome, CallReturn, HostArgsFunction, Program, Value, Vm, VmError, VmResult, VmStatus,
    jit::JitConfig,
};

#[derive(Clone, Copy)]
enum HostOp {
    Context(fn(&mut TorchContext, &[Value]) -> VmResult<CallOutcome>),
    Static(fn(&[Value]) -> VmResult<CallOutcome>),
}

struct BoundHost {
    context: Arc<TorchContextCell>,
    name: &'static str,
    op: HostOp,
}

impl HostArgsFunction for BoundHost {
    fn call(&mut self, args: &[Value]) -> VmResult<CallOutcome> {
        let context = self.context.get();
        let previous_host_op = context.active_host_op.replace(self.name);
        if context.host_op_profile_enabled {
            let started = Instant::now();
            let outcome = self.op.call(context, args);
            context.record_host_op(self.name, started.elapsed());
            context.active_host_op = previous_host_op;
            outcome
        } else {
            let outcome = self.op.call(context, args);
            context.active_host_op = previous_host_op;
            outcome
        }
    }
}

impl HostOp {
    fn call(self, context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
        match self {
            Self::Context(op) => op(context, args),
            Self::Static(op) => op(args),
        }
    }
}

thread_local! {
    static CURRENT_CONTEXT: Cell<*mut TorchContext> = const { Cell::new(std::ptr::null_mut()) };
}

struct TorchContextCell {
    inner: UnsafeCell<TorchContext>,
}

// `TorchHostRuntime::execution` serializes every run, so host calls for a
// runtime cannot mutate this context concurrently.
unsafe impl Send for TorchContextCell {}
unsafe impl Sync for TorchContextCell {}

impl TorchContextCell {
    fn new(context: TorchContext) -> Self {
        Self {
            inner: UnsafeCell::new(context),
        }
    }

    fn get(&self) -> &mut TorchContext {
        unsafe { &mut *self.inner.get() }
    }

    fn enter(&self) -> CurrentContextGuard {
        let next = self.inner.get();
        let previous = CURRENT_CONTEXT.replace(next);
        CurrentContextGuard { previous }
    }
}

struct CurrentContextGuard {
    previous: *mut TorchContext,
}

impl Drop for CurrentContextGuard {
    fn drop(&mut self) {
        CURRENT_CONTEXT.set(self.previous);
    }
}

fn with_context<T>(op: impl FnOnce(&mut TorchContext) -> T) -> T {
    CURRENT_CONTEXT.with(|slot| {
        let pointer = slot.get();
        assert!(
            !pointer.is_null(),
            "flint host context is not active for this thread"
        );
        unsafe { op(&mut *pointer) }
    })
}

#[derive(Debug, Clone, Copy)]
struct FfcPair {
    local: i64,
    global: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RopeKind {
    Cos,
    Sin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RopeCacheKey {
    kind: RopeKind,
    head_dim: usize,
    theta_bits: u32,
}

struct Conv1dStepWeights {
    w0: Tensor,
    w1: Tensor,
    w2: Tensor,
}

impl Clone for Conv1dStepWeights {
    fn clone(&self) -> Self {
        Self {
            w0: self.w0.shallow_clone(),
            w1: self.w1.shallow_clone(),
            w2: self.w2.shallow_clone(),
        }
    }
}

struct TorchContext {
    device: Device,
    weights: HashMap<String, Tensor>,
    weight_handles: HashMap<String, i64>,
    weights_path: Option<PathBuf>,
    weights_kind: Option<Kind>,
    tokenizer: Option<Tokenizer>,
    tokenizer_path: Option<PathBuf>,
    cache: HashMap<String, Tensor>,
    rope_cache: HashMap<RopeCacheKey, Tensor>,
    conv1d_step_weights: HashMap<i64, Conv1dStepWeights>,
    tensors: HashMap<i64, Tensor>,
    missing_weight_handles: HashMap<i64, String>,
    pairs: HashMap<i64, FfcPair>,
    inputs: Vec<i64>,
    args: Vec<String>,
    output: Option<i64>,
    text_output: Option<String>,
    generation_started_at: Option<Instant>,
    generated_tokens: Option<i64>,
    decode_started_at: Option<Instant>,
    decode_tokens: Option<i64>,
    generated_token_tensors: Vec<Tensor>,
    host_op_profile_enabled: bool,
    host_op_stats: HashMap<&'static str, HostOpStats>,
    active_host_op: Option<&'static str>,
    next_tensor: i64,
    next_missing_weight: i64,
    next_pair: i64,
}

#[derive(Default)]
struct HostOpStats {
    count: u64,
    elapsed: Duration,
}

impl TorchContext {
    fn insert_tensor(&mut self, tensor: Tensor) -> i64 {
        let handle = self.next_tensor;
        self.next_tensor += 1;
        self.tensors.insert(handle, tensor);
        handle
    }

    fn tensor(&self, handle: i64) -> VmResult<&Tensor> {
        if let Some(name) = self.missing_weight_handles.get(&handle) {
            return Err(host_error(format!("missing optional weight '{name}'")));
        }
        self.tensors
            .get(&handle)
            .ok_or_else(|| match self.active_host_op {
                Some(op) => host_error(format!("unknown tensor handle {handle} in {op}")),
                None => host_error(format!("unknown tensor handle {handle}")),
            })
    }

    fn weight(&self, name: &str) -> VmResult<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| host_error(format!("missing weight '{name}'")))
    }

    fn insert_pair(&mut self, pair: FfcPair) -> i64 {
        let handle = self.next_pair;
        self.next_pair += 1;
        self.pairs.insert(handle, pair);
        handle
    }

    fn pair(&self, handle: i64) -> VmResult<FfcPair> {
        self.pairs
            .get(&handle)
            .copied()
            .ok_or_else(|| host_error(format!("unknown FFC pair handle {handle}")))
    }

    fn begin(&mut self, image: Tensor, mask: Tensor, args: Vec<String>) {
        self.begin_args(args);
        let image = self.insert_tensor(image);
        let mask = self.insert_tensor(mask);
        self.inputs.extend([image, mask]);
    }

    fn begin_args(&mut self, args: Vec<String>) {
        self.tensors.clear();
        self.missing_weight_handles.clear();
        self.pairs.clear();
        self.inputs.clear();
        self.cache.clear();
        self.weight_handles.clear();
        self.rope_cache.clear();
        self.conv1d_step_weights.clear();
        self.output = None;
        self.text_output = None;
        self.generation_started_at = None;
        self.generated_tokens = None;
        self.decode_started_at = None;
        self.decode_tokens = None;
        self.generated_token_tensors.clear();
        self.host_op_stats.clear();
        self.args = args;
        self.next_tensor = 1;
        self.next_missing_weight = -1;
        self.next_pair = 1;
    }

    fn record_host_op(&mut self, name: &'static str, elapsed: Duration) {
        let stats = self.host_op_stats.entry(name).or_default();
        stats.count += 1;
        stats.elapsed += elapsed;
    }

    fn print_host_op_stats(&self) {
        if !self.host_op_profile_enabled || self.host_op_stats.is_empty() {
            return;
        }
        let mut entries = self.host_op_stats.iter().collect::<Vec<_>>();
        entries.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.elapsed.as_nanos()));
        eprintln!("host op profile:");
        for (name, stats) in entries {
            eprintln!(
                "  {name}: count={}, {:.3} ms",
                stats.count,
                stats.elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    fn finish(&mut self) -> Result<Tensor> {
        let handle = self
            .output
            .context("RustScript did not publish an output tensor")?;
        let output = self
            .tensors
            .get(&handle)
            .with_context(|| format!("RustScript returned unknown tensor handle {handle}"))?
            .shallow_clone();
        self.tensors.clear();
        self.missing_weight_handles.clear();
        self.pairs.clear();
        self.inputs.clear();
        self.cache.clear();
        self.weight_handles.clear();
        self.rope_cache.clear();
        self.conv1d_step_weights.clear();
        self.args.clear();
        self.output = None;
        self.text_output = None;
        self.generation_started_at = None;
        self.generated_tokens = None;
        self.decode_started_at = None;
        self.decode_tokens = None;
        self.print_host_op_stats();
        Ok(output)
    }

    fn finish_text(&mut self) -> ScriptTextOutput {
        let text = self.text_output.take().unwrap_or_default();
        let elapsed = self
            .generation_started_at
            .take()
            .map(|start| start.elapsed());
        let generated_tokens = self.generated_tokens.take();
        let decode_elapsed = self.decode_started_at.take().map(|start| start.elapsed());
        let decode_tokens = self.decode_tokens.take();
        self.tensors.clear();
        self.missing_weight_handles.clear();
        self.pairs.clear();
        self.inputs.clear();
        self.cache.clear();
        self.weight_handles.clear();
        self.rope_cache.clear();
        self.conv1d_step_weights.clear();
        self.args.clear();
        self.output = None;
        self.print_host_op_stats();
        ScriptTextOutput {
            text,
            generated_tokens,
            elapsed,
            decode_tokens,
            decode_elapsed,
        }
    }
}

pub(crate) struct TorchHostRuntime {
    context: Arc<TorchContextCell>,
    execution: Mutex<()>,
}

impl TorchHostRuntime {
    pub(crate) fn new(device: Device) -> Self {
        Self {
            context: Arc::new(TorchContextCell::new(TorchContext {
                device,
                weights: HashMap::new(),
                weight_handles: HashMap::new(),
                weights_path: None,
                weights_kind: None,
                tokenizer: None,
                tokenizer_path: None,
                cache: HashMap::new(),
                rope_cache: HashMap::new(),
                conv1d_step_weights: HashMap::new(),
                tensors: HashMap::new(),
                missing_weight_handles: HashMap::new(),
                pairs: HashMap::new(),
                inputs: Vec::new(),
                args: Vec::new(),
                output: None,
                text_output: None,
                generation_started_at: None,
                generated_tokens: None,
                decode_started_at: None,
                decode_tokens: None,
                generated_token_tensors: Vec::new(),
                host_op_profile_enabled: std::env::var_os("KOHARU_TORCH_PROFILE_OPS").is_some(),
                host_op_stats: HashMap::new(),
                active_host_op: None,
                next_tensor: 1,
                next_missing_weight: -1,
                next_pair: 1,
            })),
            execution: Mutex::new(()),
        }
    }

    pub(crate) fn run(
        &self,
        program: Arc<Program>,
        image: Tensor,
        mask: Tensor,
        args: Vec<String>,
    ) -> Result<Tensor> {
        let _execution = self
            .execution
            .lock()
            .map_err(|_| anyhow!("Torch execution lock is poisoned"))?;
        self.context.get().begin(image, mask, args);
        let mut vm = Vm::new_shared_with_jit_config(program, torch_jit_config());
        self.bind(&mut vm);
        let _current_context = self.context.enter();
        let status = koharu_torch::no_grad(|| vm.run()).map_err(|err| anyhow!(err.to_string()))?;
        if status != VmStatus::Halted {
            bail!("RustScript did not halt: {status:?}");
        }
        self.context.get().finish()
    }

    fn run_text(
        &self,
        program: Arc<Program>,
        args: Vec<String>,
        mode: ScriptExecutionMode,
    ) -> Result<ScriptTextOutput> {
        let _execution = self
            .execution
            .lock()
            .map_err(|_| anyhow!("Script execution lock is poisoned"))?;
        self.context.get().begin_args(args);
        let mut vm = Vm::new_shared_with_jit_config(program, torch_jit_config());
        self.bind(&mut vm);
        let _current_context = self.context.enter();
        let status = match mode {
            ScriptExecutionMode::Native => vm.run(),
            ScriptExecutionMode::Torch => koharu_torch::no_grad(|| vm.run()),
        }
        .map_err(|err| anyhow!(err.to_string()))?;
        if status != VmStatus::Halted {
            bail!("RustScript did not halt: {status:?}");
        }
        Ok(self.context.get().finish_text())
    }

    fn bind(&self, vm: &mut Vm) {
        for (name, op) in HOST_OPS {
            vm.bind_args_function(
                *name,
                Box::new(BoundHost {
                    context: Arc::clone(&self.context),
                    name: *name,
                    op: *op,
                }),
            );
        }
    }
}

fn torch_jit_config() -> JitConfig {
    JitConfig {
        hot_loop_threshold: 1,
        max_trace_len: 512,
        ..JitConfig::default()
    }
}

pub struct ScriptRunner {
    runtime: TorchHostRuntime,
    mode: ScriptExecutionMode,
}

#[derive(Clone, Copy)]
enum ScriptExecutionMode {
    Native,
    Torch,
}

pub struct ScriptTextOutput {
    pub text: String,
    pub generated_tokens: Option<i64>,
    pub elapsed: Option<Duration>,
    pub decode_tokens: Option<i64>,
    pub decode_elapsed: Option<Duration>,
}

impl ScriptRunner {
    pub fn new() -> Self {
        Self {
            runtime: TorchHostRuntime::new(Device::Cpu),
            mode: ScriptExecutionMode::Native,
        }
    }

    pub async fn with_device(device: Device) -> Result<Self> {
        crate::preload_libtorch()
            .await
            .context("failed to initialize LibTorch runtime")?;
        Ok(Self {
            runtime: TorchHostRuntime::new(device),
            mode: ScriptExecutionMode::Torch,
        })
    }

    pub fn run_text(&self, program: Arc<Program>, args: Vec<String>) -> Result<ScriptTextOutput> {
        self.runtime.run_text(program, args, self.mode)
    }
}

impl Default for ScriptRunner {
    fn default() -> Self {
        Self::new()
    }
}

const HOST_OPS: &[(&str, HostOp)] = &[
    ("flint::runtime::arg", HostOp::Static(runtime::runtime_arg)),
    (
        "flint::runtime::arg_int",
        HostOp::Static(runtime::runtime_arg_int),
    ),
    (
        "flint::runtime::arg_int_or",
        HostOp::Static(runtime::runtime_arg_int_or),
    ),
    (
        "flint::runtime::arg_float_or",
        HostOp::Static(runtime::runtime_arg_float_or),
    ),
    (
        "flint::runtime::arg_or",
        HostOp::Static(runtime::runtime_arg_or),
    ),
    (
        "flint::runtime::input",
        HostOp::Static(runtime::runtime_input),
    ),
    (
        "flint::runtime::set_output",
        HostOp::Static(runtime::runtime_set_output),
    ),
    (
        "flint::runtime::set_text_output",
        HostOp::Static(runtime::runtime_set_text_output),
    ),
    (
        "flint::runtime::start_timer",
        HostOp::Static(runtime::runtime_start_timer),
    ),
    (
        "flint::runtime::start_decode_timer",
        HostOp::Static(runtime::runtime_start_decode_timer),
    ),
    (
        "flint::runtime::set_token_count",
        HostOp::Static(runtime::runtime_set_token_count),
    ),
    (
        "flint::runtime::set_decode_token_count",
        HostOp::Static(runtime::runtime_set_decode_token_count),
    ),
    (
        "flint::runtime::compact2",
        HostOp::Static(runtime::runtime_compact2),
    ),
    ("flint::cache::clear", HostOp::Static(cache::cache_clear)),
    ("flint::cache::has", HostOp::Static(cache::cache_has)),
    ("flint::cache::get", HostOp::Static(cache::cache_get)),
    ("flint::cache::set", HostOp::Static(cache::cache_set)),
    (
        "flint::ggml::load_backends",
        HostOp::Static(ggml::ggml_load_backends),
    ),
    (
        "flint::ggml::list_devices",
        HostOp::Static(ggml::ggml_list_devices),
    ),
    (
        "flint::ggml::stable_diffusion_package_dir",
        HostOp::Static(ggml::ggml_stable_diffusion_package_dir),
    ),
    (
        "flint::ggml::load_stable_diffusion_backends",
        HostOp::Static(ggml::ggml_load_stable_diffusion_backends),
    ),
    (
        "flint::ggml::list_stable_diffusion_devices",
        HostOp::Static(ggml::ggml_list_stable_diffusion_devices),
    ),
    (
        "flint::llama::backend_init",
        HostOp::Static(llama::llama_backend_init),
    ),
    (
        "flint::llama::backend_supports_gpu_offload",
        HostOp::Static(llama::llama_backend_supports_gpu_offload),
    ),
    (
        "flint::llama::backend_list_devices",
        HostOp::Static(llama::llama_backend_list_devices),
    ),
    (
        "flint::llama::backend_free",
        HostOp::Static(llama::llama_backend_free),
    ),
    (
        "flint::llama::model_params_init",
        HostOp::Static(llama::llama_model_params_init),
    ),
    (
        "flint::llama::model_params_set_gpu_layers",
        HostOp::Static(llama::llama_model_params_set_gpu_layers),
    ),
    (
        "flint::llama::model_params_set_main_gpu",
        HostOp::Static(llama::llama_model_params_set_main_gpu),
    ),
    (
        "flint::llama::model_params_set_memory",
        HostOp::Static(llama::llama_model_params_set_memory),
    ),
    (
        "flint::llama::model_load",
        HostOp::Static(llama::llama_model_load),
    ),
    (
        "flint::llama::model_free",
        HostOp::Static(llama::llama_model_free),
    ),
    (
        "flint::llama::model_n_ctx_train",
        HostOp::Static(llama::llama_model_n_ctx_train),
    ),
    (
        "flint::llama::model_n_vocab",
        HostOp::Static(llama::llama_model_n_vocab),
    ),
    (
        "flint::llama::model_tokenize",
        HostOp::Static(llama::llama_model_tokenize),
    ),
    (
        "flint::llama::model_is_eog",
        HostOp::Static(llama::llama_model_is_eog),
    ),
    (
        "flint::llama::chat_template",
        HostOp::Static(llama::llama_chat_template),
    ),
    (
        "flint::llama::chat_messages_init",
        HostOp::Static(llama::llama_chat_messages_init),
    ),
    (
        "flint::llama::chat_messages_add",
        HostOp::Static(llama::llama_chat_messages_add),
    ),
    (
        "flint::llama::apply_chat_template",
        HostOp::Static(llama::llama_apply_chat_template),
    ),
    (
        "flint::llama::chat_free",
        HostOp::Static(llama::llama_chat_free),
    ),
    (
        "flint::llama::tokens_len",
        HostOp::Static(llama::llama_tokens_len),
    ),
    (
        "flint::llama::tokens_get",
        HostOp::Static(llama::llama_tokens_get),
    ),
    (
        "flint::llama::tokens_free",
        HostOp::Static(llama::llama_tokens_free),
    ),
    (
        "flint::llama::context_params_init",
        HostOp::Static(llama::llama_context_params_init),
    ),
    (
        "flint::llama::context_params_set_sizes",
        HostOp::Static(llama::llama_context_params_set_sizes),
    ),
    (
        "flint::llama::context_params_set_threads",
        HostOp::Static(llama::llama_context_params_set_threads),
    ),
    (
        "flint::llama::context_new",
        HostOp::Static(llama::llama_context_new),
    ),
    (
        "flint::llama::context_n_ctx",
        HostOp::Static(llama::llama_context_n_ctx),
    ),
    (
        "flint::llama::context_decode",
        HostOp::Static(llama::llama_context_decode),
    ),
    (
        "flint::llama::context_free",
        HostOp::Static(llama::llama_context_free),
    ),
    (
        "flint::llama::batch_init",
        HostOp::Static(llama::llama_batch_init),
    ),
    (
        "flint::llama::batch_add",
        HostOp::Static(llama::llama_batch_add),
    ),
    (
        "flint::llama::batch_add_sequence",
        HostOp::Static(llama::llama_batch_add_sequence),
    ),
    (
        "flint::llama::batch_clear",
        HostOp::Static(llama::llama_batch_clear),
    ),
    (
        "flint::llama::batch_free",
        HostOp::Static(llama::llama_batch_free),
    ),
    (
        "flint::llama::sampler_chain_init",
        HostOp::Static(llama::llama_sampler_chain_init),
    ),
    (
        "flint::llama::sampler_add_top_k",
        HostOp::Static(llama::llama_sampler_add_top_k),
    ),
    (
        "flint::llama::sampler_add_top_p",
        HostOp::Static(llama::llama_sampler_add_top_p),
    ),
    (
        "flint::llama::sampler_add_min_p",
        HostOp::Static(llama::llama_sampler_add_min_p),
    ),
    (
        "flint::llama::sampler_add_temp",
        HostOp::Static(llama::llama_sampler_add_temp),
    ),
    (
        "flint::llama::sampler_add_dist",
        HostOp::Static(llama::llama_sampler_add_dist),
    ),
    (
        "flint::llama::sampler_add_greedy",
        HostOp::Static(llama::llama_sampler_add_greedy),
    ),
    (
        "flint::llama::sampler_chain_build",
        HostOp::Static(llama::llama_sampler_chain_build),
    ),
    (
        "flint::llama::sampler_sample",
        HostOp::Static(llama::llama_sampler_sample),
    ),
    (
        "flint::llama::sampler_accept",
        HostOp::Static(llama::llama_sampler_accept),
    ),
    (
        "flint::llama::sampler_free",
        HostOp::Static(llama::llama_sampler_free),
    ),
    (
        "flint::llama::decoder_init",
        HostOp::Static(llama::llama_decoder_init),
    ),
    (
        "flint::llama::decoder_push",
        HostOp::Static(llama::llama_decoder_push),
    ),
    (
        "flint::llama::decoder_free",
        HostOp::Static(llama::llama_decoder_free),
    ),
    (
        "flint::sd::ctx_params_init",
        HostOp::Static(sd::sd_ctx_params_init),
    ),
    (
        "flint::sd::ctx_params_set_paths",
        HostOp::Static(sd::sd_ctx_params_set_paths),
    ),
    (
        "flint::sd::ctx_params_set_backend",
        HostOp::Static(sd::sd_ctx_params_set_backend),
    ),
    (
        "flint::sd::ctx_params_set_wtype",
        HostOp::Static(sd::sd_ctx_params_set_wtype),
    ),
    (
        "flint::sd::ctx_params_set_vae_format",
        HostOp::Static(sd::sd_ctx_params_set_vae_format),
    ),
    (
        "flint::sd::ctx_params_set_flags",
        HostOp::Static(sd::sd_ctx_params_set_flags),
    ),
    ("flint::sd::new_sd_ctx", HostOp::Static(sd::sd_new_sd_ctx)),
    ("flint::sd::free_sd_ctx", HostOp::Static(sd::sd_free_sd_ctx)),
    (
        "flint::sd::img_gen_params_init",
        HostOp::Static(sd::sd_img_gen_params_init),
    ),
    (
        "flint::sd::img_gen_params_set_prompt",
        HostOp::Static(sd::sd_img_gen_params_set_prompt),
    ),
    (
        "flint::sd::img_gen_params_set_size",
        HostOp::Static(sd::sd_img_gen_params_set_size),
    ),
    (
        "flint::sd::img_gen_params_set_sample",
        HostOp::Static(sd::sd_img_gen_params_set_sample),
    ),
    (
        "flint::sd::img_gen_params_set_sampler",
        HostOp::Static(sd::sd_img_gen_params_set_sampler),
    ),
    (
        "flint::sd::str_to_sample_method",
        HostOp::Static(sd::sd_str_to_sample_method),
    ),
    (
        "flint::sd::str_to_scheduler",
        HostOp::Static(sd::sd_str_to_scheduler),
    ),
    (
        "flint::sd::sample_method_name",
        HostOp::Static(sd::sd_sample_method_name),
    ),
    (
        "flint::sd::scheduler_name",
        HostOp::Static(sd::sd_scheduler_name),
    ),
    (
        "flint::sd::get_default_sample_method",
        HostOp::Static(sd::sd_get_default_sample_method),
    ),
    (
        "flint::sd::get_default_scheduler",
        HostOp::Static(sd::sd_get_default_scheduler),
    ),
    (
        "flint::sd::generate_image",
        HostOp::Static(sd::sd_generate_image),
    ),
    ("flint::sd::images_save", HostOp::Static(sd::sd_images_save)),
    (
        "flint::sd::free_sd_images",
        HostOp::Static(sd::sd_free_sd_images),
    ),
    ("flint::tokenizer::load", HostOp::Context(tokenizer_load)),
    (
        "flint::tokenizer::encode_chat",
        HostOp::Context(tokenizer_encode_chat),
    ),
    (
        "flint::tokenizer::encode_vl_chat",
        HostOp::Context(tokenizer_encode_vl_chat),
    ),
    (
        "flint::tokenizer::encode_padded",
        HostOp::Context(tokenizer_encode_padded),
    ),
    (
        "flint::tokenizer::format_token_labels",
        HostOp::Context(tokenizer_format_token_labels),
    ),
    (
        "flint::tokenizer::decode_generated",
        HostOp::Context(tokenizer_decode_generated),
    ),
    (
        "flint::tokenizer::append_token",
        HostOp::Context(tokenizer_append_token),
    ),
    (
        "flint::tokenizer::append_token_tensor",
        HostOp::Context(tokenizer_append_token_tensor),
    ),
    (
        "flint::tokenizer::clear_generated_tokens",
        HostOp::Context(tokenizer_clear_generated_tokens),
    ),
    (
        "flint::tokenizer::push_generated_token_tensor",
        HostOp::Context(tokenizer_push_generated_token_tensor),
    ),
    (
        "flint::tokenizer::decode_generated_tokens",
        HostOp::Context(tokenizer_decode_generated_tokens),
    ),
    (
        "flint::tokenizer::single_token",
        HostOp::Context(tokenizer_single_token),
    ),
    (
        "flint::tokenizer::is_eos",
        HostOp::Context(tokenizer_is_eos),
    ),
    ("flint::weights::load", HostOp::Context(weights_load)),
    ("flint::weights::get", HostOp::Context(weights_get)),
    (
        "flint::weights::get_indexed",
        HostOp::Context(weights_get_indexed),
    ),
    ("flint::weights::get_or", HostOp::Context(weights_get_or)),
    (
        "flint::weights::get_optional",
        HostOp::Context(weights_get_optional),
    ),
    ("flint::pair::new", HostOp::Static(pair::pair_new)),
    ("flint::pair::local", HostOp::Static(pair::pair_local)),
    ("flint::pair::global", HostOp::Static(pair::pair_global)),
    ("flint::tensor::size", HostOp::Context(tensor_size)),
    ("flint::tensor::shape", HostOp::Context(tensor_shape)),
    (
        "flint::tensor::save_safetensors",
        HostOp::Context(tensor_save_safetensors),
    ),
    (
        "flint::tensor::load_safetensors",
        HostOp::Context(tensor_load_safetensors),
    ),
    ("flint::tensor::to_float", HostOp::Context(tensor_to_float)),
    (
        "flint::tensor::to_bfloat16",
        HostOp::Context(tensor_to_bfloat16),
    ),
    (
        "flint::tensor::ones_like",
        HostOp::Context(tensor_ones_like),
    ),
    (
        "flint::tensor::zeros_like",
        HostOp::Context(tensor_zeros_like),
    ),
    (
        "flint::tensor::zeros_like_int",
        HostOp::Context(tensor_zeros_like_int),
    ),
    ("flint::tensor::arange", HostOp::Context(tensor_arange)),
    (
        "flint::tensor::arange_start",
        HostOp::Context(tensor_arange_start),
    ),
    (
        "flint::tensor::causal_mask",
        HostOp::Context(tensor_causal_mask),
    ),
    (
        "flint::tensor::causal_padding_mask",
        HostOp::Context(tensor_causal_padding_mask),
    ),
    (
        "flint::tensor::padding_mask",
        HostOp::Context(tensor_padding_mask),
    ),
    ("flint::tensor::rope_cos", HostOp::Context(tensor_rope_cos)),
    ("flint::tensor::rope_sin", HostOp::Context(tensor_rope_sin)),
    (
        "flint::tensor::rope_cos_at",
        HostOp::Context(tensor_rope_cos_at),
    ),
    (
        "flint::tensor::rope_sin_at",
        HostOp::Context(tensor_rope_sin_at),
    ),
    ("flint::tensor::add", HostOp::Context(tensor_add)),
    ("flint::tensor::sub", HostOp::Context(tensor_sub)),
    ("flint::tensor::mul", HostOp::Context(tensor_mul)),
    (
        "flint::tensor::add_scalar",
        HostOp::Context(tensor_add_scalar),
    ),
    (
        "flint::tensor::mul_scalar",
        HostOp::Context(tensor_mul_scalar),
    ),
    (
        "flint::tensor::div_scalar",
        HostOp::Context(tensor_div_scalar),
    ),
    (
        "flint::tensor::pow_scalar",
        HostOp::Context(tensor_pow_scalar),
    ),
    ("flint::tensor::mean_dim", HostOp::Context(tensor_mean_dim)),
    ("flint::tensor::rsqrt", HostOp::Context(tensor_rsqrt)),
    ("flint::tensor::neg", HostOp::Context(tensor_neg)),
    ("flint::tensor::cos", HostOp::Context(tensor_cos)),
    ("flint::tensor::sin", HostOp::Context(tensor_sin)),
    ("flint::tensor::matmul", HostOp::Context(tensor_matmul)),
    ("flint::tensor::softmax", HostOp::Context(tensor_softmax)),
    (
        "flint::tensor::masked_fill",
        HostOp::Context(tensor_masked_fill),
    ),
    ("flint::tensor::cat2", HostOp::Context(tensor_cat2)),
    ("flint::tensor::stack2", HostOp::Context(tensor_stack2)),
    ("flint::tensor::chunk", HostOp::Context(tensor_chunk)),
    ("flint::tensor::narrow", HostOp::Context(tensor_narrow)),
    ("flint::tensor::tail", HostOp::Context(tensor_tail)),
    (
        "flint::tensor::transpose",
        HostOp::Context(tensor_transpose),
    ),
    (
        "flint::tensor::unsqueeze",
        HostOp::Context(tensor_unsqueeze),
    ),
    (
        "flint::tensor::repeat_interleave",
        HostOp::Context(tensor_repeat_interleave),
    ),
    (
        "flint::tensor::argmax_int",
        HostOp::Context(tensor_argmax_int),
    ),
    ("flint::tensor::argmax", HostOp::Context(tensor_argmax)),
    (
        "flint::tensor::argmax_token",
        HostOp::Context(tensor_argmax_token),
    ),
    (
        "flint::tensor::pad_reflect2d",
        HostOp::Context(tensor_pad_reflect2d),
    ),
    ("flint::tensor::relu", HostOp::Context(tensor_relu)),
    ("flint::tensor::sigmoid", HostOp::Context(tensor_sigmoid)),
    ("flint::tensor::silu", HostOp::Context(tensor_silu)),
    ("flint::tensor::gelu", HostOp::Context(tensor_gelu)),
    ("flint::tensor::swiglu", HostOp::Context(tensor_swiglu)),
    (
        "flint::tensor::contiguous",
        HostOp::Context(tensor_contiguous),
    ),
    ("flint::tensor::permute3", HostOp::Context(tensor_permute3)),
    ("flint::tensor::permute4", HostOp::Context(tensor_permute4)),
    ("flint::tensor::permute5", HostOp::Context(tensor_permute5)),
    ("flint::tensor::view2", HostOp::Context(tensor_view2)),
    ("flint::tensor::view3", HostOp::Context(tensor_view3)),
    ("flint::tensor::view4", HostOp::Context(tensor_view4)),
    ("flint::tensor::view5", HostOp::Context(tensor_view5)),
    ("flint::tensor::select", HostOp::Context(tensor_select)),
    ("flint::tensor::real", HostOp::Context(tensor_real)),
    ("flint::tensor::imag", HostOp::Context(tensor_imag)),
    ("flint::tensor::complex", HostOp::Context(tensor_complex)),
    (
        "flint::tensor::fft_rfftn2",
        HostOp::Context(tensor_fft_rfftn2),
    ),
    (
        "flint::tensor::fft_irfftn2",
        HostOp::Context(tensor_fft_irfftn2),
    ),
    (
        "flint::tensor::avg_pool2d_2",
        HostOp::Context(tensor_avg_pool2d_2),
    ),
    ("flint::nn::embedding", HostOp::Context(nn_embedding)),
    ("flint::nn::linear", HostOp::Context(nn_linear)),
    ("flint::nn::layer_norm", HostOp::Context(nn_layer_norm)),
    (
        "flint::nn::swiglu_linear",
        HostOp::Context(nn_swiglu_linear),
    ),
    ("flint::nn::rms_norm", HostOp::Context(nn_rms_norm)),
    ("flint::nn::add_rms_norm", HostOp::Context(nn_add_rms_norm)),
    ("flint::nn::apply_rope", HostOp::Context(nn_apply_rope)),
    (
        "flint::nn::apply_rope_pair",
        HostOp::Context(nn_apply_rope_pair),
    ),
    (
        "flint::nn::scaled_dot_product_attention",
        HostOp::Context(nn_scaled_dot_product_attention),
    ),
    (
        "flint::nn::scaled_dot_product_attention_masked",
        HostOp::Context(nn_scaled_dot_product_attention_masked),
    ),
    ("flint::nn::conv1d", HostOp::Context(nn_conv1d)),
    ("flint::nn::conv1d_step", HostOp::Context(nn_conv1d_step)),
    ("flint::nn::conv2d", HostOp::Context(nn_conv2d)),
    (
        "flint::nn::conv_transpose2d",
        HostOp::Context(nn_conv_transpose2d),
    ),
    ("flint::nn::batch_norm2d", HostOp::Context(nn_batch_norm2d)),
    (
        "flint::image::lfm2_vl_patches",
        HostOp::Context(image_lfm2_vl_patches),
    ),
    (
        "flint::vl::siglip2_position_embedding",
        HostOp::Context(vl_siglip2_position_embedding),
    ),
    (
        "flint::vl::pixel_unshuffle2",
        HostOp::Context(vl_pixel_unshuffle2),
    ),
    (
        "flint::vl::scatter_image_embeddings",
        HostOp::Context(vl_scatter_image_embeddings),
    ),
];

fn host_error(message: impl Into<String>) -> VmError {
    VmError::HostError(message.into())
}

fn arg<'a>(args: &'a [Value], index: usize, label: &str) -> VmResult<&'a Value> {
    args.get(index)
        .ok_or_else(|| host_error(format!("missing argument '{label}'")))
}

fn int_arg(args: &[Value], index: usize, label: &str) -> VmResult<i64> {
    match arg(args, index, label)? {
        Value::Int(value) => Ok(*value),
        other => Err(host_arg_type_error(label, "int", other)),
    }
}

fn bool_arg(args: &[Value], index: usize, label: &str) -> VmResult<bool> {
    match arg(args, index, label)? {
        Value::Bool(value) => Ok(*value),
        other => Err(host_arg_type_error(label, "bool", other)),
    }
}

fn float_arg(args: &[Value], index: usize, label: &str) -> VmResult<f64> {
    match arg(args, index, label)? {
        Value::Float(value) => Ok(*value),
        Value::Int(value) => Ok(*value as f64),
        other => Err(host_arg_type_error(label, "float", other)),
    }
}

fn string_arg<'a>(args: &'a [Value], index: usize, label: &str) -> VmResult<&'a str> {
    match arg(args, index, label)? {
        Value::String(value) => Ok(value.as_str()),
        other => Err(host_arg_type_error(label, "string", other)),
    }
}

fn host_arg_type_error(label: &str, expected: &str, value: &Value) -> VmError {
    let active = CURRENT_CONTEXT.with(|slot| {
        let pointer = slot.get();
        if pointer.is_null() {
            None
        } else {
            unsafe { (*pointer).active_host_op }
        }
    });
    match active {
        Some(op) => host_error(format!(
            "argument '{label}' in {op} must be {expected}, got {}",
            value_kind(value)
        )),
        None => host_error(format!(
            "argument '{label}' must be {expected}, got {}",
            value_kind(value)
        )),
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::Bytes(_) => "bytes",
        Value::Array(_) => "array",
        Value::Map(_) => "map",
    }
}

trait BorrowHostArg<'a>: Sized {
    fn borrow_host_arg(value: &'a Value, label: &str) -> VmResult<Self>;

    fn missing_host_arg(label: &str) -> VmResult<Self> {
        Err(host_error(format!("missing argument '{label}'")))
    }
}

fn borrow_arg<'a, T>(args: &'a [Value], index: usize, label: &str) -> VmResult<T>
where
    T: BorrowHostArg<'a>,
{
    match args.get(index) {
        Some(value) => T::borrow_host_arg(value, label),
        None => T::missing_host_arg(label),
    }
}

#[allow(dead_code)]
fn take_arg<'a, T>(args: &'a mut [Value], index: usize, label: &str) -> VmResult<T>
where
    T: BorrowHostArg<'a>,
{
    borrow_arg(args, index, label)
}

impl BorrowHostArg<'_> for i64 {
    fn borrow_host_arg(value: &Value, _label: &str) -> VmResult<Self> {
        match value {
            Value::Int(value) => Ok(*value),
            _ => Err(VmError::TypeMismatch("int")),
        }
    }
}

impl BorrowHostArg<'_> for bool {
    fn borrow_host_arg(value: &Value, _label: &str) -> VmResult<Self> {
        match value {
            Value::Bool(value) => Ok(*value),
            _ => Err(VmError::TypeMismatch("bool")),
        }
    }
}

impl BorrowHostArg<'_> for f64 {
    fn borrow_host_arg(value: &Value, _label: &str) -> VmResult<Self> {
        match value {
            Value::Float(value) => Ok(*value),
            Value::Int(value) => Ok(*value as f64),
            _ => Err(VmError::TypeMismatch("float")),
        }
    }
}

impl<'a> BorrowHostArg<'a> for &'a str {
    fn borrow_host_arg(value: &'a Value, _label: &str) -> VmResult<Self> {
        match value {
            Value::String(value) => Ok(value.as_str()),
            _ => Err(VmError::TypeMismatch("string")),
        }
    }
}

impl BorrowHostArg<'_> for String {
    fn borrow_host_arg(value: &Value, _label: &str) -> VmResult<Self> {
        match value {
            Value::String(value) => Ok(value.to_string()),
            _ => Err(VmError::TypeMismatch("string")),
        }
    }
}

impl BorrowHostArg<'_> for Value {
    fn borrow_host_arg(value: &Value, _label: &str) -> VmResult<Self> {
        Ok(value.clone())
    }
}

fn return_value(value: Value) -> VmResult<CallOutcome> {
    Ok(CallOutcome::Return(CallReturn::one(value)))
}

fn return_int(value: i64) -> VmResult<CallOutcome> {
    return_value(Value::Int(value))
}

fn return_tensor(context: &mut TorchContext, tensor: Tensor) -> VmResult<CallOutcome> {
    return_int(context.insert_tensor(tensor))
}

fn weights_load(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let path = PathBuf::from(string_arg(args, 0, "path")?);
    let target_kind = requested_weight_kind().map_err(host_error)?;
    if context.weights_path.as_deref() != Some(path.as_path())
        || context.weights_kind != target_kind
    {
        context.weight_handles.clear();
        context.conv1d_step_weights.clear();
        let weights = read_weight_safetensors(&path)?
            .into_iter()
            .map(|(name, tensor)| {
                (
                    name,
                    tensor_to_model_device(tensor, context.device, target_kind),
                )
            })
            .collect();
        context.weights = weights;
        context.weights_path = Some(path);
        context.weights_kind = target_kind;
    }
    let count = i64::try_from(context.weights.len())
        .map_err(|_| host_error("weight count exceeds RustScript integer range"))?;
    return_int(count)
}

fn read_weight_safetensors(path: &Path) -> VmResult<Vec<(String, Tensor)>> {
    if path.is_dir() {
        let mut files = std::fs::read_dir(path)
            .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))
            })
            .collect::<VmResult<Vec<_>>>()?;
        files.retain(|file| {
            file.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
        });
        files.sort();
        if files.is_empty() {
            return Err(host_error(format!(
                "{} contains no safetensors files",
                path.display()
            )));
        }
        let mut tensors = Vec::new();
        for file in files {
            tensors.extend(
                Tensor::read_safetensors(&file).map_err(|err| {
                    host_error(format!("failed to read {}: {err}", file.display()))
                })?,
            );
        }
        Ok(tensors)
    } else {
        Tensor::read_safetensors(path)
            .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))
    }
}

fn weights_get(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?;
    if let Some(handle) = cached_weight_handle(context, name) {
        return return_int(handle);
    } else {
        let tensor = get_or_build_weight(context, name)?;
        return_weight_tensor(context, name, tensor)
    }
}

fn weights_get_indexed(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let prefix = string_arg(args, 0, "prefix")?;
    let index = int_arg(args, 1, "index")?;
    let suffix = string_arg(args, 2, "suffix")?;
    let name = format!("{prefix}{index}{suffix}");
    if let Some(handle) = cached_weight_handle(context, &name) {
        return return_int(handle);
    }
    let tensor = get_or_build_weight(context, &name)?;
    return_weight_tensor(context, &name, tensor)
}

fn cached_weight_handle(context: &TorchContext, name: &str) -> Option<i64> {
    context
        .weight_handles
        .get(name)
        .copied()
        .filter(|handle| context.tensors.contains_key(handle))
}

fn return_weight_tensor(
    context: &mut TorchContext,
    name: &str,
    tensor: Tensor,
) -> VmResult<CallOutcome> {
    let handle = context.insert_tensor(tensor);
    context.weight_handles.insert(name.to_owned(), handle);
    return_int(handle)
}

fn weights_get_or(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?;
    let fallback = string_arg(args, 1, "fallback")?;
    if let Some(handle) = cached_weight_handle(context, name) {
        return return_int(handle);
    }
    let cache_name = if context.weights.contains_key(name) {
        name
    } else {
        fallback
    };
    if let Some(handle) = cached_weight_handle(context, cache_name) {
        if cache_name != name {
            context.weight_handles.insert(name.to_owned(), handle);
        }
        return return_int(handle);
    }
    let tensor = get_or_build_weight(context, name)
        .or_else(|_| context.weight(fallback).map(Tensor::shallow_clone))?;
    return_weight_tensor(context, name, tensor)
}

fn weights_get_optional(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?;
    if let Some(handle) = cached_weight_handle(context, name) {
        return return_int(handle);
    }
    match get_or_build_weight(context, name) {
        Ok(tensor) => return_weight_tensor(context, name, tensor),
        Err(_) => {
            let handle = context.next_missing_weight;
            context.next_missing_weight -= 1;
            context
                .missing_weight_handles
                .insert(handle, name.to_owned());
            return_int(handle)
        }
    }
}

fn get_or_build_weight(context: &mut TorchContext, name: &str) -> VmResult<Tensor> {
    if let Some(tensor) = context.weights.get(name) {
        return Ok(tensor.shallow_clone());
    }
    let tensor = if let Some(prefix) = name.strip_suffix(".self_attn.qkv_proj.weight") {
        let q = context
            .weight(&format!("{prefix}.self_attn.q_proj.weight"))?
            .shallow_clone();
        let k = context
            .weight(&format!("{prefix}.self_attn.k_proj.weight"))?
            .shallow_clone();
        let v = context
            .weight(&format!("{prefix}.self_attn.v_proj.weight"))?
            .shallow_clone();
        Tensor::cat(&[&q, &k, &v], 0)
    } else if let Some(prefix) = name.strip_suffix(".feed_forward.w1_w3.weight") {
        let w1 = context
            .weight(&format!("{prefix}.feed_forward.w1.weight"))?
            .shallow_clone();
        let w3 = context
            .weight(&format!("{prefix}.feed_forward.w3.weight"))?
            .shallow_clone();
        Tensor::cat(&[&w1, &w3], 0)
    } else {
        return Err(host_error(format!("missing weight '{name}'")));
    };
    context
        .weights
        .insert(name.to_owned(), tensor.shallow_clone());
    Ok(tensor)
}

fn tokenizer_load(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let path = PathBuf::from(string_arg(args, 0, "path")?);
    if context.tokenizer_path.as_deref() != Some(path.as_path()) {
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))?;
        context.tokenizer = Some(tokenizer);
        context.tokenizer_path = Some(path);
    }
    return_value(Value::Bool(true))
}

fn tokenizer_encode_chat(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let system = string_arg(args, 0, "system")?;
    let user = string_arg(args, 1, "user")?;
    let prompt = format!(
        "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let tokenizer = context
        .tokenizer
        .as_ref()
        .ok_or_else(|| host_error("tokenizer has not been loaded"))?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|err| host_error(format!("tokenizer encode failed: {err}")))?;
    let ids = encoding
        .get_ids()
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    let len = i64::try_from(ids.len()).map_err(|_| host_error("token count out of range"))?;
    let tensor = Tensor::from_slice(&ids)
        .view([1, len])
        .to_device(context.device);
    return_tensor(context, tensor)
}

fn tokenizer_encode_vl_chat(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let system = string_arg(args, 0, "system")?;
    let user = string_arg(args, 1, "user")?;
    let image_tokens = usize::try_from(int_arg(args, 2, "image_tokens")?)
        .map_err(|_| host_error("image_tokens must be non-negative"))?;
    let image_tokens = "<image>".repeat(image_tokens);
    let prompt = format!(
        "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|image_start|>{image_tokens}<|image_end|>{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let tokenizer = context
        .tokenizer
        .as_ref()
        .ok_or_else(|| host_error("tokenizer has not been loaded"))?;
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|err| host_error(format!("tokenizer encode failed: {err}")))?;
    let ids = encoding
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    let tensor = Tensor::from_slice(&ids)
        .view([1, ids.len() as i64])
        .to_device(context.device);
    return_tensor(context, tensor)
}

fn tokenizer_encode_padded(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let text = string_arg(args, 0, "text")?;
    let max_len = usize::try_from(int_arg(args, 1, "max_len")?)
        .map_err(|_| host_error("max_len must be non-negative"))?;
    let pad_token = int_arg(args, 2, "pad_token")?;
    let add_special_tokens = bool_arg(args, 3, "add_special_tokens")?;
    let tokenizer = context
        .tokenizer
        .as_ref()
        .ok_or_else(|| host_error("tokenizer has not been loaded"))?;
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|err| host_error(format!("tokenizer encode failed: {err}")))?;
    let mut ids = encoding
        .get_ids()
        .iter()
        .map(|value| i64::from(*value))
        .collect::<Vec<_>>();
    ids.truncate(max_len);
    let mut mask = vec![1i64; ids.len()];
    if ids.len() < max_len {
        let missing = max_len - ids.len();
        ids.extend(std::iter::repeat_n(pad_token, missing));
        mask.extend(std::iter::repeat_n(0i64, missing));
    }
    let max_len = i64::try_from(max_len).map_err(|_| host_error("max_len out of range"))?;
    let ids_tensor = Tensor::from_slice(&ids)
        .view([1, max_len])
        .to_device(context.device);
    let mask_tensor = Tensor::from_slice(&mask)
        .view([1, max_len])
        .to_device(context.device);
    let ids_handle = context.insert_tensor(ids_tensor);
    let mask_handle = context.insert_tensor(mask_tensor);
    return_int(context.insert_pair(FfcPair {
        local: ids_handle,
        global: mask_handle,
    }))
}

fn tokenizer_format_token_labels(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let text = string_arg(args, 0, "text")?;
    let labels = context
        .tensor(int_arg(args, 1, "labels")?)?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    let label_names = string_arg(args, 2, "label_names")?
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    let add_special_tokens = bool_arg(args, 3, "add_special_tokens")?;
    let tokenizer = context
        .tokenizer
        .as_ref()
        .ok_or_else(|| host_error("tokenizer has not been loaded"))?;
    let encoding = tokenizer
        .encode(text, add_special_tokens)
        .map_err(|err| host_error(format!("tokenizer encode failed: {err}")))?;
    let label_ids = Vec::<i64>::try_from(&labels.view([-1]))
        .map_err(|err| host_error(format!("failed to copy labels: {err}")))?;

    let mut spans: Vec<(String, usize, usize)> = Vec::new();
    for (index, (start, end)) in encoding.get_offsets().iter().copied().enumerate() {
        let Some(label_id) = label_ids.get(index).copied() else {
            break;
        };
        if start == end || label_id == 0 {
            continue;
        }
        let label_index =
            usize::try_from(label_id).map_err(|_| host_error("label id must be non-negative"))?;
        let label = label_names
            .get(label_index)
            .ok_or_else(|| host_error(format!("label id {label_id} is out of range")))?
            .to_string();
        if let Some((last_label, _last_start, last_end)) = spans.last_mut()
            && *last_label == label
            && start <= *last_end
        {
            *last_end = (*last_end).max(end);
            continue;
        }
        spans.push((label, start, end));
    }

    let mut output = String::new();
    for (label, start, end) in spans {
        let surface = text.get(start..end).unwrap_or("");
        output.push_str(&format!("{label}\t{start}\t{end}\t{surface}\n"));
    }
    if output.is_empty() {
        output.push_str("O\n");
    }
    return_value(Value::String(output.into()))
}

fn tokenizer_decode_generated(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let tokens = context
        .tensor(int_arg(args, 0, "tokens")?)?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64)
        .view([-1]);
    let prompt_len = usize::try_from(int_arg(args, 1, "prompt_len")?)
        .map_err(|_| host_error("prompt length must be non-negative"))?;
    let ids = Vec::<i64>::try_from(&tokens)
        .map_err(|err| host_error(format!("failed to copy token ids: {err}")))?;
    let generated = ids
        .into_iter()
        .skip(prompt_len)
        .filter_map(|value| u32::try_from(value).ok())
        .collect::<Vec<_>>();
    let tokenizer = context
        .tokenizer
        .as_ref()
        .ok_or_else(|| host_error("tokenizer has not been loaded"))?;
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|err| host_error(format!("tokenizer decode failed: {err}")))?;
    return_value(Value::String(text.into()))
}

fn tokenizer_append_token(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let tokens = context.tensor(int_arg(args, 0, "tokens")?)?;
    let token = int_arg(args, 1, "token")?;
    let next = Tensor::from_slice(&[token])
        .view([1, 1])
        .to_device(context.device);
    let output = Tensor::cat(&[tokens, &next], 1);
    return_tensor(context, output)
}

fn tokenizer_append_token_tensor(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let tokens = context.tensor(int_arg(args, 0, "tokens")?)?;
    let next = context.tensor(int_arg(args, 1, "token")?)?;
    let output = Tensor::cat(&[tokens, next], 1);
    return_tensor(context, output)
}

fn tokenizer_clear_generated_tokens(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    if !args.is_empty() {
        return Err(host_error("clear_generated_tokens takes no arguments"));
    }
    context.generated_token_tensors.clear();
    return_value(Value::Bool(true))
}

fn tokenizer_push_generated_token_tensor(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let next = context.tensor(int_arg(args, 0, "token")?)?.shallow_clone();
    context.generated_token_tensors.push(next);
    return_value(Value::Bool(true))
}

fn tokenizer_decode_generated_tokens(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    if !args.is_empty() {
        return Err(host_error("decode_generated_tokens takes no arguments"));
    }
    let generated = if context.generated_token_tensors.is_empty() {
        Vec::new()
    } else {
        let tokens = context.generated_token_tensors.iter().collect::<Vec<_>>();
        let ids_tensor = Tensor::cat(&tokens, 1)
            .to_device(Device::Cpu)
            .to_kind(Kind::Int64)
            .view([-1]);
        let ids = Vec::<i64>::try_from(&ids_tensor)
            .map_err(|err| host_error(format!("failed to copy token ids: {err}")))?;
        ids.into_iter()
            .filter_map(|value| u32::try_from(value).ok())
            .collect::<Vec<_>>()
    };
    let tokenizer = context
        .tokenizer
        .as_ref()
        .ok_or_else(|| host_error("tokenizer has not been loaded"))?;
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|err| host_error(format!("tokenizer decode failed: {err}")))?;
    return_value(Value::String(text.into()))
}

fn tokenizer_single_token(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let token = int_arg(args, 0, "token")?;
    let output = Tensor::from_slice(&[token])
        .view([1, 1])
        .to_device(context.device);
    return_tensor(context, output)
}

fn tokenizer_is_eos(_context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let token = int_arg(args, 0, "token")?;
    return_value(Value::Bool(token == 7))
}

fn tensor_size(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let tensor = context.tensor(int_arg(args, 0, "tensor")?)?;
    let raw_dim = int_arg(args, 1, "dim")?;
    let rank = i64::try_from(tensor.size().len()).map_err(|_| host_error("rank out of range"))?;
    let dim = if raw_dim < 0 { rank + raw_dim } else { raw_dim };
    let dim = usize::try_from(dim).map_err(|_| host_error("dimension is out of range"))?;
    let value = tensor
        .size()
        .get(dim)
        .copied()
        .ok_or_else(|| host_error(format!("dimension {dim} is out of range")))?;
    return_int(value)
}

fn tensor_shape(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let tensor = context.tensor(int_arg(args, 0, "tensor")?)?;
    return_value(Value::String(format!("{:?}", tensor.size()).into()))
}

fn tensor_save_safetensors(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let tensor = context
        .tensor(int_arg(args, 0, "tensor")?)?
        .to_device(Device::Cpu)
        .contiguous();
    let path = PathBuf::from(string_arg(args, 1, "path")?);
    let name = string_arg(args, 2, "name")?;
    ensure_parent_dir(&path)?;
    Tensor::write_safetensors(&[(name, &tensor)], &path)
        .map_err(|err| host_error(format!("failed to write {}: {err}", path.display())))?;
    return_value(Value::Bool(true))
}

fn tensor_load_safetensors(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let path = PathBuf::from(string_arg(args, 0, "path")?);
    let name = string_arg(args, 1, "name")?;
    let tensors = Tensor::read_safetensors(&path)
        .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))?;
    let tensor = tensors
        .into_iter()
        .find_map(|(tensor_name, tensor)| (tensor_name == name).then_some(tensor))
        .ok_or_else(|| host_error(format!("missing tensor '{name}' in {}", path.display())))?
        .to_device(context.device);
    return_tensor(context, tensor)
}

fn tensor_to_float(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, |input| input.to_kind(Kind::Float))
}

fn tensor_to_bfloat16(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, |input| input.to_kind(Kind::BFloat16))
}

fn unary_tensor(
    context: &mut TorchContext,
    args: &[Value],
    op: impl FnOnce(&Tensor) -> Tensor,
) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let output = op(input);
    return_tensor(context, output)
}

fn binary_tensor(
    context: &mut TorchContext,
    args: &[Value],
    op: impl FnOnce(&Tensor, &Tensor) -> Tensor,
) -> VmResult<CallOutcome> {
    let left = context.tensor(int_arg(args, 0, "left")?)?;
    let right = context.tensor(int_arg(args, 1, "right")?)?;
    let output = op(left, right);
    return_tensor(context, output)
}

fn tensor_ones_like(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::ones_like)
}

fn tensor_zeros_like(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::zeros_like)
}

fn tensor_zeros_like_int(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let output = Tensor::zeros(input.size(), (Kind::Int64, context.device));
    return_tensor(context, output)
}

fn tensor_arange(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let end = int_arg(args, 0, "end")?;
    let output = Tensor::arange(end, (Kind::Int64, context.device));
    return_tensor(context, output)
}

fn tensor_arange_start(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let start = int_arg(args, 0, "start")?;
    let end = int_arg(args, 1, "end")?;
    let output = Tensor::arange_start(start, end, (Kind::Int64, context.device));
    return_tensor(context, output)
}

fn tensor_causal_mask(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let seq_len = int_arg(args, 0, "seq_len")?;
    let allowed = Tensor::ones([seq_len, seq_len], (Kind::Float, context.device)).tril(0);
    let blocked = allowed.lt(0.5);
    let zeros = Tensor::zeros([seq_len, seq_len], (Kind::Float, context.device));
    let output = zeros
        .masked_fill(&blocked, f64::NEG_INFINITY)
        .view([1, 1, seq_len, seq_len]);
    return_tensor(context, output)
}

fn tensor_causal_padding_mask(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let attention_mask = context
        .tensor(int_arg(args, 0, "attention_mask")?)?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    let size = attention_mask.size();
    if size.len() != 2 {
        return Err(host_error("attention_mask must have rank 2"));
    }
    let batch = usize::try_from(size[0]).map_err(|_| host_error("batch size out of range"))?;
    let seq_len = usize::try_from(size[1]).map_err(|_| host_error("seq_len out of range"))?;
    let mask_values = Vec::<i64>::try_from(&attention_mask.view([-1]))
        .map_err(|err| host_error(format!("failed to copy attention mask: {err}")))?;
    let masked = f32::NEG_INFINITY;
    let mut values = Vec::with_capacity(batch * seq_len * seq_len);
    for b in 0..batch {
        for i in 0..seq_len {
            for j in 0..seq_len {
                let key_valid = mask_values[b * seq_len + j] != 0;
                let causal = j <= i;
                values.push(if key_valid && causal { 0.0 } else { masked });
            }
        }
    }
    let output = Tensor::from_slice(&values)
        .view([batch as i64, 1, seq_len as i64, seq_len as i64])
        .to_device(context.device);
    return_tensor(context, output)
}

fn tensor_padding_mask(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let attention_mask = context
        .tensor(int_arg(args, 0, "attention_mask")?)?
        .to_device(Device::Cpu)
        .to_kind(Kind::Int64);
    let size = attention_mask.size();
    if size.len() != 2 {
        return Err(host_error("attention_mask must have rank 2"));
    }
    let batch = usize::try_from(size[0]).map_err(|_| host_error("batch size out of range"))?;
    let seq_len = usize::try_from(size[1]).map_err(|_| host_error("seq_len out of range"))?;
    let mask_values = Vec::<i64>::try_from(&attention_mask.view([-1]))
        .map_err(|err| host_error(format!("failed to copy attention mask: {err}")))?;
    let mut values = Vec::with_capacity(batch * seq_len);
    for value in mask_values {
        values.push(if value != 0 { 0.0 } else { f32::NEG_INFINITY });
    }
    let output = Tensor::from_slice(&values)
        .view([batch as i64, 1, 1, seq_len as i64])
        .to_device(context.device);
    return_tensor(context, output)
}

fn tensor_rope_cos(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    rope_table(context, args, 0, RopeKind::Cos)
}

fn tensor_rope_sin(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    rope_table(context, args, 0, RopeKind::Sin)
}

fn tensor_rope_cos_at(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let start = int_arg(args, 3, "start")?;
    rope_table(context, args, start, RopeKind::Cos)
}

fn tensor_rope_sin_at(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let start = int_arg(args, 3, "start")?;
    rope_table(context, args, start, RopeKind::Sin)
}

fn rope_table(
    context: &mut TorchContext,
    args: &[Value],
    start: i64,
    kind: RopeKind,
) -> VmResult<CallOutcome> {
    let seq_len = usize::try_from(int_arg(args, 0, "seq_len")?)
        .map_err(|_| host_error("seq_len must be non-negative"))?;
    let head_dim = usize::try_from(int_arg(args, 1, "head_dim")?)
        .map_err(|_| host_error("head_dim must be non-negative"))?;
    let theta = float_arg(args, 2, "theta")? as f32;
    if start < 0 {
        return Err(host_error("start must be non-negative"));
    }
    let end = start
        .checked_add(seq_len as i64)
        .ok_or_else(|| host_error("rope range overflow"))?;
    let key = RopeCacheKey {
        kind,
        head_dim,
        theta_bits: theta.to_bits(),
    };
    let current_len = context
        .rope_cache
        .get(&key)
        .map(|tensor| tensor.size()[2])
        .unwrap_or(0);
    if current_len < end {
        let target_len = end.max(current_len.saturating_mul(2)).max(128);
        let table = build_rope_table(target_len, head_dim, theta, context.device, kind);
        context.rope_cache.insert(key, table);
    }
    let output = context
        .rope_cache
        .get(&key)
        .expect("rope cache should contain key after build")
        .narrow(2, start, seq_len as i64);
    return_tensor(context, output)
}

fn build_rope_table(
    len: i64,
    head_dim: usize,
    theta: f32,
    device: Device,
    kind: RopeKind,
) -> Tensor {
    let half = head_dim / 2;
    let mut data = Vec::with_capacity(len as usize * head_dim);
    for pos in 0..len as usize {
        let mut row = vec![0.0f32; head_dim];
        for idx in 0..half {
            let exponent = (idx * 2) as f32 / head_dim as f32;
            let angle = pos as f32 / theta.powf(exponent);
            let value = match kind {
                RopeKind::Cos => angle.cos(),
                RopeKind::Sin => angle.sin(),
            };
            row[idx] = value;
            row[idx + half] = value;
        }
        data.extend(row);
    }
    Tensor::from_slice(&data)
        .view([1, 1, len, head_dim as i64])
        .to_device(device)
        .to_kind(
            requested_weight_kind()
                .ok()
                .flatten()
                .unwrap_or(Kind::BFloat16),
        )
}

fn tensor_to_model_device(tensor: Tensor, device: Device, target_kind: Option<Kind>) -> Tensor {
    let tensor = tensor.to_device(device);
    match target_kind {
        Some(kind) if is_float_kind(tensor.kind()) => tensor.to_kind(kind),
        _ => tensor,
    }
}

fn requested_weight_kind() -> std::result::Result<Option<Kind>, String> {
    let Some(value) = std::env::var_os("KOHARU_TORCH_WEIGHT_KIND") else {
        return Ok(Some(Kind::Float));
    };
    let value = value.to_string_lossy().to_ascii_lowercase();
    match value.as_str() {
        "" | "native" | "auto" => Ok(None),
        "half" | "fp16" | "f16" => Ok(Some(Kind::Half)),
        "bf16" | "bfloat16" => Ok(Some(Kind::BFloat16)),
        "float" | "fp32" | "f32" => Ok(Some(Kind::Float)),
        other => Err(format!(
            "KOHARU_TORCH_WEIGHT_KIND must be native, half, bf16, or float, got '{other}'"
        )),
    }
}

fn is_float_kind(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Half
            | Kind::Float
            | Kind::Double
            | Kind::BFloat16
            | Kind::Float8e5m2
            | Kind::Float8e4m3fn
            | Kind::Float8e5m2fnuz
            | Kind::Float8e4m3fnuz
    )
}

fn ensure_parent_dir(path: &Path) -> VmResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|err| host_error(format!("failed to create {}: {err}", parent.display())))?;
    }
    Ok(())
}

fn tensor_add(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, |left, right| left + right)
}

fn tensor_sub(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, |left, right| left - right)
}

fn tensor_mul(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, |left, right| left * right)
}

fn tensor_add_scalar(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let value = float_arg(args, 1, "value")?;
    return_tensor(context, input + value)
}

fn tensor_mul_scalar(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let value = float_arg(args, 1, "value")?;
    return_tensor(context, input * value)
}

fn tensor_div_scalar(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let value = float_arg(args, 1, "value")?;
    return_tensor(context, input / value)
}

fn tensor_pow_scalar(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let value = float_arg(args, 1, "value")?;
    return_tensor(context, input.pow_tensor_scalar(value))
}

fn tensor_mean_dim(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    let keepdim = bool_arg(args, 2, "keepdim")?;
    let output = input.mean_dim(&[dim][..], keepdim, Kind::Float);
    return_tensor(context, output)
}

fn tensor_rsqrt(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::rsqrt)
}

fn tensor_neg(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::neg)
}

fn tensor_cos(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::cos)
}

fn tensor_sin(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::sin)
}

fn tensor_matmul(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, Tensor::matmul)
}

fn tensor_softmax(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    let output = input.softmax(dim, Kind::Float);
    return_tensor(context, output)
}

fn tensor_masked_fill(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let mask = context.tensor(int_arg(args, 1, "mask")?)?;
    let value = float_arg(args, 2, "value")?;
    return_tensor(context, input.masked_fill(mask, value))
}

fn tensor_cat2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let left = context.tensor(int_arg(args, 0, "left")?)?;
    let right = context.tensor(int_arg(args, 1, "right")?)?;
    let dim = int_arg(args, 2, "dim")?;
    let output = Tensor::cat(&[left, right], dim);
    return_tensor(context, output)
}

fn tensor_stack2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let left = context.tensor(int_arg(args, 0, "left")?)?;
    let right = context.tensor(int_arg(args, 1, "right")?)?;
    let dim = int_arg(args, 2, "dim")?;
    let output = Tensor::stack(&[left, right], dim);
    return_tensor(context, output)
}

fn tensor_chunk(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let chunks = int_arg(args, 1, "chunks")?;
    if chunks <= 0 {
        return Err(host_error("chunks must be positive"));
    }
    let raw_dim = int_arg(args, 2, "dim")?;
    let rank = i64::try_from(input.size().len()).map_err(|_| host_error("rank out of range"))?;
    let dim = if raw_dim < 0 { rank + raw_dim } else { raw_dim };
    let dim_index = usize::try_from(dim).map_err(|_| host_error("dimension is out of range"))?;
    let index = usize::try_from(int_arg(args, 3, "index")?)
        .map_err(|_| host_error("chunk index must be non-negative"))?;
    let dim_size = *input
        .size()
        .get(dim_index)
        .ok_or_else(|| host_error(format!("dimension {dim} is out of range")))?;
    let chunk_size = (dim_size + chunks - 1) / chunks;
    let start =
        i64::try_from(index).map_err(|_| host_error("chunk index out of range"))? * chunk_size;
    if start >= dim_size {
        return Err(host_error(format!("chunk index {index} is out of range")));
    }
    let len = chunk_size.min(dim_size - start);
    let output = input.narrow(dim, start, len);
    return_tensor(context, output)
}

fn tensor_narrow(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    let start = int_arg(args, 2, "start")?;
    let len = int_arg(args, 3, "len")?;
    return_tensor(context, input.narrow(dim, start, len))
}

fn tensor_tail(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let raw_dim = int_arg(args, 1, "dim")?;
    let len = int_arg(args, 2, "len")?;
    if len < 0 {
        return Err(host_error("tail length must be non-negative"));
    }
    let rank = i64::try_from(input.size().len()).map_err(|_| host_error("rank out of range"))?;
    let dim = if raw_dim < 0 { rank + raw_dim } else { raw_dim };
    let dim_index = usize::try_from(dim).map_err(|_| host_error("dimension is out of range"))?;
    let dim_size = *input
        .size()
        .get(dim_index)
        .ok_or_else(|| host_error(format!("dimension {dim} is out of range")))?;
    let actual_len = len.min(dim_size);
    let output = input.narrow(dim, dim_size - actual_len, actual_len);
    return_tensor(context, output)
}

fn tensor_transpose(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim0 = int_arg(args, 1, "dim0")?;
    let dim1 = int_arg(args, 2, "dim1")?;
    return_tensor(context, input.transpose(dim0, dim1))
}

fn tensor_unsqueeze(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    return_tensor(context, input.unsqueeze(dim))
}

fn tensor_repeat_interleave(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let repeats = int_arg(args, 1, "repeats")?;
    let dim = int_arg(args, 2, "dim")?;
    let output = input.repeat_interleave_self_int(repeats, dim, None);
    return_tensor(context, output)
}

fn tensor_argmax_int(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    let output = input.argmax(dim, false).to_device(Device::Cpu).view([-1]);
    return_int(output.int64_value(&[0]))
}

fn tensor_argmax(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    return_tensor(context, input.argmax(dim, false))
}

fn tensor_argmax_token(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    let output = input.argmax(dim, false).view([1, 1]);
    return_tensor(context, output)
}

fn tensor_pad_reflect2d(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let left = int_arg(args, 1, "left")?;
    let right = int_arg(args, 2, "right")?;
    let top = int_arg(args, 3, "top")?;
    let bottom = int_arg(args, 4, "bottom")?;
    let output = input.reflection_pad2d([left, right, top, bottom]);
    return_tensor(context, output)
}

fn tensor_relu(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::relu)
}

fn tensor_sigmoid(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::sigmoid)
}

fn tensor_silu(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::silu)
}

fn tensor_gelu(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let approximate = string_arg(args, 1, "approximate")?;
    return_tensor(context, input.gelu(approximate))
}

fn tensor_swiglu(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let Some(&last_dim) = input.size().last() else {
        return Err(host_error("swiglu input must have at least one dimension"));
    };
    if last_dim % 2 != 0 {
        return Err(host_error("swiglu input last dimension must be even"));
    }
    let intermediate = last_dim / 2;
    let gate = input.narrow(-1, 0, intermediate).silu();
    let up = input.narrow(-1, intermediate, intermediate);
    return_tensor(context, gate * up)
}

fn tensor_contiguous(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::contiguous)
}

fn tensor_permute3(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dims = [
        int_arg(args, 1, "d0")?,
        int_arg(args, 2, "d1")?,
        int_arg(args, 3, "d2")?,
    ];
    let output = input.permute(dims);
    return_tensor(context, output)
}

fn tensor_permute4(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dims = [
        int_arg(args, 1, "d0")?,
        int_arg(args, 2, "d1")?,
        int_arg(args, 3, "d2")?,
        int_arg(args, 4, "d3")?,
    ];
    let output = input.permute(dims);
    return_tensor(context, output)
}

fn tensor_permute5(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dims = [
        int_arg(args, 1, "d0")?,
        int_arg(args, 2, "d1")?,
        int_arg(args, 3, "d2")?,
        int_arg(args, 4, "d3")?,
        int_arg(args, 5, "d4")?,
    ];
    let output = input.permute(dims);
    return_tensor(context, output)
}

fn tensor_view4(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let shape = [
        int_arg(args, 1, "d0")?,
        int_arg(args, 2, "d1")?,
        int_arg(args, 3, "d2")?,
        int_arg(args, 4, "d3")?,
    ];
    let output = input.view(shape);
    return_tensor(context, output)
}

fn tensor_view2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let shape = [int_arg(args, 1, "d0")?, int_arg(args, 2, "d1")?];
    let output = input.view(shape);
    return_tensor(context, output)
}

fn tensor_view3(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let shape = [
        int_arg(args, 1, "d0")?,
        int_arg(args, 2, "d1")?,
        int_arg(args, 3, "d2")?,
    ];
    let output = input.view(shape);
    return_tensor(context, output)
}

fn tensor_view5(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let shape = [
        int_arg(args, 1, "d0")?,
        int_arg(args, 2, "d1")?,
        int_arg(args, 3, "d2")?,
        int_arg(args, 4, "d3")?,
        int_arg(args, 5, "d4")?,
    ];
    let output = input.view(shape);
    return_tensor(context, output)
}

fn tensor_select(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let dim = int_arg(args, 1, "dim")?;
    let index = int_arg(args, 2, "index")?;
    let output = input.select(dim, index);
    return_tensor(context, output)
}

fn tensor_real(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::real)
}

fn tensor_imag(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::imag)
}

fn tensor_complex(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let real = context.tensor(int_arg(args, 0, "real")?)?;
    let imag = context.tensor(int_arg(args, 1, "imag")?)?;
    let output = Tensor::complex(real, imag);
    return_tensor(context, output)
}

fn tensor_fft_rfftn2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let output = input.fft_rfftn(None::<&[i64]>, &[-2, -1][..], "ortho");
    return_tensor(context, output)
}

fn tensor_fft_irfftn2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let height = int_arg(args, 1, "height")?;
    let width = int_arg(args, 2, "width")?;
    let output = input.fft_irfftn(&[height, width][..], &[-2, -1][..], "ortho");
    return_tensor(context, output)
}

fn tensor_avg_pool2d_2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?;
    let output = input.avg_pool2d([2, 2], [2, 2], [0, 0], false, true, None);
    return_tensor(context, output)
}

fn nn_embedding(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let indices = context.tensor(int_arg(args, 0, "indices")?)?;
    let weight = context.tensor(int_arg(args, 1, "weight")?)?;
    let output = Tensor::embedding(weight, indices, -1, false, false);
    return_tensor(context, output)
}

fn nn_linear(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input_handle = int_arg(args, 0, "tensor")?;
    let weight_handle = int_arg(args, 1, "weight")?;
    if input_handle == 0 {
        return Err(host_error("linear input handle is 0"));
    }
    if weight_handle == 0 {
        return Err(host_error("linear weight handle is 0"));
    }
    let input = context.tensor(input_handle)?.shallow_clone();
    let weight = context.tensor(weight_handle)?.shallow_clone();
    let bias_handle = int_arg(args, 2, "bias")?;
    let bias = if bias_handle == 0 {
        None
    } else {
        Some(context.tensor(bias_handle)?.shallow_clone())
    };
    let output = match bias {
        Some(bias) => input.linear(&weight, Some(&bias)),
        None if linear_mv_enabled() => linear_or_mv(&input, &weight),
        None => input.linear(&weight, None::<&Tensor>),
    };
    return_tensor(context, output)
}

fn nn_layer_norm(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let weight_handle = int_arg(args, 1, "weight")?;
    let bias_handle = int_arg(args, 2, "bias")?;
    let eps = float_arg(args, 3, "eps")?;
    let normalized_size = int_arg(args, 4, "normalized_size")?;
    let weight = if weight_handle == 0 {
        None
    } else {
        Some(context.tensor(weight_handle)?.shallow_clone())
    };
    let bias = if bias_handle == 0 {
        None
    } else {
        Some(context.tensor(bias_handle)?.shallow_clone())
    };
    let output = match (weight.as_ref(), bias.as_ref()) {
        (Some(weight), Some(bias)) => {
            input.layer_norm([normalized_size], Some(weight), Some(bias), eps, false)
        }
        (Some(weight), None) => input.layer_norm([normalized_size], Some(weight), None, eps, false),
        (None, Some(bias)) => input.layer_norm([normalized_size], None, Some(bias), eps, false),
        (None, None) => input.layer_norm(
            [normalized_size],
            None::<&Tensor>,
            None::<&Tensor>,
            eps,
            false,
        ),
    };
    return_tensor(context, output)
}

fn linear_or_mv(input: &Tensor, weight: &Tensor) -> Tensor {
    let input_size = input.size();
    let weight_size = weight.size();
    let Some(&out_features) = weight_size.first() else {
        return input.linear(weight, None::<&Tensor>);
    };
    if input_size.len() == 3 && input_size[0] == 1 && input_size[1] == 1 {
        return weight
            .mv(&input.view([input_size[2]]))
            .view([1, 1, out_features]);
    }
    if input_size.len() == 2 && input_size[0] == 1 {
        return weight
            .mv(&input.view([input_size[1]]))
            .view([1, out_features]);
    }
    input.linear(weight, None::<&Tensor>)
}

fn linear_mv_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("KOHARU_TORCH_LINEAR_MV")
            .is_none_or(|value| value != "0" && !value.is_empty())
    })
}

fn nn_swiglu_linear(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let weight = context.tensor(int_arg(args, 1, "weight")?)?.shallow_clone();
    let bias_handle = int_arg(args, 2, "bias")?;
    let Some(&last_dim) = input.size().last() else {
        return Err(host_error(
            "swiglu_linear input must have at least one dimension",
        ));
    };
    if last_dim % 2 != 0 {
        return Err(host_error(
            "swiglu_linear input last dimension must be even",
        ));
    }
    let intermediate = last_dim / 2;
    let gate = input.narrow(-1, 0, intermediate).silu();
    let up = input.narrow(-1, intermediate, intermediate);
    let activated = gate * up;
    let output = if bias_handle == 0 {
        linear_or_mv(&activated, &weight)
    } else {
        let bias = context.tensor(bias_handle)?.shallow_clone();
        activated.linear(&weight, Some(&bias))
    };
    return_tensor(context, output)
}

fn nn_rms_norm(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let weight = context.tensor(int_arg(args, 1, "weight")?)?.shallow_clone();
    let eps = float_arg(args, 2, "eps")?;
    let normalized_shape = weight.size();
    let output = input
        .internal_fused_rms_norm(normalized_shape, Some(&weight), eps)
        .0;
    return_tensor(context, output)
}

fn nn_add_rms_norm(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "input")?)?.shallow_clone();
    let residual = context
        .tensor(int_arg(args, 1, "residual")?)?
        .shallow_clone();
    let weight = context.tensor(int_arg(args, 2, "weight")?)?.shallow_clone();
    let eps = float_arg(args, 3, "eps")?;
    let hidden = input + residual;
    let normalized_shape = weight.size();
    let normalized = hidden
        .shallow_clone()
        .internal_fused_rms_norm(normalized_shape, Some(&weight), eps)
        .0;
    let hidden = context.insert_tensor(hidden);
    let normalized = context.insert_tensor(normalized);
    return_int(context.insert_pair(FfcPair {
        local: hidden,
        global: normalized,
    }))
}

fn nn_apply_rope(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let cos = context.tensor(int_arg(args, 1, "cos")?)?.shallow_clone();
    let sin = context.tensor(int_arg(args, 2, "sin")?)?.shallow_clone();
    let output = apply_rope_tensor(input, &cos, &sin)?;
    return_tensor(context, output)
}

fn nn_apply_rope_pair(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let query = context.tensor(int_arg(args, 0, "query")?)?.shallow_clone();
    let key = context.tensor(int_arg(args, 1, "key")?)?.shallow_clone();
    let cos = context.tensor(int_arg(args, 2, "cos")?)?.shallow_clone();
    let sin = context.tensor(int_arg(args, 3, "sin")?)?.shallow_clone();
    let query_heads = query
        .size()
        .get(1)
        .copied()
        .ok_or_else(|| host_error("query must have at least 2 dimensions"))?;
    let key_heads = key
        .size()
        .get(1)
        .copied()
        .ok_or_else(|| host_error("key must have at least 2 dimensions"))?;
    let joined = Tensor::cat(&[&query, &key], 1);
    let rotated = apply_rope_tensor(joined, &cos, &sin)?;
    let query = rotated.narrow(1, 0, query_heads);
    let key = rotated.narrow(1, query_heads, key_heads);
    let query = context.insert_tensor(query);
    let key = context.insert_tensor(key);
    return_int(context.insert_pair(FfcPair {
        local: query,
        global: key,
    }))
}

fn apply_rope_tensor(input: Tensor, cos: &Tensor, sin: &Tensor) -> VmResult<Tensor> {
    let Some(&head_dim) = input.size().last() else {
        return Err(host_error("input must have at least one dimension"));
    };
    let half = head_dim / 2;
    let first = input.narrow(-1, 0, half);
    let second = input.narrow(-1, half, half);
    let rotated = Tensor::cat(&[&second.neg(), &first], -1);
    let output = input * cos + rotated * sin;
    Ok(output)
}

fn nn_scaled_dot_product_attention(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let query = context.tensor(int_arg(args, 0, "query")?)?.shallow_clone();
    let key = context.tensor(int_arg(args, 1, "key")?)?.shallow_clone();
    let value = context.tensor(int_arg(args, 2, "value")?)?.shallow_clone();
    let is_causal = bool_arg(args, 3, "is_causal")?;
    let output = Tensor::scaled_dot_product_attention(
        &query,
        &key,
        &value,
        None::<&Tensor>,
        0.0,
        is_causal,
        None,
        false,
    );
    return_tensor(context, output)
}

fn nn_scaled_dot_product_attention_masked(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let query = context.tensor(int_arg(args, 0, "query")?)?.shallow_clone();
    let key = context.tensor(int_arg(args, 1, "key")?)?.shallow_clone();
    let value = context.tensor(int_arg(args, 2, "value")?)?.shallow_clone();
    let mask = context.tensor(int_arg(args, 3, "mask")?)?.shallow_clone();
    let output = Tensor::scaled_dot_product_attention(
        &query,
        &key,
        &value,
        Some(&mask),
        0.0,
        false,
        None,
        false,
    );
    return_tensor(context, output)
}

fn nn_conv1d(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let weight = context.tensor(int_arg(args, 1, "weight")?)?.shallow_clone();
    let output_kind = input.kind();
    let bias_handle = int_arg(args, 2, "bias")?;
    let stride = int_arg(args, 3, "stride")?;
    let padding = int_arg(args, 4, "padding")?;
    let dilation = int_arg(args, 5, "dilation")?;
    let groups = int_arg(args, 6, "groups")?;
    let bias = if bias_handle == 0 {
        None
    } else {
        Some(context.tensor(bias_handle)?.shallow_clone())
    };
    let native_output = input.f_conv1d(
        &weight,
        bias.as_ref(),
        [stride],
        [padding],
        [dilation],
        groups,
    );
    let output = match native_output {
        Ok(output) => output,
        Err(native_err) => {
            let input_float = input.to_kind(Kind::Float);
            let weight_float = weight.to_kind(Kind::Float);
            if stride == 1
                && dilation == 1
                && input_float.size().len() == 3
                && weight_float.size().as_slice() == [groups, 1, 3]
                && input_float.size()[1] == groups
            {
                let out_len = input_float.size()[2] + (2 * padding) - 2;
                let padded = input_float.zero_pad1d(padding, padding);
                let w0 = weight_float.select(2, 0).view([1, groups, 1]);
                let w1 = weight_float.select(2, 1).view([1, groups, 1]);
                let w2 = weight_float.select(2, 2).view([1, groups, 1]);
                let mut output = padded.narrow(2, 0, out_len) * w0;
                output = output + padded.narrow(2, 1, out_len) * w1;
                output = output + padded.narrow(2, 2, out_len) * w2;
                if let Some(bias) = bias.as_ref() {
                    output = output + bias.to_kind(Kind::Float).view([1, groups, 1]);
                }
                output
            } else {
                return Err(host_error(format!("conv1d failed: {native_err}")));
            }
        }
    }
    .to_kind(output_kind);
    return_tensor(context, output)
}

fn nn_conv1d_step(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let state = context.tensor(int_arg(args, 0, "state")?)?.shallow_clone();
    let input = context.tensor(int_arg(args, 1, "input")?)?.shallow_clone();
    let weight_handle = int_arg(args, 2, "weight")?;
    let state_size = state.size();
    let input_size = input.size();
    if state_size.len() != 3 || input_size.len() != 3 {
        return Err(host_error(
            "conv1d_step expects state [B,C,2], input [B,C,1], weight [C,1,3]",
        ));
    }
    if state_size[0] != input_size[0]
        || state_size[1] != input_size[1]
        || state_size[2] != 2
        || input_size[2] != 1
    {
        return Err(host_error(
            "conv1d_step state/input shapes are incompatible",
        ));
    }
    let groups = input_size[1];
    let weights = cached_conv1d_step_weights(context, weight_handle, groups)?;
    let old0 = state.narrow(2, 0, 1);
    let old1 = state.narrow(2, 1, 1);
    let output =
        old0 * weights.w0 + old1.shallow_clone() * weights.w1 + input.shallow_clone() * weights.w2;
    let next_state = Tensor::cat(&[&old1, &input], 2);
    let local = context.insert_tensor(output);
    let global = context.insert_tensor(next_state);
    return_int(context.insert_pair(FfcPair { local, global }))
}

fn cached_conv1d_step_weights(
    context: &mut TorchContext,
    weight_handle: i64,
    groups: i64,
) -> VmResult<Conv1dStepWeights> {
    if let Some(weights) = context.conv1d_step_weights.get(&weight_handle) {
        return Ok(weights.clone());
    }
    let weight = context.tensor(weight_handle)?.shallow_clone();
    if weight.size().as_slice() != [groups, 1, 3] {
        return Err(host_error(
            "conv1d_step expects state [B,C,2], input [B,C,1], weight [C,1,3]",
        ));
    }
    let weights = Conv1dStepWeights {
        w0: weight.select(2, 0).view([1, groups, 1]),
        w1: weight.select(2, 1).view([1, groups, 1]),
        w2: weight.select(2, 2).view([1, groups, 1]),
    };
    context
        .conv1d_step_weights
        .insert(weight_handle, weights.clone());
    Ok(weights)
}

fn nn_conv2d(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let prefix = string_arg(args, 1, "weight prefix")?;
    let stride = int_arg(args, 2, "stride")?;
    let padding = int_arg(args, 3, "padding")?;
    let reflect = bool_arg(args, 4, "reflect")?;
    let has_bias = bool_arg(args, 5, "bias")?;
    let weight = context.weight(&format!("{prefix}.weight"))?.shallow_clone();
    let bias = if has_bias {
        Some(context.weight(&format!("{prefix}.bias"))?.shallow_clone())
    } else {
        None
    };
    let (input, padding) = if reflect && padding > 0 {
        (
            input.reflection_pad2d([padding, padding, padding, padding]),
            0,
        )
    } else {
        (input, padding)
    };
    let output = input
        .f_conv2d(
            &weight,
            bias.as_ref(),
            [stride, stride],
            [padding, padding],
            [1, 1],
            1,
        )
        .map_err(|err| host_error(format!("conv2d '{prefix}': {err}")))?;
    return_tensor(context, output)
}

fn nn_conv_transpose2d(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let prefix = string_arg(args, 1, "weight prefix")?;
    let weight = context.weight(&format!("{prefix}.weight"))?.shallow_clone();
    let bias = context.weight(&format!("{prefix}.bias"))?.shallow_clone();
    let output = input
        .f_conv_transpose2d(&weight, Some(&bias), [2, 2], [1, 1], [1, 1], 1, [1, 1])
        .map_err(|err| host_error(format!("conv_transpose2d '{prefix}': {err}")))?;
    return_tensor(context, output)
}

fn nn_batch_norm2d(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let prefix = string_arg(args, 1, "weight prefix")?;
    let weight = context.weight(&format!("{prefix}.weight"))?.shallow_clone();
    let bias = context.weight(&format!("{prefix}.bias"))?.shallow_clone();
    let mean = context
        .weight(&format!("{prefix}.running_mean"))?
        .shallow_clone();
    let variance = context
        .weight(&format!("{prefix}.running_var"))?
        .shallow_clone();
    let output = input
        .f_batch_norm(
            Some(&weight),
            Some(&bias),
            Some(&mean),
            Some(&variance),
            false,
            0.1,
            1e-5,
            true,
        )
        .map_err(|err| host_error(format!("batch_norm2d '{prefix}': {err}")))?;
    return_tensor(context, output)
}

fn image_lfm2_vl_patches(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let path = PathBuf::from(string_arg(args, 0, "path")?);
    let image = image::open(&path)
        .map_err(|err| host_error(format!("failed to read image {}: {err}", path.display())))?
        .resize_exact(512, 512, FilterType::Triangle)
        .to_rgb8();
    let raw = image.as_raw();
    let patch = 16usize;
    let grid = 32usize;
    let mut data = Vec::with_capacity(grid * grid * 3 * patch * patch);
    for patch_y in 0..grid {
        for patch_x in 0..grid {
            for y in 0..patch {
                for x in 0..patch {
                    for channel in 0..3usize {
                        let image_x = patch_x * patch + x;
                        let image_y = patch_y * patch + y;
                        let offset = (image_y * 512 + image_x) * 3 + channel;
                        let value = raw[offset] as f32 / 255.0;
                        data.push((value - 0.5) / 0.5);
                    }
                }
            }
        }
    }
    let output = Tensor::from_slice(&data)
        .view([1, 1024, 768])
        .to_device(context.device)
        .to_kind(
            requested_weight_kind()
                .ok()
                .flatten()
                .unwrap_or(Kind::Float),
        );
    return_tensor(context, output)
}

fn vl_siglip2_position_embedding(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let weight = context.tensor(int_arg(args, 0, "weight")?)?.shallow_clone();
    let height = int_arg(args, 1, "height")?;
    let width = int_arg(args, 2, "width")?;
    let max_len = int_arg(args, 3, "max_len")?;
    let size = weight.size();
    if size.len() != 2 {
        return Err(host_error(
            "position embedding weight must be [tokens, dim]",
        ));
    }
    let source = (size[0] as f64).sqrt() as i64;
    if source * source != size[0] {
        return Err(host_error("position embedding token count must be square"));
    }
    let embed_dim = size[1];
    if height <= 0 || width <= 0 || max_len < height * width {
        return Err(host_error("invalid position embedding target shape"));
    }
    let resized = weight
        .view([source, source, embed_dim])
        .permute([2, 0, 1])
        .unsqueeze(0)
        .to_kind(Kind::Float)
        .upsample_bilinear2d([height, width], false, None::<f64>, None::<f64>)
        .to_kind(weight.kind())
        .view([embed_dim, height * width])
        .transpose(0, 1);
    let output = if height * width == max_len {
        resized
    } else {
        let pad = resized
            .select(0, 0)
            .view([1, embed_dim])
            .repeat([max_len - height * width, 1]);
        Tensor::cat(&[&resized, &pad], 0)
    }
    .view([1, max_len, embed_dim]);
    return_tensor(context, output)
}

fn vl_pixel_unshuffle2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let size = input.size();
    if size.len() != 4 || size[1] % 2 != 0 || size[2] % 2 != 0 {
        return Err(host_error(
            "pixel_unshuffle2 expects [N,H,W,C] with even H/W",
        ));
    }
    let n = size[0];
    let h = size[1];
    let w = size[2];
    let c = size[3];
    let output = input
        .contiguous()
        .view([n, h, w / 2, c * 2])
        .permute([0, 2, 1, 3])
        .contiguous()
        .view([n, w / 2, h / 2, c * 4])
        .permute([0, 2, 1, 3])
        .contiguous();
    return_tensor(context, output)
}

fn vl_scatter_image_embeddings(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let input_ids = context
        .tensor(int_arg(args, 0, "input_ids")?)?
        .shallow_clone();
    let inputs_embeds = context
        .tensor(int_arg(args, 1, "inputs_embeds")?)?
        .shallow_clone();
    let image_features = context
        .tensor(int_arg(args, 2, "image_features")?)?
        .shallow_clone()
        .to_kind(inputs_embeds.kind());
    let image_token_id = int_arg(args, 3, "image_token_id")?;
    let mask = input_ids
        .eq(image_token_id)
        .unsqueeze(-1)
        .expand_as(&inputs_embeds);
    let output = inputs_embeds.masked_scatter(&mask, &image_features);
    return_tensor(context, output)
}
