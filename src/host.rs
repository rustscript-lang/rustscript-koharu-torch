use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use koharu_torch::{Device, Kind, Tensor};
use tokenizers::Tokenizer;
use vm::{
    CallOutcome, CallReturn, HostArgsFunction, Program, Value, Vm, VmError, VmResult, VmStatus,
    jit::JitConfig,
};

type HostOp = fn(&mut TorchContext, &[Value]) -> VmResult<CallOutcome>;

struct BoundHost {
    context: Arc<TorchContextCell>,
    name: &'static str,
    op: HostOp,
}

impl HostArgsFunction for BoundHost {
    fn call(&mut self, args: &[Value]) -> VmResult<CallOutcome> {
        let context = self.context.get();
        if context.host_op_profile_enabled {
            let started = Instant::now();
            let outcome = (self.op)(context, args);
            context.record_host_op(self.name, started.elapsed());
            outcome
        } else {
            (self.op)(context, args)
        }
    }
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
    next_tensor: i64,
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
        self.tensors
            .get(&handle)
            .ok_or_else(|| host_error(format!("unknown tensor handle {handle}")))
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
                next_tensor: 1,
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
        let status = koharu_torch::no_grad(|| vm.run()).map_err(|err| anyhow!(err.to_string()))?;
        if status != VmStatus::Halted {
            bail!("RustScript did not halt: {status:?}");
        }
        self.context.get().finish()
    }

    pub(crate) fn run_text(
        &self,
        program: Arc<Program>,
        args: Vec<String>,
    ) -> Result<ScriptTextOutput> {
        let _execution = self
            .execution
            .lock()
            .map_err(|_| anyhow!("Torch execution lock is poisoned"))?;
        self.context.get().begin_args(args);
        let mut vm = Vm::new_shared_with_jit_config(program, torch_jit_config());
        self.bind(&mut vm);
        let status = koharu_torch::no_grad(|| vm.run()).map_err(|err| anyhow!(err.to_string()))?;
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

pub struct TorchScriptRunner {
    runtime: TorchHostRuntime,
}

pub struct ScriptTextOutput {
    pub text: String,
    pub generated_tokens: Option<i64>,
    pub elapsed: Option<Duration>,
    pub decode_tokens: Option<i64>,
    pub decode_elapsed: Option<Duration>,
}

impl TorchScriptRunner {
    pub async fn new(device: Device) -> Result<Self> {
        crate::preload_libtorch()
            .await
            .context("failed to initialize LibTorch runtime")?;
        Ok(Self {
            runtime: TorchHostRuntime::new(device),
        })
    }

    pub fn run_text(&self, program: Arc<Program>, args: Vec<String>) -> Result<ScriptTextOutput> {
        self.runtime.run_text(program, args)
    }
}

const HOST_OPS: &[(&str, HostOp)] = &[
    ("torch::runtime::arg", runtime_arg),
    ("torch::runtime::arg_int", runtime_arg_int),
    ("torch::runtime::arg_int_or", runtime_arg_int_or),
    ("torch::runtime::input", runtime_input),
    ("torch::runtime::set_output", runtime_set_output),
    ("torch::runtime::set_text_output", runtime_set_text_output),
    ("torch::runtime::start_timer", runtime_start_timer),
    (
        "torch::runtime::start_decode_timer",
        runtime_start_decode_timer,
    ),
    ("torch::runtime::set_token_count", runtime_set_token_count),
    (
        "torch::runtime::set_decode_token_count",
        runtime_set_decode_token_count,
    ),
    ("torch::runtime::compact2", runtime_compact2),
    ("torch::cache::clear", cache_clear),
    ("torch::cache::has", cache_has),
    ("torch::cache::get", cache_get),
    ("torch::cache::set", cache_set),
    ("torch::tokenizer::load", tokenizer_load),
    ("torch::tokenizer::encode_chat", tokenizer_encode_chat),
    (
        "torch::tokenizer::decode_generated",
        tokenizer_decode_generated,
    ),
    ("torch::tokenizer::append_token", tokenizer_append_token),
    (
        "torch::tokenizer::append_token_tensor",
        tokenizer_append_token_tensor,
    ),
    (
        "torch::tokenizer::clear_generated_tokens",
        tokenizer_clear_generated_tokens,
    ),
    (
        "torch::tokenizer::push_generated_token_tensor",
        tokenizer_push_generated_token_tensor,
    ),
    (
        "torch::tokenizer::decode_generated_tokens",
        tokenizer_decode_generated_tokens,
    ),
    ("torch::tokenizer::single_token", tokenizer_single_token),
    ("torch::tokenizer::is_eos", tokenizer_is_eos),
    ("torch::weights::load", weights_load),
    ("torch::weights::get", weights_get),
    ("torch::weights::get_or", weights_get_or),
    ("torch::pair::new", pair_new),
    ("torch::pair::local", pair_local),
    ("torch::pair::global", pair_global),
    ("torch::tensor::size", tensor_size),
    ("torch::tensor::shape", tensor_shape),
    ("torch::tensor::to_float", tensor_to_float),
    ("torch::tensor::to_bfloat16", tensor_to_bfloat16),
    ("torch::tensor::ones_like", tensor_ones_like),
    ("torch::tensor::arange", tensor_arange),
    ("torch::tensor::causal_mask", tensor_causal_mask),
    ("torch::tensor::rope_cos", tensor_rope_cos),
    ("torch::tensor::rope_sin", tensor_rope_sin),
    ("torch::tensor::rope_cos_at", tensor_rope_cos_at),
    ("torch::tensor::rope_sin_at", tensor_rope_sin_at),
    ("torch::tensor::add", tensor_add),
    ("torch::tensor::sub", tensor_sub),
    ("torch::tensor::mul", tensor_mul),
    ("torch::tensor::add_scalar", tensor_add_scalar),
    ("torch::tensor::mul_scalar", tensor_mul_scalar),
    ("torch::tensor::div_scalar", tensor_div_scalar),
    ("torch::tensor::pow_scalar", tensor_pow_scalar),
    ("torch::tensor::mean_dim", tensor_mean_dim),
    ("torch::tensor::rsqrt", tensor_rsqrt),
    ("torch::tensor::neg", tensor_neg),
    ("torch::tensor::cos", tensor_cos),
    ("torch::tensor::sin", tensor_sin),
    ("torch::tensor::matmul", tensor_matmul),
    ("torch::tensor::softmax", tensor_softmax),
    ("torch::tensor::masked_fill", tensor_masked_fill),
    ("torch::tensor::cat2", tensor_cat2),
    ("torch::tensor::stack2", tensor_stack2),
    ("torch::tensor::chunk", tensor_chunk),
    ("torch::tensor::narrow", tensor_narrow),
    ("torch::tensor::tail", tensor_tail),
    ("torch::tensor::transpose", tensor_transpose),
    ("torch::tensor::unsqueeze", tensor_unsqueeze),
    ("torch::tensor::repeat_interleave", tensor_repeat_interleave),
    ("torch::tensor::argmax_int", tensor_argmax_int),
    ("torch::tensor::argmax_token", tensor_argmax_token),
    ("torch::tensor::pad_reflect2d", tensor_pad_reflect2d),
    ("torch::tensor::relu", tensor_relu),
    ("torch::tensor::sigmoid", tensor_sigmoid),
    ("torch::tensor::silu", tensor_silu),
    ("torch::tensor::swiglu", tensor_swiglu),
    ("torch::tensor::contiguous", tensor_contiguous),
    ("torch::tensor::permute3", tensor_permute3),
    ("torch::tensor::permute4", tensor_permute4),
    ("torch::tensor::permute5", tensor_permute5),
    ("torch::tensor::view2", tensor_view2),
    ("torch::tensor::view3", tensor_view3),
    ("torch::tensor::view4", tensor_view4),
    ("torch::tensor::view5", tensor_view5),
    ("torch::tensor::select", tensor_select),
    ("torch::tensor::real", tensor_real),
    ("torch::tensor::imag", tensor_imag),
    ("torch::tensor::complex", tensor_complex),
    ("torch::tensor::fft_rfftn2", tensor_fft_rfftn2),
    ("torch::tensor::fft_irfftn2", tensor_fft_irfftn2),
    ("torch::tensor::avg_pool2d_2", tensor_avg_pool2d_2),
    ("torch::nn::embedding", nn_embedding),
    ("torch::nn::linear", nn_linear),
    ("torch::nn::swiglu_linear", nn_swiglu_linear),
    ("torch::nn::rms_norm", nn_rms_norm),
    ("torch::nn::add_rms_norm", nn_add_rms_norm),
    ("torch::nn::apply_rope", nn_apply_rope),
    ("torch::nn::apply_rope_pair", nn_apply_rope_pair),
    (
        "torch::nn::scaled_dot_product_attention",
        nn_scaled_dot_product_attention,
    ),
    ("torch::nn::conv1d", nn_conv1d),
    ("torch::nn::conv1d_step", nn_conv1d_step),
    ("torch::nn::conv2d", nn_conv2d),
    ("torch::nn::conv_transpose2d", nn_conv_transpose2d),
    ("torch::nn::batch_norm2d", nn_batch_norm2d),
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
        _ => Err(VmError::TypeMismatch("int")),
    }
}

fn bool_arg(args: &[Value], index: usize, label: &str) -> VmResult<bool> {
    match arg(args, index, label)? {
        Value::Bool(value) => Ok(*value),
        _ => Err(VmError::TypeMismatch("bool")),
    }
}

fn float_arg(args: &[Value], index: usize, label: &str) -> VmResult<f64> {
    match arg(args, index, label)? {
        Value::Float(value) => Ok(*value),
        Value::Int(value) => Ok(*value as f64),
        _ => Err(VmError::TypeMismatch("float")),
    }
}

fn string_arg<'a>(args: &'a [Value], index: usize, label: &str) -> VmResult<&'a str> {
    match arg(args, index, label)? {
        Value::String(value) => Ok(value.as_str()),
        _ => Err(VmError::TypeMismatch("string")),
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

fn runtime_input(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let index = usize::try_from(int_arg(args, 0, "index")?)
        .map_err(|_| host_error("input index must be non-negative"))?;
    let handle = *context
        .inputs
        .get(index)
        .ok_or_else(|| host_error(format!("unknown input index {index}")))?;
    return_int(handle)
}

fn runtime_arg(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let index = usize::try_from(int_arg(args, 0, "index")?)
        .map_err(|_| host_error("argument index must be non-negative"))?;
    let value = context
        .args
        .get(index)
        .ok_or_else(|| host_error(format!("unknown runtime argument {index}")))?
        .clone();
    return_value(Value::String(value.into()))
}

fn runtime_arg_int(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let index = usize::try_from(int_arg(args, 0, "index")?)
        .map_err(|_| host_error("argument index must be non-negative"))?;
    let value = context
        .args
        .get(index)
        .ok_or_else(|| host_error(format!("unknown runtime argument {index}")))?
        .parse::<i64>()
        .map_err(|err| host_error(format!("runtime argument {index} is not an int: {err}")))?;
    return_int(value)
}

fn runtime_arg_int_or(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let index = usize::try_from(int_arg(args, 0, "index")?)
        .map_err(|_| host_error("argument index must be non-negative"))?;
    let default = int_arg(args, 1, "default")?;
    let Some(value) = context.args.get(index) else {
        return return_int(default);
    };
    let value = value
        .parse::<i64>()
        .map_err(|err| host_error(format!("runtime argument {index} is not an int: {err}")))?;
    return_int(value)
}

fn weights_load(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let path = PathBuf::from(string_arg(args, 0, "path")?);
    let target_kind = requested_weight_kind().map_err(host_error)?;
    if context.weights_path.as_deref() != Some(path.as_path())
        || context.weights_kind != target_kind
    {
        context.weight_handles.clear();
        context.conv1d_step_weights.clear();
        let weights = Tensor::read_safetensors(&path)
            .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))?
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

fn weights_get(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?;
    if let Some(handle) = cached_weight_handle(context, name) {
        return return_int(handle);
    } else {
        let tensor = get_or_build_weight(context, name)?;
        return_weight_tensor(context, name, tensor)
    }
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

fn runtime_set_output(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let handle = int_arg(args, 0, "tensor")?;
    context.tensor(handle)?;
    context.output = Some(handle);
    return_value(Value::Bool(true))
}

fn runtime_set_text_output(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let text = string_arg(args, 0, "text")?.to_owned();
    context.text_output = Some(text);
    return_value(Value::Bool(true))
}

fn runtime_start_timer(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    if !args.is_empty() {
        return Err(host_error("start_timer takes no arguments"));
    }
    context.generation_started_at = Some(Instant::now());
    context.generated_tokens = None;
    context.decode_started_at = None;
    context.decode_tokens = None;
    context.generated_token_tensors.clear();
    return_value(Value::Bool(true))
}

fn runtime_start_decode_timer(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    if !args.is_empty() {
        return Err(host_error("start_decode_timer takes no arguments"));
    }
    context.decode_started_at = Some(Instant::now());
    context.decode_tokens = None;
    return_value(Value::Bool(true))
}

fn runtime_set_token_count(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let generated_tokens = int_arg(args, 0, "generated tokens")?;
    if generated_tokens < 0 {
        return Err(host_error("generated token count must be non-negative"));
    }
    context.generated_tokens = Some(generated_tokens);
    return_value(Value::Bool(true))
}

fn runtime_set_decode_token_count(
    context: &mut TorchContext,
    args: &[Value],
) -> VmResult<CallOutcome> {
    let decode_tokens = int_arg(args, 0, "decode tokens")?;
    if decode_tokens < 0 {
        return Err(host_error("decode token count must be non-negative"));
    }
    context.decode_tokens = Some(decode_tokens);
    return_value(Value::Bool(true))
}

fn runtime_compact2(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let first = int_arg(args, 0, "first")?;
    let second = int_arg(args, 1, "second")?;
    let mut keep = HashSet::new();
    keep.insert(first);
    keep.insert(second);
    keep.extend(context.inputs.iter().copied());
    keep.extend(context.weight_handles.values().copied());
    if let Some(output) = context.output {
        keep.insert(output);
    }
    context.tensors.retain(|handle, _| keep.contains(handle));
    context.pairs.clear();
    context
        .weight_handles
        .retain(|_, handle| context.tensors.contains_key(handle));
    let count = i64::try_from(context.tensors.len())
        .map_err(|_| host_error("tensor count exceeds RustScript integer range"))?;
    return_int(count)
}

fn cache_clear(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    if !args.is_empty() {
        return Err(host_error("cache clear takes no arguments"));
    }
    context.cache.clear();
    return_value(Value::Bool(true))
}

fn cache_has(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?;
    return_value(Value::Bool(context.cache.contains_key(name)))
}

fn cache_get(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?;
    let tensor = context
        .cache
        .get(name)
        .ok_or_else(|| host_error(format!("missing cache tensor '{name}'")))?
        .shallow_clone();
    return_tensor(context, tensor)
}

fn cache_set(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let name = string_arg(args, 0, "name")?.to_owned();
    let tensor = context.tensor(int_arg(args, 1, "tensor")?)?.shallow_clone();
    context.cache.insert(name, tensor);
    return_value(Value::Bool(true))
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

fn pair_new(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let local = int_arg(args, 0, "local")?;
    let global = int_arg(args, 1, "global")?;
    context.tensor(local)?;
    if global != 0 {
        context.tensor(global)?;
    }
    return_int(context.insert_pair(FfcPair { local, global }))
}

fn pair_local(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    return_int(context.pair(int_arg(args, 0, "pair")?)?.local)
}

fn pair_global(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    return_int(context.pair(int_arg(args, 0, "pair")?)?.global)
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

fn tensor_arange(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let end = int_arg(args, 0, "end")?;
    let output = Tensor::arange(end, (Kind::Int64, context.device));
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
    let input = context.tensor(int_arg(args, 0, "tensor")?)?.shallow_clone();
    let weight = context.tensor(int_arg(args, 1, "weight")?)?.shallow_clone();
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
