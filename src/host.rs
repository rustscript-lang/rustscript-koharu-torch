use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};

use anyhow::{Context, Result, anyhow, bail};
use koharu_torch::{Device, Tensor};
use vm::{
    CallOutcome, CallReturn, HostArgsFunction, Program, Value, Vm, VmError, VmResult, VmStatus,
};

type HostOp = fn(&mut TorchContext, &[Value]) -> VmResult<CallOutcome>;

struct BoundHost {
    context: Arc<Mutex<TorchContext>>,
    op: HostOp,
}

impl HostArgsFunction for BoundHost {
    fn call(&mut self, args: &[Value]) -> VmResult<CallOutcome> {
        let mut context = self
            .context
            .lock()
            .map_err(|_| host_error("Torch host context lock is poisoned"))?;
        (self.op)(&mut context, args)
    }
}

#[derive(Debug, Clone, Copy)]
struct FfcPair {
    local: i64,
    global: i64,
}

struct TorchContext {
    device: Device,
    weights: HashMap<String, Tensor>,
    weights_path: Option<PathBuf>,
    tensors: HashMap<i64, Tensor>,
    pairs: HashMap<i64, FfcPair>,
    inputs: Vec<i64>,
    args: Vec<String>,
    output: Option<i64>,
    next_tensor: i64,
    next_pair: i64,
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
        self.tensors.clear();
        self.pairs.clear();
        self.inputs.clear();
        self.output = None;
        self.args = args;
        self.next_tensor = 1;
        self.next_pair = 1;
        let image = self.insert_tensor(image);
        let mask = self.insert_tensor(mask);
        self.inputs.extend([image, mask]);
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
        self.args.clear();
        self.output = None;
        Ok(output)
    }
}

pub(crate) struct TorchHostRuntime {
    context: Arc<Mutex<TorchContext>>,
    execution: Mutex<()>,
}

impl TorchHostRuntime {
    pub(crate) fn new(device: Device) -> Self {
        Self {
            context: Arc::new(Mutex::new(TorchContext {
                device,
                weights: HashMap::new(),
                weights_path: None,
                tensors: HashMap::new(),
                pairs: HashMap::new(),
                inputs: Vec::new(),
                args: Vec::new(),
                output: None,
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
        self.lock()?.begin(image, mask, args);
        let mut vm = Vm::new_shared(program);
        self.bind(&mut vm);
        let status = vm.run().map_err(|err| anyhow!(err.to_string()))?;
        if status != VmStatus::Halted {
            bail!("RustScript did not halt: {status:?}");
        }
        self.lock()?.finish()
    }

    fn lock(&self) -> Result<MutexGuard<'_, TorchContext>> {
        self.context
            .lock()
            .map_err(|_| anyhow!("Torch host context lock is poisoned"))
    }

    fn bind(&self, vm: &mut Vm) {
        for (name, op) in HOST_OPS {
            vm.bind_args_function(
                *name,
                Box::new(BoundHost {
                    context: Arc::clone(&self.context),
                    op: *op,
                }),
            );
        }
    }
}

const HOST_OPS: &[(&str, HostOp)] = &[
    ("torch::runtime::arg", runtime_arg),
    ("torch::runtime::input", runtime_input),
    ("torch::runtime::set_output", runtime_set_output),
    ("torch::weights::load", weights_load),
    ("torch::pair::new", pair_new),
    ("torch::pair::local", pair_local),
    ("torch::pair::global", pair_global),
    ("torch::tensor::size", tensor_size),
    ("torch::tensor::ones_like", tensor_ones_like),
    ("torch::tensor::add", tensor_add),
    ("torch::tensor::sub", tensor_sub),
    ("torch::tensor::mul", tensor_mul),
    ("torch::tensor::cat2", tensor_cat2),
    ("torch::tensor::stack2", tensor_stack2),
    ("torch::tensor::pad_reflect2d", tensor_pad_reflect2d),
    ("torch::tensor::relu", tensor_relu),
    ("torch::tensor::sigmoid", tensor_sigmoid),
    ("torch::tensor::contiguous", tensor_contiguous),
    ("torch::tensor::permute5", tensor_permute5),
    ("torch::tensor::view4", tensor_view4),
    ("torch::tensor::view5", tensor_view5),
    ("torch::tensor::select", tensor_select),
    ("torch::tensor::real", tensor_real),
    ("torch::tensor::imag", tensor_imag),
    ("torch::tensor::complex", tensor_complex),
    ("torch::tensor::fft_rfftn2", tensor_fft_rfftn2),
    ("torch::tensor::fft_irfftn2", tensor_fft_irfftn2),
    ("torch::tensor::avg_pool2d_2", tensor_avg_pool2d_2),
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

fn weights_load(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let path = PathBuf::from(string_arg(args, 0, "path")?);
    if context.weights_path.as_deref() != Some(path.as_path()) {
        let weights = Tensor::read_safetensors(&path)
            .map_err(|err| host_error(format!("failed to read {}: {err}", path.display())))?
            .into_iter()
            .map(|(name, tensor)| (name, tensor.to_device(context.device)))
            .collect();
        context.weights = weights;
        context.weights_path = Some(path);
    }
    let count = i64::try_from(context.weights.len())
        .map_err(|_| host_error("weight count exceeds RustScript integer range"))?;
    return_int(count)
}

fn runtime_set_output(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    let handle = int_arg(args, 0, "tensor")?;
    context.tensor(handle)?;
    context.output = Some(handle);
    return_value(Value::Bool(true))
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
    let dim = usize::try_from(int_arg(args, 1, "dim")?)
        .map_err(|_| host_error("dimension must be non-negative"))?;
    let value = tensor
        .size()
        .get(dim)
        .copied()
        .ok_or_else(|| host_error(format!("dimension {dim} is out of range")))?;
    return_int(value)
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

fn tensor_add(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, |left, right| left + right)
}

fn tensor_sub(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, |left, right| left - right)
}

fn tensor_mul(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    binary_tensor(context, args, |left, right| left * right)
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

fn tensor_contiguous(context: &mut TorchContext, args: &[Value]) -> VmResult<CallOutcome> {
    unary_tensor(context, args, Tensor::contiguous)
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
