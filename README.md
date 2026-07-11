# Flint

> Script the graph. Run it native.

Flint is an AI framework for defining and running model graphs in
[RustScript](https://github.com/rustscript-lang/rustscript). It exposes native
tensor operations from [`koharu-torch`](https://github.com/mayocream/koharu) as
RustScript host functions, keeping model architecture and inference control in
scripts while Rust manages devices, weights, tokenization, and execution.

## How it works

Flint compiles a RustScript program and runs it with `TorchScriptRunner` on a
selected CPU, CUDA, MPS, or Vulkan device. Before execution, the runner binds
the `torch::*` host functions listed below.

Tensor and pair values cross the VM boundary as opaque integer handles. A
RustScript program uses those handles to compose a graph, then publishes its
result with `torch::runtime::set_output` or
`torch::runtime::set_text_output`. Safetensors weights are loaded directly onto
the selected device and reused by handle during the run.

A typical integration follows this flow:

1. Compile a `.rss` source file with RustScript.
2. Create `TorchScriptRunner::new(device)` to initialize LibTorch.
3. Pass the compiled program and string arguments to `run_text`.
4. Read the published text from `ScriptTextOutput`.

## Host functions

All functions are registered under the `torch` namespace.

### Runtime

Runtime arguments, host inputs, outputs, and tensor lifetime control:

```text
torch::runtime::arg
torch::runtime::arg_int
torch::runtime::arg_int_or
torch::runtime::input
torch::runtime::set_output
torch::runtime::set_text_output
torch::runtime::compact2
```

Arguments are addressed by zero-based index. `set_output` publishes a tensor
handle, while `set_text_output` publishes generated text. The compact helper
keeps long-running scripts from retaining unused temporary tensors.

### Cache

Named tensor storage for state shared across steps within one execution:

```text
torch::cache::clear
torch::cache::has
torch::cache::get
torch::cache::set
```

### Tokenizer

Tokenizer loading, chat encoding, incremental token collection, decoding, and
end-of-sequence checks:

```text
torch::tokenizer::load
torch::tokenizer::encode_chat
torch::tokenizer::decode_generated
torch::tokenizer::append_token
torch::tokenizer::append_token_tensor
torch::tokenizer::clear_generated_tokens
torch::tokenizer::push_generated_token_tensor
torch::tokenizer::decode_generated_tokens
torch::tokenizer::single_token
torch::tokenizer::is_eos
```

Load a tokenizer before encoding or decoding. Token tensors remain native
tensors and are passed through the VM as handles.

### Weights

Safetensors loading and lookup:

```text
torch::weights::load
torch::weights::get
torch::weights::get_or
```

`load` reads a safetensors file onto the runner device. `get` resolves a tensor
by key, and `get_or` supports alternative keys for model formats that use
different parameter names.

### Pairs

Two-handle return values used by fused operations and local/global branches:

```text
torch::pair::new
torch::pair::local
torch::pair::global
```

### Tensor operations

Shape inspection, casting, construction, arithmetic, indexing, activation,
layout, complex tensors, FFT, and pooling:

```text
torch::tensor::size
torch::tensor::shape
torch::tensor::to_float
torch::tensor::to_bfloat16
torch::tensor::ones_like
torch::tensor::arange
torch::tensor::causal_mask
torch::tensor::rope_cos
torch::tensor::rope_sin
torch::tensor::rope_cos_at
torch::tensor::rope_sin_at
torch::tensor::add
torch::tensor::sub
torch::tensor::mul
torch::tensor::add_scalar
torch::tensor::mul_scalar
torch::tensor::div_scalar
torch::tensor::pow_scalar
torch::tensor::mean_dim
torch::tensor::rsqrt
torch::tensor::neg
torch::tensor::cos
torch::tensor::sin
torch::tensor::matmul
torch::tensor::softmax
torch::tensor::masked_fill
torch::tensor::cat2
torch::tensor::stack2
torch::tensor::chunk
torch::tensor::narrow
torch::tensor::tail
torch::tensor::transpose
torch::tensor::unsqueeze
torch::tensor::repeat_interleave
torch::tensor::argmax_int
torch::tensor::argmax_token
torch::tensor::pad_reflect2d
torch::tensor::relu
torch::tensor::sigmoid
torch::tensor::silu
torch::tensor::swiglu
torch::tensor::contiguous
torch::tensor::permute3
torch::tensor::permute4
torch::tensor::permute5
torch::tensor::view2
torch::tensor::view3
torch::tensor::view4
torch::tensor::view5
torch::tensor::select
torch::tensor::real
torch::tensor::imag
torch::tensor::complex
torch::tensor::fft_rfftn2
torch::tensor::fft_irfftn2
torch::tensor::avg_pool2d_2
```

Tensor operations accept opaque tensor handles and return a new handle unless
the function name indicates a scalar result, such as `size` or `argmax_int`.
Rank-specific functions such as `view3` and `permute4` take that number of
dimensions explicitly.

### Neural network operations

Common model layers and fused inference operations:

```text
torch::nn::embedding
torch::nn::linear
torch::nn::swiglu_linear
torch::nn::rms_norm
torch::nn::add_rms_norm
torch::nn::apply_rope
torch::nn::apply_rope_pair
torch::nn::scaled_dot_product_attention
torch::nn::conv1d
torch::nn::conv1d_step
torch::nn::conv2d
torch::nn::conv_transpose2d
torch::nn::batch_norm2d
```

Fused functions may return a pair handle. Use `torch::pair::local` and
`torch::pair::global` to access each tensor result. For optional tensor
arguments such as a linear bias, handle `0` represents no tensor.

## Configuration

- `KOHARU_TORCH_WEIGHT_KIND` selects the floating-point kind used while loading
  weights: `native`, `half`, `bf16`, or `float`.

## Links

- [Flint repository](https://github.com/rustscript-lang/flint)
- [Examples](https://github.com/rustscript-lang/flint/tree/main/examples)
- [RustScript model programs](https://github.com/rustscript-lang/flint/tree/main/scripts)
- [RustScript](https://github.com/rustscript-lang/rustscript)
- [Koharu](https://github.com/mayocream/koharu)
