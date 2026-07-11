# Flint

[![Crates.io](https://img.shields.io/crates/v/flint-ai.svg)](https://crates.io/crates/flint-ai)

> Script the graph. Run it native.

Flint is an AI framework for defining and running model graphs in
[RustScript](https://github.com/rustscript-lang/rustscript). It exposes native
tensor operations from [`koharu-torch`](https://github.com/mayocream/koharu) as
RustScript host functions, keeping model architecture and inference control in
scripts while Rust manages devices, weights, tokenization, and execution.

## How it works

Flint compiles a RustScript program and runs it with `TorchScriptRunner` on a
selected CPU, CUDA, MPS, or Vulkan device. Before execution, the runner binds
the `flint::*` host functions listed below.

Tensor and pair values cross the VM boundary as opaque integer handles. A
RustScript program uses those handles to compose a graph, then publishes its
result with `flint::runtime::set_output` or
`flint::runtime::set_text_output`. Safetensors weights are loaded directly onto
the selected device and reused by handle during the run.

A typical integration follows this flow:

1. Compile a `.rss` source file with RustScript.
2. Create `TorchScriptRunner::new(device)` to initialize LibTorch.
3. Pass the compiled program and string arguments to `run_text`.
4. Read the published text from `ScriptTextOutput`.

## Host functions

All functions are registered under the `flint` namespace.

### Runtime

Runtime arguments, host inputs, outputs, and tensor lifetime control:

```text
flint::runtime::arg
flint::runtime::arg_int
flint::runtime::arg_int_or
flint::runtime::input
flint::runtime::set_output
flint::runtime::set_text_output
flint::runtime::compact2
```

Arguments are addressed by zero-based index. `set_output` publishes a tensor
handle, while `set_text_output` publishes generated text. The compact helper
keeps long-running scripts from retaining unused temporary tensors.

### Cache

Named tensor storage for state shared across steps within one execution:

```text
flint::cache::clear
flint::cache::has
flint::cache::get
flint::cache::set
```

### Tokenizer

Tokenizer loading, chat encoding, incremental token collection, decoding, and
end-of-sequence checks:

```text
flint::tokenizer::load
flint::tokenizer::encode_chat
flint::tokenizer::decode_generated
flint::tokenizer::append_token
flint::tokenizer::append_token_tensor
flint::tokenizer::clear_generated_tokens
flint::tokenizer::push_generated_token_tensor
flint::tokenizer::decode_generated_tokens
flint::tokenizer::single_token
flint::tokenizer::is_eos
```

Load a tokenizer before encoding or decoding. Token tensors remain native
tensors and are passed through the VM as handles.

### Weights

Safetensors loading and lookup:

```text
flint::weights::load
flint::weights::get
flint::weights::get_or
```

`load` reads a safetensors file onto the runner device. `get` resolves a tensor
by key, and `get_or` supports alternative keys for model formats that use
different parameter names.

### Pairs

Two-handle return values used by fused operations and local/global branches:

```text
flint::pair::new
flint::pair::local
flint::pair::global
```

### Tensor operations

Shape inspection, casting, construction, arithmetic, indexing, activation,
layout, complex tensors, FFT, and pooling:

```text
flint::tensor::size
flint::tensor::shape
flint::tensor::to_float
flint::tensor::to_bfloat16
flint::tensor::ones_like
flint::tensor::arange
flint::tensor::causal_mask
flint::tensor::rope_cos
flint::tensor::rope_sin
flint::tensor::rope_cos_at
flint::tensor::rope_sin_at
flint::tensor::add
flint::tensor::sub
flint::tensor::mul
flint::tensor::add_scalar
flint::tensor::mul_scalar
flint::tensor::div_scalar
flint::tensor::pow_scalar
flint::tensor::mean_dim
flint::tensor::rsqrt
flint::tensor::neg
flint::tensor::cos
flint::tensor::sin
flint::tensor::matmul
flint::tensor::softmax
flint::tensor::masked_fill
flint::tensor::cat2
flint::tensor::stack2
flint::tensor::chunk
flint::tensor::narrow
flint::tensor::tail
flint::tensor::transpose
flint::tensor::unsqueeze
flint::tensor::repeat_interleave
flint::tensor::argmax_int
flint::tensor::argmax_token
flint::tensor::pad_reflect2d
flint::tensor::relu
flint::tensor::sigmoid
flint::tensor::silu
flint::tensor::swiglu
flint::tensor::contiguous
flint::tensor::permute3
flint::tensor::permute4
flint::tensor::permute5
flint::tensor::view2
flint::tensor::view3
flint::tensor::view4
flint::tensor::view5
flint::tensor::select
flint::tensor::real
flint::tensor::imag
flint::tensor::complex
flint::tensor::fft_rfftn2
flint::tensor::fft_irfftn2
flint::tensor::avg_pool2d_2
```

Tensor operations accept opaque tensor handles and return a new handle unless
the function name indicates a scalar result, such as `size` or `argmax_int`.
Rank-specific functions such as `view3` and `permute4` take that number of
dimensions explicitly.

### Neural network operations

Common model layers and fused inference operations:

```text
flint::nn::embedding
flint::nn::linear
flint::nn::swiglu_linear
flint::nn::rms_norm
flint::nn::add_rms_norm
flint::nn::apply_rope
flint::nn::apply_rope_pair
flint::nn::scaled_dot_product_attention
flint::nn::conv1d
flint::nn::conv1d_step
flint::nn::conv2d
flint::nn::conv_transpose2d
flint::nn::batch_norm2d
```

Fused functions may return a pair handle. Use `flint::pair::local` and
`flint::pair::global` to access each tensor result. For optional tensor
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
