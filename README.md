# Flint

[![Crates.io](https://img.shields.io/crates/v/flint-ai.svg)](https://crates.io/crates/flint-ai)

> Flint is all you need to light the Torch.

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

## CLI

The `flint-ai` binary has explicit modes:

```text
flint-ai --llm --script scripts/lfm2.rss [--device cuda:0] <args...>
flint-ai --llm --script scripts/flux_klein_encode_prompt.rss <qwen3.safetensors> <tokenizer.json> <prompt> <prompt.safetensors>
flint-ai --lama --weights model.safetensors --image input.png --mask mask.png --output output.png [--device cuda:0]
flint-ai --sd --script scripts/flux_klein.rss <diffusion-model> <vae> <llm> <prompt> <output.png> [width height steps seed cfg backend params-backend max-vram wtype sample-method scheduler]
flint-ai --sd --script scripts/ggml_devices.rss [backend]
```

When `--device` is omitted, the CLI initializes LibTorch and selects `cuda:0`
when CUDA is available. Passing `--device` overrides that selection.

`scripts/flux_klein.rss` builds a FLUX.2 Klein text-to-image run from the
low-level `flint::sd::*` wrappers around the stable-diffusion.cpp C API
packaged by `koharu-runtime`; pass the FLUX.2 Klein diffusion model, VAE, and
Qwen3 text encoder paths explicitly.

`scripts/flux_klein_encode_prompt.rss` runs a Qwen3 text encoder with
RustScript torch operations and writes a koharu-ml-compatible prompt embedding
file. The safetensors file contains one tensor named `prompt_embeds`; for
Qwen3-4B FLUX.2 Klein this is expected to be `[1, 512, 7680]`.

## Host functions

All functions are registered under the `flint` namespace.

### Runtime

Runtime arguments, host inputs, outputs, and tensor lifetime control:

```text
flint::runtime::arg
flint::runtime::arg_or
flint::runtime::arg_int
flint::runtime::arg_int_or
flint::runtime::arg_float_or
flint::runtime::input
flint::runtime::set_output
flint::runtime::set_text_output
flint::runtime::compact2
```

Arguments are addressed by zero-based index. `set_output` publishes a tensor
handle, while `set_text_output` publishes generated text. The compact helper
keeps long-running scripts from retaining unused temporary tensors.

### GGML

GGML backend discovery helpers:

```text
flint::ggml::load_backends
flint::ggml::list_devices
flint::ggml::stable_diffusion_package_dir
flint::ggml::load_stable_diffusion_backends
flint::ggml::list_stable_diffusion_devices
```

`load_backends` and `list_devices` accept a directory containing `ggml.dll` or
`libggml.so`, or a file inside that directory. The stable-diffusion helpers
select the packaged ggml runtime with the same backend strings accepted by the
SD host functions.

### Stable Diffusion

Low-level stable-diffusion.cpp host functions:

```text
flint::sd::ctx_params_init
flint::sd::ctx_params_set_paths
flint::sd::ctx_params_set_backend
flint::sd::ctx_params_set_wtype
flint::sd::ctx_params_set_vae_format
flint::sd::ctx_params_set_flags
flint::sd::new_sd_ctx
flint::sd::free_sd_ctx
flint::sd::img_gen_params_init
flint::sd::img_gen_params_set_prompt
flint::sd::img_gen_params_set_size
flint::sd::img_gen_params_set_sample
flint::sd::img_gen_params_set_sampler
flint::sd::str_to_sample_method
flint::sd::str_to_scheduler
flint::sd::sample_method_name
flint::sd::scheduler_name
flint::sd::get_default_sample_method
flint::sd::get_default_scheduler
flint::sd::generate_image
flint::sd::images_save
flint::sd::free_sd_images
```

The `flint::sd::*` functions expose C API-shaped resource handles for context
params, contexts, image generation params, and image batches. Optional backend
strings follow sd.cpp names such as `cpu`, `cuda0`, or assignment specs like
`te=cpu,vae=cpu,diffusion=cuda0`. Passing `cpu`, `cuda*`, or `vulkan*` also
selects the matching packaged stable-diffusion.cpp runtime; `auto` keeps
koharu-runtime's automatic choice.

`scripts/flux_klein.rss` accepts optional `sample-method` and `scheduler`
arguments after `wtype`. Use `auto` to keep stable-diffusion.cpp defaults, or
pass upstream names such as `euler`, `euler_a`, `dpm++2m`,
`dpm++2m_sde`, `flux2`, `simple`, `karras`, or `beta`.

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
flint::tokenizer::encode_padded
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
tensors and are passed through the VM as handles. `encode_padded` returns a
pair: local is padded input ids and global is the attention mask.

### Weights

Safetensors loading and lookup:

```text
flint::weights::load
flint::weights::get
flint::weights::get_indexed
flint::weights::get_or
```

`load` reads a safetensors file, or every `.safetensors` file in a directory,
onto the runner device. `get` resolves a tensor by key, `get_indexed` formats
layer names from prefix/index/suffix, and `get_or` supports alternative keys for
model formats that use different parameter names.

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
flint::tensor::save_safetensors
flint::tensor::load_safetensors
flint::tensor::to_float
flint::tensor::to_bfloat16
flint::tensor::ones_like
flint::tensor::arange
flint::tensor::causal_mask
flint::tensor::causal_padding_mask
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
dimensions explicitly. The safetensors helpers read and write one named tensor;
use the name `prompt_embeds` for FLUX prompt embedding files.

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
flint::nn::scaled_dot_product_attention_masked
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
