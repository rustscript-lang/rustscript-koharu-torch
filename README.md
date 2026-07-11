# rustscript-koharu-torch

Standalone integration between RustScript and `koharu-torch`. The Torch crate is
resolved directly from `mayocream/koharu` at branch `refactor/0705`; this project
does not vendor or modify the Koharu repository.

The current end-to-end language model graph is LFM2. Rust owns device selection,
tokenizer/model host functions, and opaque tensor handles. RustScript receives
the safetensors path as a runtime argument and loads it through the
`torch::weights::load` host function. The LFM2 decoder, KV/conv caches, token
loop, and throughput reporting are implemented in [`scripts/lfm2.rss`](scripts/lfm2.rss).

The binary initializes the LibTorch dynamic runtime asynchronously. Weight
loading begins when the RustScript program invokes `torch::weights::load` during
inference.

## Run

Download LiquidAI LFM2 weights and tokenizer, then run:

```powershell
cargo run --release --bin rustscript-koharu-torch -- `
  --script scripts/lfm2.rss `
  --device cuda `
  models/LiquidAI/LFM2-350M-ENJP-MT/model.safetensors `
  models/LiquidAI/LFM2-350M-ENJP-MT/tokenizer.json `
  "You are a helpful translation assistant." `
  "I am building a fast inference runner with RustScript and torch operators." `
  128
```

Use `--device cuda:0` for a specific CUDA device. CPU is the default.

Floating-point weights default to `float` because this is faster for the current
LFM2 torch operator path on the tested CUDA setup. Override with
`KOHARU_TORCH_WEIGHT_KIND=native`, `half`, `bf16`, or `float`.

## Host ABI

Tensor values cross the VM boundary as opaque integer handles. The current ABI
covers every primitive used by LaMa: arithmetic, shape transforms, concatenation,
padding, activation, pooling, convolution, transposed convolution, batch
normalization, real FFT, inverse real FFT, complex construction, and pair helpers
for FFC local/global branches. `torch::runtime::arg` exposes invocation arguments
to the script; `torch::weights::load` reads safetensors onto the configured device
and caches the loaded map by path for subsequent crop invocations.
