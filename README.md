# rustscript-koharu-torch

Standalone integration between RustScript and `koharu-torch`. The Torch crate is
resolved directly from `mayocream/koharu` at branch `refactor/0705`; this project
does not vendor or modify the Koharu repository.

The first end-to-end graph is LaMa manga inpainting. Rust owns image I/O, crop
orchestration, device selection, and opaque tensor handles. RustScript receives
the safetensors path as a runtime argument and loads it through the
`torch::weights::load` host function. The FFC generator, 18 residual blocks,
Fourier units, and output composition are implemented in
[`scripts/lama.rss`](scripts/lama.rss).

`LamaRustScript::new` initializes the LibTorch dynamic runtime asynchronously;
it does not read model weights. Weight loading begins only when the RustScript
program invokes `torch::weights::load` during inference.

## Run

Download `lama-manga.safetensors` from
<https://huggingface.co/mayocream/lama-manga> and run:

```powershell
cargo run --release --bin lama-rustscript -- `
  --weights models/lama-manga.safetensors `
  --image path/to/image.png `
  --mask path/to/mask.png `
  --output runs/inpainted.png
```

Use `--device cuda:0` for CUDA. CPU is the default.

## Host ABI

Tensor values cross the VM boundary as opaque integer handles. The current ABI
covers every primitive used by LaMa: arithmetic, shape transforms, concatenation,
padding, activation, pooling, convolution, transposed convolution, batch
normalization, real FFT, inverse real FFT, complex construction, and pair helpers
for FFC local/global branches. `torch::runtime::arg` exposes invocation arguments
to the script; `torch::weights::load` reads safetensors onto the configured device
and caches the loaded map by path for subsequent crop invocations.
