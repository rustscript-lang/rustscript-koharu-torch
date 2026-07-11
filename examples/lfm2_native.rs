use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use koharu_torch::{Cuda, Device, Kind, Tensor};
use rustscript_koharu_torch::{parse_device, preload_libtorch};
use tokenizers::Tokenizer;

#[derive(Debug, Parser)]
#[command(about = "Run LFM2 inference directly in Rust with koharu-torch")]
struct Cli {
    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(value_name = "WEIGHTS")]
    weights: PathBuf,

    #[arg(value_name = "TOKENIZER")]
    tokenizer: PathBuf,

    #[arg(value_name = "SYSTEM")]
    system: String,

    #[arg(value_name = "TEXT")]
    text: String,

    #[arg(value_name = "MAX_NEW_TOKENS")]
    max_new_tokens: usize,

    #[arg(long)]
    profile: bool,

    #[arg(long)]
    ignore_eos: bool,
}

struct Lfm2Native {
    device: Device,
    weights: HashMap<String, Tensor>,
    k_cache: HashMap<usize, Tensor>,
    v_cache: HashMap<usize, Tensor>,
    conv_cache: HashMap<usize, Tensor>,
    rope_cos_cache: Option<Tensor>,
    rope_sin_cache: Option<Tensor>,
    profile: ProfileStats,
}

struct GenerationStats {
    text: String,
    generated_tokens: usize,
    total_elapsed: Duration,
    decode_elapsed: Option<Duration>,
}

#[derive(Default)]
struct ProfileStats {
    enabled: bool,
    entries: HashMap<&'static str, Duration>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = parse_device(&cli.device)?;
    preload_libtorch().await?;
    let tokenizer = Tokenizer::from_file(&cli.tokenizer)
        .map_err(|err| anyhow::anyhow!("failed to read {}: {err}", cli.tokenizer.display()))?;
    let mut model = Lfm2Native::load(&cli.weights, device, cli.profile)?;
    let stats = koharu_torch::no_grad(|| {
        model.generate(
            &tokenizer,
            &cli.system,
            &cli.text,
            cli.max_new_tokens,
            cli.ignore_eos,
        )
    })?;
    println!("{}", stats.text);
    println!("tokens: {}", stats.generated_tokens);
    let total_seconds = stats.total_elapsed.as_secs_f64();
    if stats.generated_tokens > 0 && total_seconds > 0.0 {
        println!(
            "tokens/s total: {:.2}",
            stats.generated_tokens as f64 / total_seconds
        );
    }
    if let Some(decode_elapsed) = stats.decode_elapsed {
        let decode_seconds = decode_elapsed.as_secs_f64();
        let decode_tokens = stats.generated_tokens.saturating_sub(1);
        if decode_tokens > 0 && decode_seconds > 0.0 {
            println!(
                "tokens/s decode: {:.2}",
                decode_tokens as f64 / decode_seconds
            );
        }
    }
    if cli.profile {
        model.print_profile();
    }
    Ok(())
}

impl Lfm2Native {
    fn load(path: &PathBuf, device: Device, profile: bool) -> Result<Self> {
        let target_kind = requested_weight_kind()?;
        let mut weights = Tensor::read_safetensors(path)
            .with_context(|| format!("failed to read {}", path.display()))?
            .into_iter()
            .map(|(name, tensor)| (name, tensor_to_model_device(tensor, device, target_kind)))
            .collect::<HashMap<_, _>>();
        add_fused_weights(&mut weights)?;
        Ok(Self {
            device,
            weights,
            k_cache: HashMap::new(),
            v_cache: HashMap::new(),
            conv_cache: HashMap::new(),
            rope_cos_cache: None,
            rope_sin_cache: None,
            profile: ProfileStats {
                enabled: profile,
                entries: HashMap::new(),
            },
        })
    }

    fn generate(
        &mut self,
        tokenizer: &Tokenizer,
        system: &str,
        text: &str,
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<GenerationStats> {
        self.k_cache.clear();
        self.v_cache.clear();
        self.conv_cache.clear();

        let prompt_ids = encode_chat(tokenizer, system, text)?;
        let prompt_len = prompt_ids.len();
        let mut all_ids = prompt_ids.clone();
        let prompt_tokens = self.tokens_tensor(&prompt_ids);
        let mut all_tokens = prompt_tokens.shallow_clone();
        sync_if_cuda(self.device);
        let total_started = Instant::now();

        let mut logits = if max_new_tokens > 0 {
            let timed = self.profile_start();
            let output = self.model(prompt_tokens, 0, false)?;
            self.profile_end("prefill", timed);
            output
        } else {
            Tensor::from_slice(&[0i64]).to_device(self.device)
        };
        sync_if_cuda(self.device);

        let mut generated_tokens = 0usize;
        let mut decode_started = None;
        if ignore_eos {
            for step in 0..max_new_tokens {
                let next_token = argmax_token(&logits);
                all_tokens = Tensor::cat(&[&all_tokens, &next_token], 1);
                generated_tokens += 1;
                if step + 1 < max_new_tokens {
                    if decode_started.is_none() {
                        sync_if_cuda(self.device);
                        decode_started = Some(Instant::now());
                    }
                    let decode_position = prompt_len + generated_tokens - 1;
                    let timed = self.profile_start();
                    logits = self.model(next_token, decode_position, true)?;
                    self.profile_end("decode_model", timed);
                }
            }
        } else {
            for step in 0..max_new_tokens {
                let next_token = argmax_int(&logits);
                all_ids.push(next_token);
                generated_tokens += 1;
                if next_token == 7 {
                    break;
                }
                if step + 1 < max_new_tokens {
                    if decode_started.is_none() {
                        sync_if_cuda(self.device);
                        decode_started = Some(Instant::now());
                    }
                    let decode_input = self.tokens_tensor(&[next_token]);
                    let decode_position = prompt_len + generated_tokens - 1;
                    let timed = self.profile_start();
                    logits = self.model(decode_input, decode_position, true)?;
                    self.profile_end("decode_model", timed);
                }
            }
        }
        sync_if_cuda(self.device);
        let total_elapsed = total_started.elapsed();
        let decode_elapsed = decode_started.map(|started| started.elapsed());
        let generated_ids = if ignore_eos {
            token_ids_from_tensor(&all_tokens)?
        } else {
            all_ids
        };
        let generated = generated_ids
            .into_iter()
            .skip(prompt_len)
            .filter_map(|id| u32::try_from(id).ok())
            .collect::<Vec<_>>();
        let text = tokenizer
            .decode(&generated, true)
            .map_err(|err| anyhow::anyhow!("tokenizer decode failed: {err}"))?;
        Ok(GenerationStats {
            text,
            generated_tokens,
            total_elapsed,
            decode_elapsed,
        })
    }

    fn model(&mut self, input_ids: Tensor, position: usize, incremental: bool) -> Result<Tensor> {
        let timed = self.profile_start();
        let embeddings = Tensor::embedding(
            self.weight("model.embed_tokens.weight")?,
            &input_ids,
            -1,
            false,
            false,
        );
        self.profile_end("embedding", timed);
        let mut hidden = embeddings;
        hidden = self.timed_conv_layer(hidden, 0, incremental)?;
        hidden = self.timed_conv_layer(hidden, 1, incremental)?;
        hidden = self.timed_attention_layer(hidden, 2, position, incremental)?;
        hidden = self.timed_conv_layer(hidden, 3, incremental)?;
        hidden = self.timed_conv_layer(hidden, 4, incremental)?;
        hidden = self.timed_attention_layer(hidden, 5, position, incremental)?;
        hidden = self.timed_conv_layer(hidden, 6, incremental)?;
        hidden = self.timed_conv_layer(hidden, 7, incremental)?;
        hidden = self.timed_attention_layer(hidden, 8, position, incremental)?;
        hidden = self.timed_conv_layer(hidden, 9, incremental)?;
        hidden = self.timed_attention_layer(hidden, 10, position, incremental)?;
        hidden = self.timed_conv_layer(hidden, 11, incremental)?;
        hidden = self.timed_attention_layer(hidden, 12, position, incremental)?;
        hidden = self.timed_conv_layer(hidden, 13, incremental)?;
        hidden = self.timed_attention_layer(hidden, 14, position, incremental)?;
        hidden = self.timed_conv_layer(hidden, 15, incremental)?;

        let timed = self.profile_start();
        let normalized = self.rms_norm(hidden, "model.embedding_norm.weight")?;
        let seq_len = normalized.size()[1];
        let last_hidden = normalized.select(1, seq_len - 1);
        let lm_head = self.weight_or("lm_head.weight", "model.embed_tokens.weight")?;
        let output = last_hidden.linear(lm_head, None::<&Tensor>);
        self.profile_end("final_head", timed);
        Ok(output)
    }

    fn timed_conv_layer(
        &mut self,
        input: Tensor,
        layer: usize,
        incremental: bool,
    ) -> Result<Tensor> {
        let timed = self.profile_start();
        let output = self.conv_layer(input, layer, incremental)?;
        self.profile_end("conv_layer", timed);
        Ok(output)
    }

    fn timed_attention_layer(
        &mut self,
        input: Tensor,
        layer: usize,
        position: usize,
        incremental: bool,
    ) -> Result<Tensor> {
        let timed = self.profile_start();
        let output = self.attention_layer(input, layer, position, incremental)?;
        self.profile_end("attention_layer", timed);
        Ok(output)
    }

    fn conv_layer(&mut self, input: Tensor, layer: usize, incremental: bool) -> Result<Tensor> {
        let timed = self.profile_start();
        let mixed = self.conv_mixer(&input, layer, incremental)?;
        self.profile_end("conv_mixer", timed);
        let timed = self.profile_start();
        let output = self.finish_layer(input, layer, mixed)?;
        self.profile_end("conv_finish", timed);
        Ok(output)
    }

    fn attention_layer(
        &mut self,
        input: Tensor,
        layer: usize,
        position: usize,
        incremental: bool,
    ) -> Result<Tensor> {
        let timed = self.profile_start();
        let mixed = self.attention_mixer(&input, layer, position, incremental)?;
        self.profile_end("attention_mixer", timed);
        let timed = self.profile_start();
        let output = self.finish_layer(input, layer, mixed)?;
        self.profile_end("attention_finish", timed);
        Ok(output)
    }

    fn attention_mixer(
        &mut self,
        input: &Tensor,
        layer: usize,
        position: usize,
        incremental: bool,
    ) -> Result<Tensor> {
        let size = input.size();
        let batch = size[0];
        let seq_len = size[1];
        let hidden = self.rms_norm(
            input.shallow_clone(),
            &layer_name(layer, "operator_norm.weight"),
        )?;

        let qkv_linear = self.linear(hidden, &layer_name(layer, "self_attn.qkv_proj.weight"))?;
        let q_linear = qkv_linear.narrow(-1, 0, 1024);
        let k_linear = qkv_linear.narrow(-1, 1024, 512);
        let v_linear = qkv_linear.narrow(-1, 1536, 512);

        let q_view = q_linear.view([batch, seq_len, 16, 64]);
        let k_view = k_linear.view([batch, seq_len, 8, 64]);
        let v_view = v_linear.view([batch, seq_len, 8, 64]);
        let q_norm = self.rms_norm(q_view, &layer_name(layer, "self_attn.q_layernorm.weight"))?;
        let k_norm = self.rms_norm(k_view, &layer_name(layer, "self_attn.k_layernorm.weight"))?;

        let q_heads = q_norm.transpose(1, 2);
        let k_heads = k_norm.transpose(1, 2);
        let v_heads = v_view.transpose(1, 2);
        let cos = self.rope_slice(seq_len, position, RopeKind::Cos);
        let sin = self.rope_slice(seq_len, position, RopeKind::Sin);
        let q_rot = apply_rope(q_heads, &cos, &sin);
        let k_rot = apply_rope(k_heads, &cos, &sin);

        let (k_all, v_all) = if incremental {
            let old_k = self
                .k_cache
                .get(&layer)
                .with_context(|| format!("missing k cache for layer {layer}"))?;
            let old_v = self
                .v_cache
                .get(&layer)
                .with_context(|| format!("missing v cache for layer {layer}"))?;
            (
                Tensor::cat(&[old_k, &k_rot], 2),
                Tensor::cat(&[old_v, &v_heads], 2),
            )
        } else {
            (k_rot, v_heads)
        };
        self.k_cache.insert(layer, k_all.shallow_clone());
        self.v_cache.insert(layer, v_all.shallow_clone());

        let k_full = k_all.repeat_interleave_self_int(2, 1, None);
        let v_full = v_all.repeat_interleave_self_int(2, 1, None);
        let attended = Tensor::scaled_dot_product_attention(
            &q_rot,
            &k_full,
            &v_full,
            None::<&Tensor>,
            0.0,
            !incremental,
            None,
            false,
        );
        let merged = attended
            .transpose(1, 2)
            .contiguous()
            .view([batch, seq_len, 1024]);
        self.linear(merged, &layer_name(layer, "self_attn.out_proj.weight"))
    }

    fn conv_mixer(&mut self, input: &Tensor, layer: usize, incremental: bool) -> Result<Tensor> {
        let seq_len = input.size()[1];
        let hidden = self.rms_norm(
            input.shallow_clone(),
            &layer_name(layer, "operator_norm.weight"),
        )?;
        let projected = self
            .linear(hidden, &layer_name(layer, "conv.in_proj.weight"))?
            .transpose(-1, -2);
        let b_part = projected.narrow(1, 0, 1024);
        let c_part = projected.narrow(1, 1024, 1024);
        let x_part = projected.narrow(1, 2048, 1024);
        let bx = b_part * x_part;
        let (conv_input, conv_start) = if incremental {
            let old_state = self
                .conv_cache
                .get(&layer)
                .with_context(|| format!("missing conv cache for layer {layer}"))?;
            let start = old_state.size()[2];
            (Tensor::cat(&[old_state, &bx], 2), start)
        } else {
            (bx, 0)
        };
        let weight = self.weight(&layer_name(layer, "conv.conv.weight"))?;
        let conv_full = depthwise_conv1d(&conv_input, weight, 2);
        let state = tail(&conv_input, 2, 2);
        self.conv_cache.insert(layer, state);
        let conv_trimmed = if incremental {
            conv_full.narrow(-1, conv_start, 1)
        } else {
            conv_full.narrow(-1, 0, seq_len)
        };
        let mixed = (c_part * conv_trimmed).transpose(-1, -2).contiguous();
        self.linear(mixed, &layer_name(layer, "conv.out_proj.weight"))
    }

    fn finish_layer(&self, input: Tensor, layer: usize, mixed: Tensor) -> Result<Tensor> {
        let hidden = input + mixed;
        let ffn_input = self.rms_norm(
            hidden.shallow_clone(),
            &layer_name(layer, "ffn_norm.weight"),
        )?;
        let feed_forward = self.mlp(ffn_input, layer)?;
        Ok(hidden + feed_forward)
    }

    fn mlp(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        let gate_up = self.linear(input, &layer_name(layer, "feed_forward.w1_w3.weight"))?;
        self.linear(
            swiglu(&gate_up),
            &layer_name(layer, "feed_forward.w2.weight"),
        )
    }

    fn rms_norm(&self, input: Tensor, weight_name: &str) -> Result<Tensor> {
        let scale_weight = self.weight(weight_name)?;
        Ok(input
            .internal_fused_rms_norm(scale_weight.size(), Some(scale_weight), 0.00001)
            .0)
    }

    fn linear(&self, input: Tensor, weight_name: &str) -> Result<Tensor> {
        let weight = self.weight(weight_name)?;
        Ok(if linear_mv_enabled() {
            linear_or_mv(&input, weight)
        } else {
            input.linear(weight, None::<&Tensor>)
        })
    }

    fn weight(&self, name: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .with_context(|| format!("missing weight '{name}'"))
    }

    fn weight_or(&self, name: &str, fallback: &str) -> Result<&Tensor> {
        self.weights
            .get(name)
            .or_else(|| self.weights.get(fallback))
            .with_context(|| format!("missing weight '{name}' and fallback '{fallback}'"))
    }

    fn tokens_tensor(&self, ids: &[i64]) -> Tensor {
        Tensor::from_slice(ids)
            .view([1, ids.len() as i64])
            .to_device(self.device)
    }

    fn rope_slice(&mut self, seq_len: i64, start: usize, kind: RopeKind) -> Tensor {
        let end = start as i64 + seq_len;
        let device = self.device;
        let cache = match kind {
            RopeKind::Cos => &mut self.rope_cos_cache,
            RopeKind::Sin => &mut self.rope_sin_cache,
        };
        let current_len = cache.as_ref().map(|tensor| tensor.size()[2]).unwrap_or(0);
        if current_len < end {
            let target_len = end.max(current_len.saturating_mul(2)).max(128);
            *cache = Some(build_rope_table(target_len, 64, 1_000_000.0, device, kind));
        }
        cache
            .as_ref()
            .expect("rope cache should be initialized")
            .narrow(2, start as i64, seq_len)
    }

    fn profile_start(&self) -> Option<Instant> {
        if self.profile.enabled {
            sync_if_cuda(self.device);
            Some(Instant::now())
        } else {
            None
        }
    }

    fn profile_end(&mut self, label: &'static str, started: Option<Instant>) {
        if let Some(started) = started {
            sync_if_cuda(self.device);
            *self.profile.entries.entry(label).or_default() += started.elapsed();
        }
    }

    fn print_profile(&self) {
        let mut entries = self.profile.entries.iter().collect::<Vec<_>>();
        entries.sort_by_key(|(_, duration)| std::cmp::Reverse(duration.as_nanos()));
        println!("profile:");
        for (label, duration) in entries {
            println!("  {label}: {:.3} ms", duration.as_secs_f64() * 1000.0);
        }
    }
}

#[derive(Clone, Copy)]
enum RopeKind {
    Cos,
    Sin,
}

fn encode_chat(tokenizer: &Tokenizer, system: &str, user: &str) -> Result<Vec<i64>> {
    let prompt = format!(
        "<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    );
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|err| anyhow::anyhow!("tokenizer encode failed: {err}"))?;
    Ok(encoding.get_ids().iter().map(|id| i64::from(*id)).collect())
}

fn add_fused_weights(weights: &mut HashMap<String, Tensor>) -> Result<()> {
    let mut fused = Vec::new();
    for layer in [2usize, 5, 8, 10, 12, 14] {
        let q = weights
            .get(&layer_name(layer, "self_attn.q_proj.weight"))
            .with_context(|| format!("missing q projection for layer {layer}"))?;
        let k = weights
            .get(&layer_name(layer, "self_attn.k_proj.weight"))
            .with_context(|| format!("missing k projection for layer {layer}"))?;
        let v = weights
            .get(&layer_name(layer, "self_attn.v_proj.weight"))
            .with_context(|| format!("missing v projection for layer {layer}"))?;
        fused.push((
            layer_name(layer, "self_attn.qkv_proj.weight"),
            Tensor::cat(&[q, k, v], 0),
        ));
    }
    for layer in 0..16 {
        let w1 = weights
            .get(&layer_name(layer, "feed_forward.w1.weight"))
            .with_context(|| format!("missing feed-forward w1 for layer {layer}"))?;
        let w3 = weights
            .get(&layer_name(layer, "feed_forward.w3.weight"))
            .with_context(|| format!("missing feed-forward w3 for layer {layer}"))?;
        fused.push((
            layer_name(layer, "feed_forward.w1_w3.weight"),
            Tensor::cat(&[w1, w3], 0),
        ));
    }
    weights.extend(fused);
    Ok(())
}

fn layer_name(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

fn rotate_half(input: &Tensor) -> Tensor {
    let head_dim = input.size()[input.size().len() - 1];
    let half = head_dim / 2;
    let first = input.narrow(-1, 0, half);
    let second = input.narrow(-1, half, half);
    Tensor::cat(&[&second.neg(), &first], -1)
}

fn apply_rope(input: Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let rotated = rotate_half(&input);
    input * cos + rotated * sin
}

fn swiglu(input: &Tensor) -> Tensor {
    let last_dim = input.size()[input.size().len() - 1];
    let intermediate = last_dim / 2;
    let gate = input.narrow(-1, 0, intermediate).silu();
    let up = input.narrow(-1, intermediate, intermediate);
    gate * up
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
            .is_some_and(|value| value != "0" && !value.is_empty())
    })
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

fn requested_weight_kind() -> Result<Option<Kind>> {
    let Some(value) = std::env::var_os("KOHARU_TORCH_WEIGHT_KIND") else {
        return Ok(None);
    };
    let value = value.to_string_lossy().to_ascii_lowercase();
    match value.as_str() {
        "" | "native" | "auto" => Ok(None),
        "half" | "fp16" | "f16" => Ok(Some(Kind::Half)),
        "bf16" | "bfloat16" => Ok(Some(Kind::BFloat16)),
        "float" | "fp32" | "f32" => Ok(Some(Kind::Float)),
        other => anyhow::bail!(
            "KOHARU_TORCH_WEIGHT_KIND must be native, half, bf16, or float, got '{other}'"
        ),
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

fn depthwise_conv1d(input: &Tensor, weight: &Tensor, padding: i64) -> Tensor {
    let output_kind = input.kind();
    let input_float = input.to_kind(Kind::Float);
    let weight_float = weight.to_kind(Kind::Float);
    let groups = input_float.size()[1];
    if let Ok(output) =
        input_float.f_conv1d(&weight_float, None::<&Tensor>, [1], [padding], [1], groups)
    {
        return output.to_kind(output_kind);
    }
    let out_len = input_float.size()[2] + (2 * padding) - 2;
    let padded = input_float.zero_pad1d(padding, padding);
    let w0 = weight_float.select(2, 0).view([1, groups, 1]);
    let w1 = weight_float.select(2, 1).view([1, groups, 1]);
    let w2 = weight_float.select(2, 2).view([1, groups, 1]);
    (padded.narrow(2, 0, out_len) * w0
        + padded.narrow(2, 1, out_len) * w1
        + padded.narrow(2, 2, out_len) * w2)
        .to_kind(output_kind)
}

fn tail(input: &Tensor, raw_dim: i64, len: i64) -> Tensor {
    let rank = input.size().len() as i64;
    let dim = if raw_dim < 0 { rank + raw_dim } else { raw_dim };
    let dim_size = input.size()[dim as usize];
    let actual_len = len.min(dim_size);
    input.narrow(dim, dim_size - actual_len, actual_len)
}

fn argmax_int(input: &Tensor) -> i64 {
    let output = input.argmax(-1, false).to_device(Device::Cpu).view([-1]);
    output.int64_value(&[0])
}

fn argmax_token(input: &Tensor) -> Tensor {
    input.argmax(-1, false).view([1, 1])
}

fn token_ids_from_tensor(input: &Tensor) -> Result<Vec<i64>> {
    let tokens = input.to_device(Device::Cpu).to_kind(Kind::Int64).view([-1]);
    Vec::<i64>::try_from(&tokens).context("failed to copy token ids")
}

fn sync_if_cuda(device: Device) {
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}
