use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use clap::{ArgAction, Parser};
use flint_ai::{LamaRustScript, TorchScriptRunner, preload_libtorch, resolve_device};
use vm::compile_source;

#[derive(Debug, Parser)]
#[command(about = "Run Flint inference programs")]
struct Cli {
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "lama")]
    llm: bool,

    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "llm")]
    lama: bool,

    #[arg(long, value_name = "DEVICE")]
    device: Option<String>,

    #[arg(long, value_name = "FILE")]
    script: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    weights: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    image: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    mask: Option<PathBuf>,

    #[arg(long, value_name = "FILE")]
    output: Option<PathBuf>,

    #[arg(value_name = "ARG", trailing_var_arg = true)]
    args: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    preload_libtorch().await?;
    let device = resolve_device(cli.device.as_deref())?;

    match (cli.llm, cli.lama) {
        (true, false) => run_llm(device, cli.script, cli.args).await,
        (false, true) => run_lama(device, cli.weights, cli.image, cli.mask, cli.output).await,
        _ => bail!("choose one mode: --llm or --lama"),
    }
}

async fn run_llm(
    device: koharu_torch::Device,
    script: Option<PathBuf>,
    args: Vec<String>,
) -> Result<()> {
    let script = required_path(script, "--script")?;
    let source = std::fs::read_to_string(&script)
        .with_context(|| format!("failed to read {}", script.display()))?;
    let compiled = compile_source(&source)
        .map_err(|err| anyhow!("failed to compile {}: {err}", script.display()))?;
    let runner = TorchScriptRunner::new(device).await?;
    let output = runner.run_text(Arc::new(compiled.program), args)?;
    if !output.text.is_empty() {
        println!("{}", output.text);
    }
    print_token_rates(&output);
    Ok(())
}

async fn run_lama(
    device: koharu_torch::Device,
    weights: Option<PathBuf>,
    image: Option<PathBuf>,
    mask: Option<PathBuf>,
    output: Option<PathBuf>,
) -> Result<()> {
    let weights = required_path(weights, "--weights")?;
    let image_path = required_path(image, "--image")?;
    let mask_path = required_path(mask, "--mask")?;
    let output_path = required_path(output, "--output")?;

    let image = image::open(&image_path)
        .with_context(|| format!("failed to read image {}", image_path.display()))?;
    let mask = image::open(&mask_path)
        .with_context(|| format!("failed to read mask {}", mask_path.display()))?
        .to_luma8();

    let model = LamaRustScript::new(device).await?;
    let result = model.inference(&weights, &image, &mask)?;
    ensure_parent_dir(&output_path)?;
    result
        .save(&output_path)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    Ok(())
}

fn required_path(value: Option<PathBuf>, name: &str) -> Result<PathBuf> {
    value.with_context(|| format!("{name} is required"))
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}

fn print_token_rates(output: &flint_ai::ScriptTextOutput) {
    if let (Some(tokens), Some(elapsed)) = (output.generated_tokens, output.elapsed) {
        let seconds = elapsed.as_secs_f64();
        if tokens > 0 && seconds > 0.0 {
            if output.decode_tokens.is_some() && output.decode_elapsed.is_some() {
                println!("tokens/s total: {:.2}", tokens as f64 / seconds);
            } else {
                println!("tokens/s: {:.2}", tokens as f64 / seconds);
            }
        }
    }
    if let (Some(tokens), Some(elapsed)) = (output.decode_tokens, output.decode_elapsed) {
        let seconds = elapsed.as_secs_f64();
        if tokens > 0 && seconds > 0.0 {
            println!("tokens/s decode: {:.2}", tokens as f64 / seconds);
        }
    }
}
