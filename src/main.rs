use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use rustscript_koharu_torch::{TorchScriptRunner, parse_device};
use vm::compile_source;

#[derive(Debug, Parser)]
#[command(about = "Run a RustScript program with koharu-torch host functions")]
struct Cli {
    #[arg(long, value_name = "FILE")]
    script: PathBuf,

    #[arg(long, default_value = "cpu")]
    device: String,

    #[arg(value_name = "ARG")]
    args: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = parse_device(&cli.device)?;
    let source = std::fs::read_to_string(&cli.script)
        .with_context(|| format!("failed to read {}", cli.script.display()))?;
    let compiled = compile_source(&source)
        .map_err(|err| anyhow!("failed to compile {}: {err}", cli.script.display()))?;
    let runner = TorchScriptRunner::new(device).await?;
    let output = runner.run_text(Arc::new(compiled.program), cli.args)?;
    if !output.text.is_empty() {
        println!("{}", output.text);
    }
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
    Ok(())
}
