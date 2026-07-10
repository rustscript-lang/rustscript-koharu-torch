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
    if !output.is_empty() {
        println!("{output}");
    }
    Ok(())
}
