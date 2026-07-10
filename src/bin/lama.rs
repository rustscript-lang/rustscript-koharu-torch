use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use rustscript_koharu_torch::{LamaRustScript, parse_device};

#[derive(Debug, Parser)]
#[command(about = "Run LaMa inpainting through a RustScript inference graph")]
struct Cli {
    #[arg(long, value_name = "FILE")]
    weights: PathBuf,

    #[arg(long, value_name = "FILE")]
    image: PathBuf,

    #[arg(long, value_name = "FILE")]
    mask: PathBuf,

    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    #[arg(long, default_value = "cpu")]
    device: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let device = parse_device(&cli.device)?;
    let image = image::open(&cli.image)
        .with_context(|| format!("failed to open {}", cli.image.display()))?;
    let mask = image::open(&cli.mask)
        .with_context(|| format!("failed to open {}", cli.mask.display()))?
        .to_luma8();
    let lama = LamaRustScript::new(device).await?;
    let output = lama.inference(&cli.weights, &image, &mask)?;
    output
        .save(&cli.output)
        .with_context(|| format!("failed to save {}", cli.output.display()))?;
    Ok(())
}
