use std::path::PathBuf;
use std::sync::Arc;

use flint_ai::{ScriptRunner, compile_script_file};

#[test]
fn cli_scripts_compile_with_host_argparse() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts");
    for name in [
        "flux_klein.rss",
        "flux_klein_encode_prompt.rss",
        "ggml_devices.rss",
        "ggml_devices_from_path.rss",
        "lfm2.rss",
        "lfm2_5.rss",
        "llama_quantized.rss",
        "xlm_roberta_ner_japanese.rss",
    ] {
        let path = root.join(name);
        compile_script_file(&path)
            .unwrap_or_else(|error| panic!("failed to compile {}: {error}", path.display()));
    }
}

#[test]
fn cli_host_parses_typed_options() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli.rss");
    let program = Arc::new(compile_script_file(path).unwrap().program);
    let output = ScriptRunner::new()
        .run_text(
            program,
            vec![
                "--name=demo".to_owned(),
                "--count".to_owned(),
                "7".to_owned(),
                "--no-mmap".to_owned(),
            ],
        )
        .unwrap();
    assert_eq!(output.text, "demo");
}

#[test]
fn cli_host_rejects_missing_required_options() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli.rss");
    let program = Arc::new(compile_script_file(path).unwrap().program);
    let error = match ScriptRunner::new().run_text(program, Vec::new()) {
        Ok(_) => panic!("missing required option was accepted"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("required argument '-n' is missing")
    );
}

#[test]
fn diffusion_namespace_is_bound_at_runtime() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/diffusion.rss");
    let program = Arc::new(compile_script_file(path).unwrap().program);
    let output = ScriptRunner::new().run_text(program, Vec::new()).unwrap();
    assert_eq!(output.text, "euler:simple");
}
