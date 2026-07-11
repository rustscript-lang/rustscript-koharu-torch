"""Patch koharu-diffusion's C enum wrappers for the current target ABI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def koharu_enums_path() -> Path:
    metadata = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    packages = json.loads(metadata.stdout)["packages"]
    matches = [
        package
        for package in packages
        if package["name"] == "koharu-diffusion"
        and package["source"].startswith("git+")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Git koharu-diffusion package, found {len(matches)}"
        )
    return Path(matches[0]["manifest_path"]).parent / "src" / "enums.rs"


def koharu_libtorch_path() -> Path:
    metadata = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    packages = json.loads(metadata.stdout)["packages"]
    matches = [
        package
        for package in packages
        if package["name"] == "koharu-runtime"
        and package["source"].startswith("git+")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Git koharu-runtime package, found {len(matches)}"
        )
    return Path(matches[0]["manifest_path"]).parent / "src" / "package" / "libtorch.rs"


def koharu_llama_build_path() -> Path:
    metadata = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    packages = json.loads(metadata.stdout)["packages"]
    matches = [
        package
        for package in packages
        if package["name"] == "koharu-llama-sys"
        and package["source"].startswith("git+")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one Git koharu-llama-sys package, found {len(matches)}"
        )
    return Path(matches[0]["manifest_path"]).parent / "build.rs"


def patch_libtorch_selection() -> None:
    libtorch_path = koharu_libtorch_path()
    libtorch_source = libtorch_path.read_text(encoding="utf-8")
    libtorch_marker = "// Patched by Flint: allow release CI to select a LibTorch device."
    if libtorch_marker in libtorch_source:
        print(f"already patched {libtorch_path}")
        return

    libtorch_old = "    pub fn for_current_target() -> Result<Self> {\n"
    libtorch_new = libtorch_old + f"""        {libtorch_marker}
        if let Some(device) = std::env::var_os("FLINT_LIBTORCH_DEVICE") {{
            match device.to_str() {{
                Some("cpu") => return Ok(Self::Cpu),
                Some("cu126") => return Ok(Self::Cuda126),
                Some("cu129") => return Ok(Self::Cuda129),
                Some("cu130") => return Ok(Self::Cuda130),
                Some(device) => bail!("unsupported FLINT_LIBTORCH_DEVICE '{{device}}'"),
                None => bail!("FLINT_LIBTORCH_DEVICE is not valid UTF-8"),
            }}
        }}
"""
    if libtorch_source.count(libtorch_old) != 1:
        raise RuntimeError("could not locate Koharu LibTorch target selection")
    libtorch_path.write_text(
        libtorch_source.replace(libtorch_old, libtorch_new), encoding="utf-8"
    )
    print(f"patched {libtorch_path}")


def patch_llama_windows_cmake_paths() -> None:
    build_path = koharu_llama_build_path()
    build_source = build_path.read_text(encoding="utf-8")
    marker = "// Patched by Flint: normalize CMake paths on Windows."
    if marker in build_source:
        print(f"already patched {build_path}")
        return

    old = """fn build_shim(manifest_dir: &Path) -> Result<()> {
    let target_dir = output_dir()?;
    fs::create_dir_all(&target_dir)?;

    let cmake_dir = cmake::Config::new(\"shim\")
        .define(
            \"KOHARU_LLAMA_COMMON_SHIM_SOURCE\",
            manifest_dir.join(\"shim/common.cpp\"),
        )
        .define(
            \"KOHARU_LLAMA_JSON_SCHEMA_SOURCE\",
            manifest_dir.join(\"common/json-schema-to-grammar.cpp\"),
        )
        .define(
            \"KOHARU_LLAMA_COMMON_SUPPORT_SOURCE\",
            manifest_dir.join(\"common/common_support.cpp\"),
        )
        .define(\"KOHARU_LLAMA_ROOT_DIR\", manifest_dir)
        .define(\"KOHARU_LLAMA_INCLUDE_DIR\", manifest_dir.join(\"include\"))
        .define(\"KOHARU_LLAMA_COMMON_DIR\", manifest_dir.join(\"common\"))
        .define(\"KOHARU_LLAMA_VENDOR_DIR\", manifest_dir.join(\"vendor\"))
        .build();
"""
    new = """// Patched by Flint: normalize CMake paths on Windows.
fn cmake_path(path: impl AsRef<Path>) -> String {
    path.as_ref().to_string_lossy().replace('\\\\', \"/\")
}

fn build_shim(manifest_dir: &Path) -> Result<()> {
    let target_dir = output_dir()?;
    fs::create_dir_all(&target_dir)?;

    let cmake_dir = cmake::Config::new(\"shim\")
        .define(
            \"KOHARU_LLAMA_COMMON_SHIM_SOURCE\",
            cmake_path(manifest_dir.join(\"shim/common.cpp\")),
        )
        .define(
            \"KOHARU_LLAMA_JSON_SCHEMA_SOURCE\",
            cmake_path(manifest_dir.join(\"common/json-schema-to-grammar.cpp\")),
        )
        .define(
            \"KOHARU_LLAMA_COMMON_SUPPORT_SOURCE\",
            cmake_path(manifest_dir.join(\"common/common_support.cpp\")),
        )
        .define(\"KOHARU_LLAMA_ROOT_DIR\", cmake_path(manifest_dir))
        .define(
            \"KOHARU_LLAMA_INCLUDE_DIR\",
            cmake_path(manifest_dir.join(\"include\")),
        )
        .define(
            \"KOHARU_LLAMA_COMMON_DIR\",
            cmake_path(manifest_dir.join(\"common\")),
        )
        .define(
            \"KOHARU_LLAMA_VENDOR_DIR\",
            cmake_path(manifest_dir.join(\"vendor\")),
        )
        .build();
"""
    if build_source.count(old) != 1:
        raise RuntimeError("could not locate Koharu llama CMake configuration")
    build_path.write_text(build_source.replace(old, new), encoding="utf-8")
    print(f"patched {build_path}")


patch_libtorch_selection()
patch_llama_windows_cmake_paths()


path = koharu_enums_path()
source = path.read_text(encoding="utf-8")

marker = "// Patched by Flint: preserve each bindgen enum's target-specific ABI."
if marker in source:
    print(f"already patched {path}")
    raise SystemExit(0)

replacements = {
    "use crate::{Error, Result, ffi::NativeCall, sys};\n": (
        "use crate::{Error, Result, ffi::NativeCall, sys};\n\n" + marker + "\n"
    ),
    "pub enum $name:ident, $kind:literal, $name_fn:path, $parse_fn:path, $invalid:path;": (
        "pub enum $name:ident, $kind:literal, $raw_ty:ty, $name_fn:path, "
        "$parse_fn:path, $invalid:path;"
    ),
    "pub enum $name:ident, $kind:literal {": (
        "pub enum $name:ident, $kind:literal, $raw_ty:ty {"
    ),
    "$($variant = $raw),+": "$($variant = $raw as i32),+",
    "pub const fn as_raw(self) -> i32": "pub const fn as_raw(self) -> $raw_ty",
    "self as i32": "self as i32 as $raw_ty",
    "impl TryFrom<i32> for $name": "impl TryFrom<$raw_ty> for $name",
    "fn try_from(value: i32) -> Result<Self>": "fn try_from(value: $raw_ty) -> Result<Self>",
    "value => Err(Error::InvalidEnum { kind: $kind, value }),": (
        "value => Err(Error::InvalidEnum { kind: $kind, value: value as i32 }),"
    ),
    'pub enum WeightType, "weight type",': (
        'pub enum WeightType, "weight type", sys::sd_type_t,'
    ),
    'pub enum RngType, "RNG type",': (
        'pub enum RngType, "RNG type", sys::rng_type_t,'
    ),
    'pub enum SampleMethod, "sample method",': (
        'pub enum SampleMethod, "sample method", sys::sample_method_t,'
    ),
    'pub enum Scheduler, "scheduler",': (
        'pub enum Scheduler, "scheduler", sys::scheduler_t,'
    ),
    'pub enum Prediction, "prediction",': (
        'pub enum Prediction, "prediction", sys::prediction_t,'
    ),
    'pub enum PreviewMode, "preview mode",': (
        'pub enum PreviewMode, "preview mode", sys::preview_t,'
    ),
    'pub enum LoraApplyMode, "LoRA apply mode",': (
        'pub enum LoraApplyMode, "LoRA apply mode", sys::lora_apply_mode_t,'
    ),
    'pub enum HiresUpscaler, "high-resolution upscaler",': (
        'pub enum HiresUpscaler, "high-resolution upscaler", '
        'sys::sd_hires_upscaler_t,'
    ),
    'pub enum VaeFormat, "VAE format" {': (
        'pub enum VaeFormat, "VAE format", sys::sd_vae_format_t {'
    ),
    'pub enum CacheMode, "cache mode" {': (
        'pub enum CacheMode, "cache mode", sys::sd_cache_mode_t {'
    ),
    'pub enum LogLevel, "log level" {': (
        'pub enum LogLevel, "log level", sys::sd_log_level_t {'
    ),
    'pub enum CancelMode, "cancel mode" {': (
        'pub enum CancelMode, "cancel mode", sys::sd_cancel_mode_t {'
    ),
}

expected_counts = {
    "use crate::{Error, Result, ffi::NativeCall, sys};\n": 1,
    "pub enum $name:ident, $kind:literal, $name_fn:path, $parse_fn:path, $invalid:path;": 1,
    "pub enum $name:ident, $kind:literal {": 1,
    "$($variant = $raw),+": 2,
    "pub const fn as_raw(self) -> i32": 2,
    "self as i32": 2,
    "impl TryFrom<i32> for $name": 2,
    "fn try_from(value: i32) -> Result<Self>": 2,
    "value => Err(Error::InvalidEnum { kind: $kind, value }),": 2,
    'pub enum WeightType, "weight type",': 1,
    'pub enum RngType, "RNG type",': 1,
    'pub enum SampleMethod, "sample method",': 1,
    'pub enum Scheduler, "scheduler",': 1,
    'pub enum Prediction, "prediction",': 1,
    'pub enum PreviewMode, "preview mode",': 1,
    'pub enum LoraApplyMode, "LoRA apply mode",': 1,
    'pub enum HiresUpscaler, "high-resolution upscaler",': 1,
    'pub enum VaeFormat, "VAE format" {': 1,
    'pub enum CacheMode, "cache mode" {': 1,
    'pub enum LogLevel, "log level" {': 1,
    'pub enum CancelMode, "cancel mode" {': 1,
}

for old, new in replacements.items():
    count = source.count(old)
    if count != expected_counts[old]:
        raise RuntimeError(f"unexpected occurrence count for {old!r}: {count}")
    source = source.replace(old, new)

path.write_text(source, encoding="utf-8")
print(f"patched {path}")
