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
