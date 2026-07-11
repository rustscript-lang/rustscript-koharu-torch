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

if "type FfiEnumRepr" not in source:
    anchor = "use crate::{Error, Result, ffi::NativeCall, sys};\n"
    replacement = anchor + """

// Clang uses an unsigned C enum representation for these non-negative values,
// while MSVC uses a signed representation. Match bindgen's target ABI.
#[cfg(target_env = "msvc")]
type FfiEnumRepr = i32;
#[cfg(not(target_env = "msvc"))]
type FfiEnumRepr = u32;
"""
    if source.count(anchor) != 1:
        raise RuntimeError("koharu-diffusion import anchor changed")
    source = source.replace(anchor, replacement)

replacements = {
    "        #[repr(i32)]\n": (
        "        #[cfg_attr(target_env = \"msvc\", repr(i32))]\n"
        "        #[cfg_attr(not(target_env = \"msvc\"), repr(u32))]\n"
    ),
    "pub const fn as_raw(self) -> i32": "pub const fn as_raw(self) -> FfiEnumRepr",
    "impl TryFrom<i32> for $name": "impl TryFrom<FfiEnumRepr> for $name",
    "fn try_from(value: i32) -> Result<Self>": (
        "fn try_from(value: FfiEnumRepr) -> Result<Self>"
    ),
    "value => Err(Error::InvalidEnum { kind: $kind, value }),": (
        "value => Err(Error::InvalidEnum { kind: $kind, value: value as i32 }),"
    ),
}

expected_counts = {
    "        #[repr(i32)]\n": 2,
    "pub const fn as_raw(self) -> i32": 2,
    "impl TryFrom<i32> for $name": 2,
    "fn try_from(value: i32) -> Result<Self>": 2,
    "value => Err(Error::InvalidEnum { kind: $kind, value }),": 2,
}

for old, new in replacements.items():
    count = source.count(old)
    if count == 0 and new in source:
        continue
    if count != expected_counts[old]:
        raise RuntimeError(f"unexpected occurrence count for {old!r}: {count}")
    source = source.replace(old, new)

path.write_text(source, encoding="utf-8")
print(f"patched {path}")
