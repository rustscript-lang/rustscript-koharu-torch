use std::future::Future;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use koharu_runtime::package::loading::preload;
use libloading::Library;

pub(super) fn block_on<T>(future: impl Future<Output = Result<T>>) -> Result<T> {
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?
                .block_on(future)
        })
    } else {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(future)
    }
}

pub(super) fn library_path(directory: &Path, stem: &str) -> PathBuf {
    if cfg!(windows) {
        directory.join(format!("{stem}.dll"))
    } else if cfg!(target_os = "macos") {
        directory.join(format!("lib{stem}.dylib"))
    } else {
        directory.join(format!("lib{stem}.so"))
    }
}

pub(super) fn load_library(path: &Path) -> Result<Library> {
    if !path.exists() {
        bail!("dynamic library not found: {}", path.display());
    }
    load_library_impl(path)
}

pub(super) fn preload_directory(directory: &Path, preferred_order: &[&str]) -> Result<()> {
    if cfg!(windows) {
        for name in preferred_order {
            let path = directory.join(name);
            if path.is_file() {
                preload(path)?;
            }
        }
        return Ok(());
    }

    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        let is_library = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| matches!(extension, "so" | "dylib"));
        if is_library {
            preload(path)?;
        }
    }
    Ok(())
}

#[cfg(windows)]
fn load_library_impl(path: &Path) -> Result<Library> {
    use libloading::os::windows::{
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32, Library as WindowsLibrary,
    };

    let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    let library = unsafe { WindowsLibrary::load_with_flags(path.as_os_str(), flags) }?;
    Ok(library.into())
}

#[cfg(not(windows))]
fn load_library_impl(path: &Path) -> Result<Library> {
    Ok(unsafe { Library::new(path) }?)
}
