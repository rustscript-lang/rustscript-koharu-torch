use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use koharu_runtime::package::{
    Package, PreloadablePackage, STORE_DIR, cuda::Cuda, stable_diffusion_cpp::StableDiffusionCpp,
};
use libloading::Library;
use pd_host_function::pd_host_function;

use crate::{CallOutcome, Value, VmResult};

use super::{host_error, native, return_value};

const SD_CPP_TAG: &str = "master-769-cc73429";

pub(super) struct GgmlApi {
    _library: Library,
    ggml_backend_load_all_from_path: unsafe extern "C" fn(*const c_char),
    ggml_backend_dev_count: unsafe extern "C" fn() -> usize,
    ggml_backend_dev_get: unsafe extern "C" fn(usize) -> *mut c_void,
    ggml_backend_dev_name: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
    ggml_backend_dev_description: Option<unsafe extern "C" fn(*mut c_void) -> *const c_char>,
}

/// Loads ggml backend plugins from a directory containing ggml.dll/libggml.so.
#[pd_host_function(name = "flint::ggml::load_backends")]
pub(super) fn ggml_load_backends_impl(path: &str) -> VmResult<CallOutcome> {
    load_backends_from_path(Path::new(path))
        .map_err(|err| host_error(format!("failed to load ggml backends: {err:#}")))?;
    return_value(Value::Bool(true))
}

/// Lists ggml backend devices after loading plugins from a directory.
#[pd_host_function(name = "flint::ggml::list_devices")]
pub(super) fn ggml_list_devices_impl(path: &str) -> VmResult<CallOutcome> {
    let api = load_backends_from_path(Path::new(path))
        .map_err(|err| host_error(format!("failed to list ggml devices: {err:#}")))?;
    let devices = api
        .list_devices()
        .map_err(|err| host_error(format!("failed to list ggml devices: {err:#}")))?;
    return_value(Value::String(devices.into()))
}

/// Returns the packaged stable-diffusion.cpp runtime directory for a backend.
#[pd_host_function(name = "flint::ggml::stable_diffusion_package_dir")]
pub(super) fn ggml_stable_diffusion_package_dir_impl(backend: &str) -> VmResult<CallOutcome> {
    let package = select_stable_diffusion_package(Some(backend))
        .map_err(|err| host_error(format!("failed to select ggml package: {err:#}")))?;
    let directory = stable_diffusion_package_dir(package);
    return_value(Value::String(
        directory.to_string_lossy().into_owned().into(),
    ))
}

/// Loads ggml backend plugins from a packaged stable-diffusion.cpp runtime.
#[pd_host_function(name = "flint::ggml::load_stable_diffusion_backends")]
pub(super) fn ggml_load_stable_diffusion_backends_impl(backend: &str) -> VmResult<CallOutcome> {
    let package = select_stable_diffusion_package(Some(backend))
        .map_err(|err| host_error(format!("failed to select ggml package: {err:#}")))?;
    ensure_stable_diffusion_backends(package)
        .map_err(|err| host_error(format!("failed to load ggml backends: {err:#}")))?;
    return_value(Value::Bool(true))
}

/// Lists ggml devices for a packaged stable-diffusion.cpp runtime.
#[pd_host_function(name = "flint::ggml::list_stable_diffusion_devices")]
pub(super) fn ggml_list_stable_diffusion_devices_impl(backend: &str) -> VmResult<CallOutcome> {
    let package = select_stable_diffusion_package(Some(backend))
        .map_err(|err| host_error(format!("failed to select ggml package: {err:#}")))?;
    let devices = list_stable_diffusion_devices(package)
        .map_err(|err| host_error(format!("failed to list ggml devices: {err:#}")))?;
    return_value(Value::String(devices.into()))
}

pub(super) fn ensure_stable_diffusion_backends(
    package: StableDiffusionCpp,
) -> Result<Arc<GgmlApi>> {
    preload_package_dependencies(package)?;
    native::block_on(package.resolve())?;
    load_backends_from_path(&stable_diffusion_package_dir(package))
}

pub(super) fn select_stable_diffusion_package(backend: Option<&str>) -> Result<StableDiffusionCpp> {
    let Some(backend) = backend else {
        return StableDiffusionCpp::for_current_target();
    };
    let backend = backend.to_ascii_lowercase();
    if backend.is_empty() || backend == "auto" {
        StableDiffusionCpp::for_current_target()
    } else if backend == "cpu" {
        stable_diffusion_cpu_package()
    } else if backend.starts_with("cuda") {
        stable_diffusion_cuda_package()
    } else if backend.starts_with("vulkan") {
        stable_diffusion_vulkan_package()
    } else {
        StableDiffusionCpp::for_current_target()
    }
}

pub(super) fn stable_diffusion_package_dir(package: StableDiffusionCpp) -> PathBuf {
    STORE_DIR
        .join("stable-diffusion.cpp")
        .join(SD_CPP_TAG)
        .join(package.to_string())
}

impl GgmlApi {
    fn load(directory: &Path) -> Result<Self> {
        let library_path = ggml_library_path(directory)?;
        let library = native::load_library(&library_path)
            .with_context(|| format!("failed to load {}", library_path.display()))?;
        unsafe {
            Ok(Self {
                ggml_backend_load_all_from_path: *library
                    .get(b"ggml_backend_load_all_from_path\0")?,
                ggml_backend_dev_count: *library.get(b"ggml_backend_dev_count\0")?,
                ggml_backend_dev_get: *library.get(b"ggml_backend_dev_get\0")?,
                ggml_backend_dev_name: library
                    .get::<unsafe extern "C" fn(*mut c_void) -> *const c_char>(
                        b"ggml_backend_dev_name\0",
                    )
                    .ok()
                    .map(|symbol| *symbol),
                ggml_backend_dev_description: library
                    .get::<unsafe extern "C" fn(*mut c_void) -> *const c_char>(
                        b"ggml_backend_dev_description\0",
                    )
                    .ok()
                    .map(|symbol| *symbol),
                _library: library,
            })
        }
    }

    fn load_backends(&self, directory: &Path) -> Result<()> {
        let directory = path_to_c_string(directory)?;
        unsafe { (self.ggml_backend_load_all_from_path)(directory.as_ptr()) };
        Ok(())
    }

    fn list_devices(&self) -> Result<String> {
        let count = unsafe { (self.ggml_backend_dev_count)() };
        let mut devices = String::new();
        for index in 0..count {
            let device = unsafe { (self.ggml_backend_dev_get)(index) };
            if device.is_null() {
                continue;
            }
            let name = self
                .ggml_backend_dev_name
                .map(|op| c_string_lossy(unsafe { op(device) }))
                .unwrap_or_else(|| format!("device{index}"));
            let description = self
                .ggml_backend_dev_description
                .map(|op| c_string_lossy(unsafe { op(device) }))
                .unwrap_or_default();
            devices.push_str(&name);
            devices.push('\t');
            devices.push_str(&description);
            devices.push('\n');
        }
        Ok(devices)
    }
}

fn load_backends_from_path(path: &Path) -> Result<Arc<GgmlApi>> {
    let directory = normalize_ggml_dir(path)?;
    static APIS: OnceLock<Mutex<HashMap<PathBuf, Arc<GgmlApi>>>> = OnceLock::new();
    let apis = APIS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut apis = apis
        .lock()
        .map_err(|_| anyhow::anyhow!("ggml API registry lock is poisoned"))?;
    if let Some(api) = apis.get(&directory) {
        api.load_backends(&directory)?;
        return Ok(Arc::clone(api));
    }
    let api = Arc::new(GgmlApi::load(&directory)?);
    api.load_backends(&directory)?;
    apis.insert(directory, Arc::clone(&api));
    Ok(api)
}

fn list_stable_diffusion_devices(package: StableDiffusionCpp) -> Result<String> {
    ensure_stable_diffusion_backends(package)?;
    let directory = stable_diffusion_package_dir(package);
    let library_path = native::library_path(&directory, "stable-diffusion");
    let library = native::load_library(&library_path)
        .with_context(|| format!("failed to load {}", library_path.display()))?;
    let sd_list_devices = unsafe {
        *library.get::<unsafe extern "C" fn(*mut c_char, usize) -> usize>(b"sd_list_devices\0")?
    };
    let required = unsafe { sd_list_devices(std::ptr::null_mut(), 0) };
    if required == 0 {
        return Ok(String::new());
    }
    let mut buffer = vec![0u8; required + 1];
    let written = unsafe { sd_list_devices(buffer.as_mut_ptr().cast::<c_char>(), buffer.len()) };
    let len = written.min(required);
    Ok(String::from_utf8_lossy(&buffer[..len]).into_owned())
}

fn normalize_ggml_dir(path: &Path) -> Result<PathBuf> {
    let path = path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", path.display()))?;
    if path.is_dir() {
        Ok(path)
    } else {
        path.parent()
            .map(Path::to_path_buf)
            .with_context(|| format!("{} has no parent directory", path.display()))
    }
}

fn ggml_library_path(directory: &Path) -> Result<PathBuf> {
    let name = if cfg!(windows) {
        "ggml.dll"
    } else if cfg!(target_os = "macos") {
        "libggml.dylib"
    } else {
        "libggml.so"
    };
    let path = directory.join(name);
    if path.exists() {
        Ok(path)
    } else {
        bail!("ggml library not found: {}", path.display())
    }
}

fn preload_package_dependencies(package: StableDiffusionCpp) -> Result<()> {
    if matches!(package, StableDiffusionCpp::WindowsX64Cuda12) {
        native::block_on(Cuda::Runtime12.preload())?;
        native::block_on(Cuda::Cublas12.preload())?;
    }
    Ok(())
}

fn stable_diffusion_cpu_package() -> Result<StableDiffusionCpp> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok(StableDiffusionCpp::WindowsX64Cpu)
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok(StableDiffusionCpp::LinuxX64Cpu)
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Ok(StableDiffusionCpp::MacosArm64)
    } else {
        bail!("unsupported stable-diffusion.cpp CPU package for this target")
    }
}

fn stable_diffusion_cuda_package() -> Result<StableDiffusionCpp> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok(StableDiffusionCpp::WindowsX64Cuda12)
    } else {
        bail!("unsupported stable-diffusion.cpp CUDA package for this target")
    }
}

fn stable_diffusion_vulkan_package() -> Result<StableDiffusionCpp> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Ok(StableDiffusionCpp::WindowsX64Vulkan)
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Ok(StableDiffusionCpp::LinuxX64Vulkan)
    } else {
        bail!("unsupported stable-diffusion.cpp Vulkan package for this target")
    }
}

fn path_to_c_string(path: &Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .with_context(|| format!("path contains NUL bytes: {}", path.display()))
}

fn c_string_lossy(value: *const c_char) -> String {
    if value.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned()
    }
}
