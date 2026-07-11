use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{LazyLock, Mutex};

use anyhow::{Context as _, Result};
use koharu_diffusion::{
    Context, ContextParams, ImageGenerationParams, RgbImage, SampleMethod, Scheduler, VaeFormat,
    WeightType,
};
use koharu_runtime::package::stable_diffusion_cpp::StableDiffusionCpp;
use libloading::Library;
use pd_host_function::pd_host_function;

use crate::{CallOutcome, Value, VmResult};

use super::{ggml, host_error, return_int, return_value};

#[cfg(target_env = "msvc")]
type SamplerEnumRepr = i32;
#[cfg(not(target_env = "msvc"))]
type SamplerEnumRepr = u32;

static NEXT_SD_HANDLE: AtomicI64 = AtomicI64::new(1);
static SD_CTX_PARAMS_HANDLES: LazyLock<Mutex<HashMap<i64, PendingContextParams>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SD_CTX_HANDLES: LazyLock<Mutex<HashMap<i64, Context>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SD_IMG_PARAMS_HANDLES: LazyLock<Mutex<HashMap<i64, ImageGenerationParams>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SD_IMAGES_HANDLES: LazyLock<Mutex<HashMap<i64, Vec<RgbImage>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static SD_NATIVE_RUNTIME: LazyLock<Mutex<Option<NativeRuntime>>> =
    LazyLock::new(|| Mutex::new(None));

#[derive(Default)]
struct PendingContextParams {
    model_path: Option<PathBuf>,
    diffusion_model_path: Option<PathBuf>,
    vae_path: Option<PathBuf>,
    llm_path: Option<PathBuf>,
    package_backend: Option<String>,
    backend: Option<String>,
    params_backend: Option<String>,
    max_vram: Option<String>,
    weight_type: WeightType,
    vae_format: VaeFormat,
    enable_mmap: bool,
    flash_attention: bool,
    diffusion_flash_attention: bool,
}

struct NativeRuntime {
    package: StableDiffusionCpp,
    _library: Library,
}

fn next_sd_handle() -> i64 {
    NEXT_SD_HANDLE.fetch_add(1, Ordering::Relaxed)
}

/// Creates default stable-diffusion.cpp context parameters and returns a handle.
#[pd_host_function(name = "flint::sd::ctx_params_init")]
pub(super) fn sd_ctx_params_init_impl() -> VmResult<CallOutcome> {
    let handle = next_sd_handle();
    SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?
        .insert(handle, PendingContextParams::default());
    return_int(handle)
}

/// Sets model-related paths on stable-diffusion.cpp context parameters.
#[pd_host_function(name = "flint::sd::ctx_params_set_paths")]
pub(super) fn sd_ctx_params_set_paths_impl(
    handle: i64,
    model_path: &str,
    diffusion_model_path: &str,
    vae_path: &str,
    llm_path: &str,
) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("context parameter", handle))?;
    params.model_path = optional_path(model_path);
    params.diffusion_model_path = optional_path(diffusion_model_path);
    params.vae_path = optional_path(vae_path);
    params.llm_path = optional_path(llm_path);
    return_value(Value::Bool(true))
}

/// Sets backend placement options on stable-diffusion.cpp context parameters.
#[pd_host_function(name = "flint::sd::ctx_params_set_backend")]
pub(super) fn sd_ctx_params_set_backend_impl(
    handle: i64,
    backend: &str,
    params_backend: &str,
    max_vram: &str,
) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("context parameter", handle))?;
    params.package_backend = optional_string(backend);
    params.backend = native_backend(backend);
    params.params_backend = optional_string(params_backend);
    params.max_vram = max_vram_value(max_vram);
    return_value(Value::Bool(true))
}

/// Sets the model weight type on stable-diffusion.cpp context parameters.
#[pd_host_function(name = "flint::sd::ctx_params_set_wtype")]
pub(super) fn sd_ctx_params_set_wtype_impl(handle: i64, wtype: &str) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("context parameter", handle))?;
    params.weight_type = parse_weight_type(wtype)?;
    return_value(Value::Bool(true))
}

/// Sets the VAE tensor naming/layout format.
#[pd_host_function(name = "flint::sd::ctx_params_set_vae_format")]
pub(super) fn sd_ctx_params_set_vae_format_impl(
    handle: i64,
    vae_format: i64,
) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("context parameter", handle))?;
    params.vae_format =
        VaeFormat::try_from(checked_i32(vae_format, "vae_format")?).map_err(diffusion_error)?;
    return_value(Value::Bool(true))
}

/// Sets mmap and attention flags on stable-diffusion.cpp context parameters.
#[pd_host_function(name = "flint::sd::ctx_params_set_flags")]
pub(super) fn sd_ctx_params_set_flags_impl(
    handle: i64,
    enable_mmap: bool,
    flash_attn: bool,
    diffusion_flash_attn: bool,
) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("context parameter", handle))?;
    params.enable_mmap = enable_mmap;
    params.flash_attention = flash_attn;
    params.diffusion_flash_attention = diffusion_flash_attn;
    return_value(Value::Bool(true))
}

/// Creates a stable-diffusion.cpp context through koharu-diffusion.
#[pd_host_function(name = "flint::sd::new_sd_ctx")]
pub(super) fn sd_new_sd_ctx_impl(params_handle: i64) -> VmResult<CallOutcome> {
    let pending = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("context parameter"))?
        .remove(&params_handle)
        .ok_or_else(|| unknown_handle("context parameter", params_handle))?;
    let package = ggml::select_stable_diffusion_package(pending.package_backend.as_deref())
        .map_err(|err| {
            host_error(format!(
                "failed to select stable diffusion backend: {err:#}"
            ))
        })?;
    prepare_native(package)?;

    let devices = koharu_diffusion::list_devices();
    if devices.is_empty() {
        return Err(host_error(
            "stable-diffusion.cpp reported no ggml backend devices",
        ));
    }
    let device_list = devices
        .iter()
        .map(|device| format!("{}\t{}", device.name, device.description))
        .collect::<Vec<_>>()
        .join("\n");
    eprintln!("stable-diffusion.cpp devices:\n{device_list}");

    let params = ContextParams {
        model_path: pending.model_path,
        diffusion_model_path: pending.diffusion_model_path,
        vae_path: pending.vae_path,
        llm_path: pending.llm_path,
        backend: pending.backend,
        params_backend: pending.params_backend,
        max_vram: pending.max_vram,
        weight_type: pending.weight_type,
        vae_format: pending.vae_format,
        enable_mmap: pending.enable_mmap,
        flash_attention: pending.flash_attention,
        diffusion_flash_attention: pending.diffusion_flash_attention,
        ..ContextParams::default()
    };
    let context = Context::new(&params).map_err(diffusion_error)?;
    let handle = next_sd_handle();
    SD_CTX_HANDLES
        .lock()
        .map_err(|_| registry_error("context"))?
        .insert(handle, context);
    return_int(handle)
}

/// Drops an owning koharu-diffusion context handle.
#[pd_host_function(name = "flint::sd::free_sd_ctx")]
pub(super) fn sd_free_sd_ctx_impl(ctx_handle: i64) -> VmResult<CallOutcome> {
    SD_CTX_HANDLES
        .lock()
        .map_err(|_| registry_error("context"))?
        .remove(&ctx_handle)
        .ok_or_else(|| unknown_handle("context", ctx_handle))?;
    return_value(Value::Bool(true))
}

/// Creates default image-generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_init")]
pub(super) fn sd_img_gen_params_init_impl() -> VmResult<CallOutcome> {
    let handle = next_sd_handle();
    SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("image parameter"))?
        .insert(handle, ImageGenerationParams::default());
    return_int(handle)
}

/// Sets prompt strings on image-generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_set_prompt")]
pub(super) fn sd_img_gen_params_set_prompt_impl(
    handle: i64,
    prompt: &str,
    negative_prompt: &str,
) -> VmResult<CallOutcome> {
    let mut handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("image parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("image parameter", handle))?;
    params.prompt = prompt.to_owned();
    params.negative_prompt = negative_prompt.to_owned();
    return_value(Value::Bool(true))
}

/// Sets output dimensions on image-generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_set_size")]
pub(super) fn sd_img_gen_params_set_size_impl(
    handle: i64,
    width: i64,
    height: i64,
) -> VmResult<CallOutcome> {
    validate_positive(width, "width")?;
    validate_positive(height, "height")?;
    let mut handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("image parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("image parameter", handle))?;
    params.width = checked_i32(width, "width")?;
    params.height = checked_i32(height, "height")?;
    return_value(Value::Bool(true))
}

/// Sets sampling options on image-generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_set_sample")]
pub(super) fn sd_img_gen_params_set_sample_impl(
    handle: i64,
    steps: i64,
    seed: i64,
    cfg_scale: f64,
) -> VmResult<CallOutcome> {
    validate_positive(steps, "steps")?;
    if cfg_scale < 0.0 {
        return Err(host_error("cfg_scale must be non-negative"));
    }
    let mut handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("image parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("image parameter", handle))?;
    params.seed = seed;
    params.batch_count = 1;
    params.sample.sample_steps = checked_i32(steps, "steps")?;
    params.sample.guidance.text_cfg = cfg_scale as f32;
    return_value(Value::Bool(true))
}

/// Sets sample method and scheduler on image-generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_set_sampler")]
pub(super) fn sd_img_gen_params_set_sampler_impl(
    handle: i64,
    sample_method: i64,
    scheduler: i64,
) -> VmResult<CallOutcome> {
    let sample_method =
        SampleMethod::try_from(checked_i32(sample_method, "sample_method")? as SamplerEnumRepr)
            .map_err(diffusion_error)?;
    let scheduler = Scheduler::try_from(checked_i32(scheduler, "scheduler")? as SamplerEnumRepr)
        .map_err(diffusion_error)?;
    let mut handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("image parameter"))?;
    let params = handles
        .get_mut(&handle)
        .ok_or_else(|| unknown_handle("image parameter", handle))?;
    params.sample.sample_method = sample_method;
    params.sample.scheduler = scheduler;
    return_value(Value::Bool(true))
}

/// Converts a sample method name to its enum value.
#[pd_host_function(name = "flint::sd::str_to_sample_method")]
pub(super) fn sd_str_to_sample_method_impl(name: &str) -> VmResult<CallOutcome> {
    let name = normalize_auto(name);
    let value = SampleMethod::from_str(name).map_err(diffusion_error)?;
    return_int(i64::from(value.as_raw()))
}

/// Converts a scheduler name to its enum value.
#[pd_host_function(name = "flint::sd::str_to_scheduler")]
pub(super) fn sd_str_to_scheduler_impl(name: &str) -> VmResult<CallOutcome> {
    let name = normalize_auto(name);
    let value = Scheduler::from_str(name).map_err(diffusion_error)?;
    return_int(i64::from(value.as_raw()))
}

/// Converts a sample method enum value to its name.
#[pd_host_function(name = "flint::sd::sample_method_name")]
pub(super) fn sd_sample_method_name_impl(sample_method: i64) -> VmResult<CallOutcome> {
    let value =
        SampleMethod::try_from(checked_i32(sample_method, "sample_method")? as SamplerEnumRepr)
            .map_err(diffusion_error)?;
    return_value(Value::String(value.as_str().to_owned().into()))
}

/// Converts a scheduler enum value to its name.
#[pd_host_function(name = "flint::sd::scheduler_name")]
pub(super) fn sd_scheduler_name_impl(scheduler: i64) -> VmResult<CallOutcome> {
    let value = Scheduler::try_from(checked_i32(scheduler, "scheduler")? as SamplerEnumRepr)
        .map_err(diffusion_error)?;
    return_value(Value::String(value.as_str().to_owned().into()))
}

/// Gets the model-specific default sample method for a context.
#[pd_host_function(name = "flint::sd::get_default_sample_method")]
pub(super) fn sd_get_default_sample_method_impl(ctx_handle: i64) -> VmResult<CallOutcome> {
    let handles = SD_CTX_HANDLES
        .lock()
        .map_err(|_| registry_error("context"))?;
    let context = handles
        .get(&ctx_handle)
        .ok_or_else(|| unknown_handle("context", ctx_handle))?;
    let value = context.default_sample_method().map_err(diffusion_error)?;
    return_int(i64::from(value.as_raw()))
}

/// Gets the model-specific default scheduler for a sampler.
#[pd_host_function(name = "flint::sd::get_default_scheduler")]
pub(super) fn sd_get_default_scheduler_impl(
    ctx_handle: i64,
    sample_method: i64,
) -> VmResult<CallOutcome> {
    let sample_method =
        SampleMethod::try_from(checked_i32(sample_method, "sample_method")? as SamplerEnumRepr)
            .map_err(diffusion_error)?;
    let handles = SD_CTX_HANDLES
        .lock()
        .map_err(|_| registry_error("context"))?;
    let context = handles
        .get(&ctx_handle)
        .ok_or_else(|| unknown_handle("context", ctx_handle))?;
    let value = context
        .default_scheduler(sample_method)
        .map_err(diffusion_error)?;
    return_int(i64::from(value.as_raw()))
}

/// Runs image generation and returns an owned image batch handle.
#[pd_host_function(name = "flint::sd::generate_image")]
pub(super) fn sd_generate_image_impl(ctx_handle: i64, params_handle: i64) -> VmResult<CallOutcome> {
    let params = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| registry_error("image parameter"))?
        .get(&params_handle)
        .cloned()
        .ok_or_else(|| unknown_handle("image parameter", params_handle))?;
    let mut contexts = SD_CTX_HANDLES
        .lock()
        .map_err(|_| registry_error("context"))?;
    let context = contexts
        .get_mut(&ctx_handle)
        .ok_or_else(|| unknown_handle("context", ctx_handle))?;
    let images = context.generate_image(&params).map_err(diffusion_error)?;
    let handle = next_sd_handle();
    SD_IMAGES_HANDLES
        .lock()
        .map_err(|_| registry_error("image batch"))?
        .insert(handle, images);
    return_int(handle)
}

/// Saves one image from an owned image batch handle.
#[pd_host_function(name = "flint::sd::images_save")]
pub(super) fn sd_images_save_impl(
    images_handle: i64,
    index: i64,
    output_path: &str,
) -> VmResult<CallOutcome> {
    let handles = SD_IMAGES_HANDLES
        .lock()
        .map_err(|_| registry_error("image batch"))?;
    let images = handles
        .get(&images_handle)
        .ok_or_else(|| unknown_handle("image batch", images_handle))?;
    let index = usize::try_from(index)
        .ok()
        .filter(|index| *index < images.len())
        .ok_or_else(|| {
            host_error(format!(
                "image index {index} is out of range for {} image(s)",
                images.len()
            ))
        })?;
    let output_path = PathBuf::from(output_path);
    ensure_parent_dir(&output_path)
        .map_err(|err| host_error(format!("failed to create output directory: {err:#}")))?;
    images[index]
        .save(&output_path)
        .map_err(|err| host_error(format!("failed to save stable diffusion image: {err:#}")))?;
    return_value(Value::Bool(true))
}

/// Drops an owned image batch handle.
#[pd_host_function(name = "flint::sd::free_sd_images")]
pub(super) fn sd_free_sd_images_impl(images_handle: i64) -> VmResult<CallOutcome> {
    SD_IMAGES_HANDLES
        .lock()
        .map_err(|_| registry_error("image batch"))?
        .remove(&images_handle)
        .ok_or_else(|| unknown_handle("image batch", images_handle))?;
    return_value(Value::Bool(true))
}

fn prepare_native(package: StableDiffusionCpp) -> VmResult<()> {
    let mut runtime = SD_NATIVE_RUNTIME
        .lock()
        .map_err(|_| host_error("stable diffusion runtime registry is poisoned"))?;
    if let Some(runtime) = runtime.as_ref() {
        if runtime.package != package {
            return Err(host_error(format!(
                "stable diffusion runtime already initialized with {}; cannot switch to {} in the same process",
                runtime.package, package
            )));
        }
        return Ok(());
    }

    ggml::ensure_stable_diffusion_backends(package)
        .map_err(|err| host_error(format!("failed to load stable diffusion backends: {err:#}")))?;
    let directory = ggml::stable_diffusion_package_dir(package);
    let library_path = stable_diffusion_library_path(&directory);
    let library = ggml::load_library(&library_path)
        .with_context(|| format!("failed to load {}", library_path.display()))
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;

    let version = koharu_diffusion::version();
    if version.is_empty() {
        return Err(host_error("stable-diffusion.cpp returned an empty version"));
    }
    koharu_diffusion::set_log_callback(|message| {
        eprintln!(
            "stable-diffusion.cpp [{}] {}",
            message.level,
            message.text.trim_end()
        );
    })
    .map_err(diffusion_error)?;
    *runtime = Some(NativeRuntime {
        package,
        _library: library,
    });
    Ok(())
}

fn stable_diffusion_library_path(directory: &Path) -> PathBuf {
    if cfg!(windows) {
        directory.join("stable-diffusion.dll")
    } else if cfg!(target_os = "macos") {
        directory.join("libstable-diffusion.dylib")
    } else {
        directory.join("libstable-diffusion.so")
    }
}

fn parse_weight_type(value: &str) -> VmResult<WeightType> {
    let value = match value.to_ascii_lowercase().as_str() {
        "" | "auto" => return Ok(WeightType::Auto),
        "float" => "f32",
        "half" => "f16",
        "q8" => "q8_0",
        _ => value,
    };
    WeightType::from_str(value).map_err(diffusion_error)
}

fn normalize_auto(value: &str) -> &str {
    if value.is_empty() { "auto" } else { value }
}

fn optional_path(value: &str) -> Option<PathBuf> {
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn optional_string(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
}

fn max_vram_value(value: &str) -> Option<String> {
    if value.eq_ignore_ascii_case("auto") {
        Some("-1".to_owned())
    } else {
        optional_string(value)
    }
}

fn native_backend(value: &str) -> Option<String> {
    let lower = value.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "" | "auto" | "cpu" | "cuda" | "cuda12" | "vulkan"
    ) {
        None
    } else {
        Some(value.to_owned())
    }
}

fn checked_i32(value: i64, name: &str) -> VmResult<i32> {
    i32::try_from(value).map_err(|_| host_error(format!("{name} is out of range for int32")))
}

fn validate_positive(value: i64, name: &str) -> VmResult<()> {
    if value <= 0 {
        Err(host_error(format!("{name} must be positive")))
    } else {
        Ok(())
    }
}

fn diffusion_error(error: koharu_diffusion::Error) -> vm::VmError {
    host_error(format!("stable diffusion error: {error}"))
}

fn registry_error(kind: &str) -> vm::VmError {
    host_error(format!("stable diffusion {kind} registry is poisoned"))
}

fn unknown_handle(kind: &str, handle: i64) -> vm::VmError {
    host_error(format!("unknown stable diffusion {kind} handle {handle}"))
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
