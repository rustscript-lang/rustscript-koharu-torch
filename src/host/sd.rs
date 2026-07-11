use std::ffi::{CStr, CString, c_char, c_float, c_int, c_void};
use std::mem::MaybeUninit;
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use image::{ImageBuffer, Luma, Rgb, Rgba};
use koharu_runtime::package::stable_diffusion_cpp::StableDiffusionCpp;
use libloading::Library;
use pd_host_function::pd_host_function;

use crate::{CallOutcome, Value, VmResult};

use super::{ggml, host_error, return_int, return_value};

const SAMPLE_METHOD_COUNT: c_int = 20;
const SCHEDULER_COUNT: c_int = 16;
const SD_TYPE_COUNT: c_int = 42;
const SD_TYPE_F32: c_int = 0;
const SD_TYPE_F16: c_int = 1;
const SD_TYPE_Q8_0: c_int = 8;
const SD_TYPE_Q4_K: c_int = 12;
const SD_TYPE_Q5_K: c_int = 13;
const SD_TYPE_Q6_K: c_int = 14;
const SD_TYPE_BF16: c_int = 30;
#[repr(C)]
struct SdTilingParams {
    enabled: bool,
    temporal_tiling: bool,
    tile_size_x: c_int,
    tile_size_y: c_int,
    target_overlap: c_float,
    rel_size_x: c_float,
    rel_size_y: c_float,
    extra_tiling_args: *const c_char,
}

#[repr(C)]
struct SdEmbedding {
    name: *const c_char,
    path: *const c_char,
}

#[repr(C)]
struct SdCtxParams {
    model_path: *const c_char,
    clip_l_path: *const c_char,
    clip_g_path: *const c_char,
    clip_vision_path: *const c_char,
    t5xxl_path: *const c_char,
    llm_path: *const c_char,
    llm_vision_path: *const c_char,
    diffusion_model_path: *const c_char,
    high_noise_diffusion_model_path: *const c_char,
    uncond_diffusion_model_path: *const c_char,
    embeddings_connectors_path: *const c_char,
    vae_path: *const c_char,
    audio_vae_path: *const c_char,
    taesd_path: *const c_char,
    control_net_path: *const c_char,
    embeddings: *const SdEmbedding,
    embedding_count: u32,
    photo_maker_path: *const c_char,
    pulid_weights_path: *const c_char,
    tensor_type_rules: *const c_char,
    n_threads: c_int,
    wtype: c_int,
    rng_type: c_int,
    sampler_rng_type: c_int,
    prediction: c_int,
    lora_apply_mode: c_int,
    enable_mmap: bool,
    flash_attn: bool,
    diffusion_flash_attn: bool,
    tae_preview_only: bool,
    diffusion_conv_direct: bool,
    vae_conv_direct: bool,
    force_sdxl_vae_conv_scale: bool,
    vae_format: c_int,
    max_vram: *const c_char,
    stream_layers: bool,
    eager_load: bool,
    backend: *const c_char,
    params_backend: *const c_char,
    split_mode: *const c_char,
    auto_fit: bool,
    rpc_servers: *const c_char,
    model_args: *const c_char,
}

#[repr(C)]
struct SdImage {
    width: u32,
    height: u32,
    channel: u32,
    data: *mut u8,
}

#[repr(C)]
struct SdSlgParams {
    layers: *mut c_int,
    layer_count: usize,
    layer_start: c_float,
    layer_end: c_float,
    scale: c_float,
}

#[repr(C)]
struct SdGuidanceParams {
    txt_cfg: c_float,
    img_cfg: c_float,
    distilled_guidance: c_float,
    slg: SdSlgParams,
}

#[repr(C)]
struct SdSampleParams {
    guidance: SdGuidanceParams,
    scheduler: c_int,
    sample_method: c_int,
    sample_steps: c_int,
    eta: c_float,
    shifted_timestep: c_int,
    custom_sigmas: *mut c_float,
    custom_sigmas_count: c_int,
    flow_shift: c_float,
    extra_sample_args: *const c_char,
}

#[repr(C)]
struct SdPmParams {
    id_images: *mut SdImage,
    id_images_count: c_int,
    id_embed_path: *const c_char,
    style_strength: c_float,
}

#[repr(C)]
struct SdPulidParams {
    id_embedding_path: *const c_char,
    id_weight: c_float,
}

#[repr(C)]
struct SdCacheParams {
    mode: c_int,
    reuse_threshold: c_float,
    start_percent: c_float,
    end_percent: c_float,
    error_decay_rate: c_float,
    use_relative_threshold: bool,
    reset_error_on_compute: bool,
    fn_compute_blocks: c_int,
    bn_compute_blocks: c_int,
    residual_diff_threshold: c_float,
    max_warmup_steps: c_int,
    max_cached_steps: c_int,
    max_continuous_cached_steps: c_int,
    taylorseer_n_derivatives: c_int,
    taylorseer_skip_interval: c_int,
    scm_mask: *const c_char,
    scm_policy_dynamic: bool,
    spectrum_w: c_float,
    spectrum_m: c_int,
    spectrum_lam: c_float,
    spectrum_window_size: c_int,
    spectrum_flex_window: c_float,
    spectrum_warmup_steps: c_int,
    spectrum_stop_percent: c_float,
}

#[repr(C)]
struct SdLora {
    is_high_noise: bool,
    multiplier: c_float,
    path: *const c_char,
}

#[repr(C)]
struct SdHiresParams {
    enabled: bool,
    upscaler: c_int,
    model_path: *const c_char,
    scale: c_float,
    target_width: c_int,
    target_height: c_int,
    steps: c_int,
    denoising_strength: c_float,
    upscale_tile_size: c_int,
    custom_sigmas: *mut c_float,
    custom_sigmas_count: c_int,
}

#[repr(C)]
struct SdImgGenParams {
    loras: *const SdLora,
    lora_count: u32,
    prompt: *const c_char,
    negative_prompt: *const c_char,
    clip_skip: c_int,
    init_image: SdImage,
    ref_images: *mut SdImage,
    ref_images_count: c_int,
    auto_resize_ref_image: bool,
    increase_ref_index: bool,
    mask_image: SdImage,
    width: c_int,
    height: c_int,
    sample_params: SdSampleParams,
    strength: c_float,
    seed: i64,
    batch_count: c_int,
    control_image: SdImage,
    control_strength: c_float,
    pm_params: SdPmParams,
    pulid_params: SdPulidParams,
    vae_tiling_params: SdTilingParams,
    cache: SdCacheParams,
    hires: SdHiresParams,
    qwen_image_layers: c_int,
    circular_x: bool,
    circular_y: bool,
}

enum SdCtx {}

struct SdApi {
    _library: Library,
    sd_set_log_callback: unsafe extern "C" fn(SdLogCallback, *mut c_void),
    sd_ctx_params_init: unsafe extern "C" fn(*mut SdCtxParams),
    new_sd_ctx: unsafe extern "C" fn(*const SdCtxParams) -> *mut SdCtx,
    free_sd_ctx: unsafe extern "C" fn(*mut SdCtx),
    sd_img_gen_params_init: unsafe extern "C" fn(*mut SdImgGenParams),
    generate_image: unsafe extern "C" fn(
        *mut SdCtx,
        *const SdImgGenParams,
        *mut *mut SdImage,
        *mut c_int,
    ) -> bool,
    sd_list_devices: unsafe extern "C" fn(*mut c_char, usize) -> usize,
    free_sd_images: unsafe extern "C" fn(*mut SdImage, c_int),
    sd_sample_method_name: unsafe extern "C" fn(c_int) -> *const c_char,
    str_to_sample_method: unsafe extern "C" fn(*const c_char) -> c_int,
    sd_scheduler_name: unsafe extern "C" fn(c_int) -> *const c_char,
    str_to_scheduler: unsafe extern "C" fn(*const c_char) -> c_int,
    sd_get_default_sample_method: unsafe extern "C" fn(*const SdCtx) -> c_int,
    sd_get_default_scheduler: unsafe extern "C" fn(*const SdCtx, c_int) -> c_int,
}

type SdLogCallback = unsafe extern "C" fn(c_int, *const c_char, *mut c_void);

static NEXT_SD_HANDLE: AtomicI64 = AtomicI64::new(1);
static SD_CTX_PARAMS_HANDLES: LazyLock<Mutex<std::collections::HashMap<i64, SdCtxParamsResource>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));
static SD_CTX_HANDLES: LazyLock<Mutex<std::collections::HashMap<i64, SdCtxResource>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));
static SD_IMG_PARAMS_HANDLES: LazyLock<Mutex<std::collections::HashMap<i64, SdImgParamsResource>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));
static SD_IMAGES_HANDLES: LazyLock<Mutex<std::collections::HashMap<i64, SdImagesResource>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashMap::new()));

struct SdCtxParamsResource {
    params: SdCtxParams,
    model_path: Option<CString>,
    diffusion_model_path: Option<CString>,
    vae_path: Option<CString>,
    llm_path: Option<CString>,
    backend: Option<CString>,
    params_backend: Option<CString>,
    max_vram: Option<CString>,
}

// The raw C pointers inside the params point at CStrings owned by the same
// resource and all access is serialized through the handle mutex.
unsafe impl Send for SdCtxParamsResource {}

struct SdCtxResource {
    package: StableDiffusionCpp,
    ctx: usize,
}

struct SdImgParamsResource {
    params: SdImgGenParams,
    prompt: CString,
    negative_prompt: CString,
}

unsafe impl Send for SdImgParamsResource {}

struct SdImagesResource {
    package: StableDiffusionCpp,
    images: usize,
    count: c_int,
}

fn next_sd_handle() -> i64 {
    NEXT_SD_HANDLE.fetch_add(1, Ordering::Relaxed)
}

/// Creates default stable-diffusion.cpp context parameters and returns a handle.
#[pd_host_function(name = "flint::sd::ctx_params_init")]
pub(super) fn sd_ctx_params_init_impl() -> VmResult<CallOutcome> {
    let package = StableDiffusionCpp::for_current_target().map_err(|err| {
        host_error(format!(
            "failed to select stable diffusion package: {err:#}"
        ))
    })?;
    let api = sd_api(package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    let mut params = unsafe { MaybeUninit::<SdCtxParams>::zeroed().assume_init() };
    unsafe { (api.sd_ctx_params_init)(&mut params) };

    let handle = next_sd_handle();
    SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?
        .insert(
            handle,
            SdCtxParamsResource {
                params,
                model_path: None,
                diffusion_model_path: None,
                vae_path: None,
                llm_path: None,
                backend: None,
                params_backend: None,
                max_vram: None,
            },
        );
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
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion ctx params handle {handle}"
        ))
    })?;
    resource.model_path = optional_c_string(model_path, "model_path")?;
    resource.diffusion_model_path =
        optional_c_string(diffusion_model_path, "diffusion_model_path")?;
    resource.vae_path = optional_c_string(vae_path, "vae_path")?;
    resource.llm_path = optional_c_string(llm_path, "llm_path")?;
    resource.params.model_path = optional_ptr(&resource.model_path);
    resource.params.diffusion_model_path = optional_ptr(&resource.diffusion_model_path);
    resource.params.vae_path = optional_ptr(&resource.vae_path);
    resource.params.llm_path = optional_ptr(&resource.llm_path);
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
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion ctx params handle {handle}"
        ))
    })?;
    resource.backend = optional_c_string(backend, "backend")?;
    resource.params_backend = optional_c_string(params_backend, "params_backend")?;
    resource.max_vram = optional_c_string(max_vram, "max_vram")?;
    resource.params.backend = sd_backend_ptr(&resource.backend);
    resource.params.params_backend = optional_ptr(&resource.params_backend);
    resource.params.max_vram = optional_ptr(&resource.max_vram);
    return_value(Value::Bool(true))
}

/// Sets the model weight type on stable-diffusion.cpp context parameters.
#[pd_host_function(name = "flint::sd::ctx_params_set_wtype")]
pub(super) fn sd_ctx_params_set_wtype_impl(handle: i64, wtype: &str) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion ctx params handle {handle}"
        ))
    })?;
    resource.params.wtype = parse_wtype(wtype)?;
    return_value(Value::Bool(true))
}

/// Sets the VAE format on stable-diffusion.cpp context parameters.
#[pd_host_function(name = "flint::sd::ctx_params_set_vae_format")]
pub(super) fn sd_ctx_params_set_vae_format_impl(
    handle: i64,
    vae_format: i64,
) -> VmResult<CallOutcome> {
    let mut handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion ctx params handle {handle}"
        ))
    })?;
    resource.params.vae_format = checked_c_int(vae_format, "vae_format")?;
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
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion ctx params handle {handle}"
        ))
    })?;
    resource.params.enable_mmap = enable_mmap;
    resource.params.flash_attn = flash_attn;
    resource.params.diffusion_flash_attn = diffusion_flash_attn;
    return_value(Value::Bool(true))
}

/// Creates a stable-diffusion.cpp context from context parameters.
#[pd_host_function(name = "flint::sd::new_sd_ctx")]
pub(super) fn sd_new_sd_ctx_impl(params_handle: i64) -> VmResult<CallOutcome> {
    let params_handles = SD_CTX_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx params registry is poisoned"))?;
    let params_resource = params_handles.get(&params_handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion ctx params handle {params_handle}"
        ))
    })?;
    let package = select_package(&params_resource.backend).map_err(|err| {
        host_error(format!(
            "failed to select stable diffusion backend: {err:#}"
        ))
    })?;
    let api = sd_api(package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    unsafe { (api.sd_set_log_callback)(sd_log_callback, ptr::null_mut()) };
    let devices = api
        .list_devices()
        .map_err(|err| host_error(format!("failed to list stable diffusion devices: {err:#}")))?;
    if devices.trim().is_empty() {
        return Err(host_error(
            "stable-diffusion.cpp reported no ggml backend devices",
        ));
    }
    eprintln!("stable-diffusion.cpp devices:\n{devices}");

    let ctx = unsafe { (api.new_sd_ctx)(&params_resource.params) };
    if ctx.is_null() {
        return Err(host_error("new_sd_ctx returned null"));
    }
    let handle = next_sd_handle();
    SD_CTX_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx registry is poisoned"))?
        .insert(
            handle,
            SdCtxResource {
                package,
                ctx: ctx.cast::<c_void>() as usize,
            },
        );
    return_int(handle)
}

/// Frees a stable-diffusion.cpp context handle.
#[pd_host_function(name = "flint::sd::free_sd_ctx")]
pub(super) fn sd_free_sd_ctx_impl(ctx_handle: i64) -> VmResult<CallOutcome> {
    let resource = SD_CTX_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx registry is poisoned"))?
        .remove(&ctx_handle)
        .ok_or_else(|| host_error(format!("unknown stable diffusion ctx handle {ctx_handle}")))?;
    let api = sd_api(resource.package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    unsafe { (api.free_sd_ctx)(resource.ctx as *mut SdCtx) };
    return_value(Value::Bool(true))
}

/// Creates default stable-diffusion.cpp image generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_init")]
pub(super) fn sd_img_gen_params_init_impl() -> VmResult<CallOutcome> {
    let package = StableDiffusionCpp::for_current_target().map_err(|err| {
        host_error(format!(
            "failed to select stable diffusion package: {err:#}"
        ))
    })?;
    let api = sd_api(package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    let mut params = unsafe { MaybeUninit::<SdImgGenParams>::zeroed().assume_init() };
    unsafe { (api.sd_img_gen_params_init)(&mut params) };
    let prompt = c_string("", "prompt")?;
    let negative_prompt = c_string("", "negative_prompt")?;
    params.prompt = prompt.as_ptr();
    params.negative_prompt = negative_prompt.as_ptr();

    let handle = next_sd_handle();
    SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion image params registry is poisoned"))?
        .insert(
            handle,
            SdImgParamsResource {
                params,
                prompt,
                negative_prompt,
            },
        );
    return_int(handle)
}

/// Sets prompt strings on stable-diffusion.cpp image generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_set_prompt")]
pub(super) fn sd_img_gen_params_set_prompt_impl(
    handle: i64,
    prompt: &str,
    negative_prompt: &str,
) -> VmResult<CallOutcome> {
    let mut handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion image params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion image params handle {handle}"
        ))
    })?;
    resource.prompt = c_string(prompt, "prompt")?;
    resource.negative_prompt = c_string(negative_prompt, "negative_prompt")?;
    resource.params.prompt = resource.prompt.as_ptr();
    resource.params.negative_prompt = resource.negative_prompt.as_ptr();
    return_value(Value::Bool(true))
}

/// Sets output dimensions on stable-diffusion.cpp image generation parameters.
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
        .map_err(|_| host_error("stable diffusion image params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion image params handle {handle}"
        ))
    })?;
    resource.params.width = checked_c_int(width, "width")?;
    resource.params.height = checked_c_int(height, "height")?;
    return_value(Value::Bool(true))
}

/// Sets sampling options on stable-diffusion.cpp image generation parameters.
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
        .map_err(|_| host_error("stable diffusion image params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion image params handle {handle}"
        ))
    })?;
    resource.params.seed = seed;
    resource.params.batch_count = 1;
    resource.params.sample_params.sample_steps = checked_c_int(steps, "steps")?;
    resource.params.sample_params.guidance.txt_cfg = cfg_scale as c_float;
    return_value(Value::Bool(true))
}

/// Sets sample method and scheduler on stable-diffusion.cpp image generation parameters.
#[pd_host_function(name = "flint::sd::img_gen_params_set_sampler")]
pub(super) fn sd_img_gen_params_set_sampler_impl(
    handle: i64,
    sample_method: i64,
    scheduler: i64,
) -> VmResult<CallOutcome> {
    let mut handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion image params registry is poisoned"))?;
    let resource = handles.get_mut(&handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion image params handle {handle}"
        ))
    })?;
    resource.params.sample_params.sample_method = checked_c_int(sample_method, "sample_method")?;
    resource.params.sample_params.scheduler = checked_c_int(scheduler, "scheduler")?;
    return_value(Value::Bool(true))
}

/// Converts a stable-diffusion.cpp sample method name to its enum value.
#[pd_host_function(name = "flint::sd::str_to_sample_method")]
pub(super) fn sd_str_to_sample_method_impl(name: &str) -> VmResult<CallOutcome> {
    if name.is_empty() || name.eq_ignore_ascii_case("auto") {
        return return_int(i64::from(SAMPLE_METHOD_COUNT));
    }
    let name = c_string(name, "sample_method")?;
    let api = sd_api_for_current_target()?;
    let value = unsafe { (api.str_to_sample_method)(name.as_ptr()) };
    if value == SAMPLE_METHOD_COUNT {
        return Err(host_error(format!(
            "unknown stable diffusion sample method '{}'",
            name.to_string_lossy()
        )));
    }
    return_int(i64::from(value))
}

/// Converts a stable-diffusion.cpp scheduler name to its enum value.
#[pd_host_function(name = "flint::sd::str_to_scheduler")]
pub(super) fn sd_str_to_scheduler_impl(name: &str) -> VmResult<CallOutcome> {
    if name.is_empty() || name.eq_ignore_ascii_case("auto") {
        return return_int(i64::from(SCHEDULER_COUNT));
    }
    let name = c_string(name, "scheduler")?;
    let api = sd_api_for_current_target()?;
    let value = unsafe { (api.str_to_scheduler)(name.as_ptr()) };
    if value == SCHEDULER_COUNT {
        return Err(host_error(format!(
            "unknown stable diffusion scheduler '{}'",
            name.to_string_lossy()
        )));
    }
    return_int(i64::from(value))
}

/// Converts a stable-diffusion.cpp sample method enum value to its name.
#[pd_host_function(name = "flint::sd::sample_method_name")]
pub(super) fn sd_sample_method_name_impl(sample_method: i64) -> VmResult<CallOutcome> {
    let sample_method = checked_c_int(sample_method, "sample_method")?;
    let api = sd_api_for_current_target()?;
    let name = unsafe { (api.sd_sample_method_name)(sample_method) };
    return_c_string(name, "sample_method")
}

/// Converts a stable-diffusion.cpp scheduler enum value to its name.
#[pd_host_function(name = "flint::sd::scheduler_name")]
pub(super) fn sd_scheduler_name_impl(scheduler: i64) -> VmResult<CallOutcome> {
    let scheduler = checked_c_int(scheduler, "scheduler")?;
    let api = sd_api_for_current_target()?;
    let name = unsafe { (api.sd_scheduler_name)(scheduler) };
    return_c_string(name, "scheduler")
}

/// Gets stable-diffusion.cpp's default sample method for a context.
#[pd_host_function(name = "flint::sd::get_default_sample_method")]
pub(super) fn sd_get_default_sample_method_impl(ctx_handle: i64) -> VmResult<CallOutcome> {
    let ctx_handles = SD_CTX_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx registry is poisoned"))?;
    let ctx_resource = ctx_handles
        .get(&ctx_handle)
        .ok_or_else(|| host_error(format!("unknown stable diffusion ctx handle {ctx_handle}")))?;
    let api = sd_api(ctx_resource.package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    let value = unsafe { (api.sd_get_default_sample_method)(ctx_resource.ctx as *const SdCtx) };
    return_int(i64::from(value))
}

/// Gets stable-diffusion.cpp's default scheduler for a context and sample method.
#[pd_host_function(name = "flint::sd::get_default_scheduler")]
pub(super) fn sd_get_default_scheduler_impl(
    ctx_handle: i64,
    sample_method: i64,
) -> VmResult<CallOutcome> {
    let sample_method = checked_c_int(sample_method, "sample_method")?;
    let ctx_handles = SD_CTX_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx registry is poisoned"))?;
    let ctx_resource = ctx_handles
        .get(&ctx_handle)
        .ok_or_else(|| host_error(format!("unknown stable diffusion ctx handle {ctx_handle}")))?;
    let api = sd_api(ctx_resource.package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    let value =
        unsafe { (api.sd_get_default_scheduler)(ctx_resource.ctx as *const SdCtx, sample_method) };
    return_int(i64::from(value))
}

/// Runs stable-diffusion.cpp image generation and returns an image batch handle.
#[pd_host_function(name = "flint::sd::generate_image")]
pub(super) fn sd_generate_image_impl(ctx_handle: i64, params_handle: i64) -> VmResult<CallOutcome> {
    let ctx_handles = SD_CTX_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion ctx registry is poisoned"))?;
    let ctx_resource = ctx_handles
        .get(&ctx_handle)
        .ok_or_else(|| host_error(format!("unknown stable diffusion ctx handle {ctx_handle}")))?;
    let params_handles = SD_IMG_PARAMS_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion image params registry is poisoned"))?;
    let params_resource = params_handles.get(&params_handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion image params handle {params_handle}"
        ))
    })?;
    let api = sd_api(ctx_resource.package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    let mut images = ptr::null_mut();
    let mut image_count = 0;
    let ok = unsafe {
        (api.generate_image)(
            ctx_resource.ctx as *mut SdCtx,
            &params_resource.params,
            &mut images,
            &mut image_count,
        )
    };
    if !ok {
        return Err(host_error("generate_image returned false"));
    }
    if images.is_null() || image_count <= 0 {
        return Err(host_error("generate_image returned no images"));
    }
    let handle = next_sd_handle();
    SD_IMAGES_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion images registry is poisoned"))?
        .insert(
            handle,
            SdImagesResource {
                package: ctx_resource.package,
                images: images.cast::<c_void>() as usize,
                count: image_count,
            },
        );
    return_int(handle)
}

/// Saves one image from a stable-diffusion.cpp image batch handle.
#[pd_host_function(name = "flint::sd::images_save")]
pub(super) fn sd_images_save_impl(
    images_handle: i64,
    index: i64,
    output_path: &str,
) -> VmResult<CallOutcome> {
    let handles = SD_IMAGES_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion images registry is poisoned"))?;
    let resource = handles.get(&images_handle).ok_or_else(|| {
        host_error(format!(
            "unknown stable diffusion images handle {images_handle}"
        ))
    })?;
    if index < 0 || index >= i64::from(resource.count) {
        return Err(host_error(format!(
            "image index {index} is out of range for {} image(s)",
            resource.count
        )));
    }
    let output_path = PathBuf::from(output_path);
    ensure_parent_dir(&output_path)
        .map_err(|err| host_error(format!("failed to create output directory: {err:#}")))?;
    let image = unsafe { &*((resource.images as *mut SdImage).add(index as usize)) };
    save_image(image, &output_path)
        .map_err(|err| host_error(format!("failed to save stable diffusion image: {err:#}")))?;
    return_value(Value::Bool(true))
}

/// Frees a stable-diffusion.cpp image batch handle.
#[pd_host_function(name = "flint::sd::free_sd_images")]
pub(super) fn sd_free_sd_images_impl(images_handle: i64) -> VmResult<CallOutcome> {
    let resource = SD_IMAGES_HANDLES
        .lock()
        .map_err(|_| host_error("stable diffusion images registry is poisoned"))?
        .remove(&images_handle)
        .ok_or_else(|| {
            host_error(format!(
                "unknown stable diffusion images handle {images_handle}"
            ))
        })?;
    let api = sd_api(resource.package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))?;
    unsafe { (api.free_sd_images)(resource.images as *mut SdImage, resource.count) };
    return_value(Value::Bool(true))
}

fn sd_api(package: StableDiffusionCpp) -> Result<&'static SdApi> {
    static WINDOWS_X64_CPU_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static WINDOWS_X64_CUDA12_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static WINDOWS_X64_VULKAN_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static WINDOWS_X64_ROCM711_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static WINDOWS_X64_ROCM7130_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static LINUX_X64_CPU_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static LINUX_X64_VULKAN_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static LINUX_X64_ROCM721_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static LINUX_X64_ROCM7130_API: OnceLock<Result<SdApi, String>> = OnceLock::new();
    static MACOS_ARM64_API: OnceLock<Result<SdApi, String>> = OnceLock::new();

    let api = match package {
        StableDiffusionCpp::WindowsX64Cpu => &WINDOWS_X64_CPU_API,
        StableDiffusionCpp::WindowsX64Cuda12 => &WINDOWS_X64_CUDA12_API,
        StableDiffusionCpp::WindowsX64Vulkan => &WINDOWS_X64_VULKAN_API,
        StableDiffusionCpp::WindowsX64Rocm711 => &WINDOWS_X64_ROCM711_API,
        StableDiffusionCpp::WindowsX64Rocm7130 => &WINDOWS_X64_ROCM7130_API,
        StableDiffusionCpp::LinuxX64Cpu => &LINUX_X64_CPU_API,
        StableDiffusionCpp::LinuxX64Vulkan => &LINUX_X64_VULKAN_API,
        StableDiffusionCpp::LinuxX64Rocm721 => &LINUX_X64_ROCM721_API,
        StableDiffusionCpp::LinuxX64Rocm7130 => &LINUX_X64_ROCM7130_API,
        StableDiffusionCpp::MacosArm64 => &MACOS_ARM64_API,
    };
    api.get_or_init(|| SdApi::load(package).map_err(|err| format!("{err:#}")))
        .as_ref()
        .map_err(|err| anyhow::anyhow!("{err}"))
}

impl SdApi {
    fn load(package: StableDiffusionCpp) -> Result<Self> {
        let _ggml_api = ggml::ensure_stable_diffusion_backends(package)?;
        let directory = ggml::stable_diffusion_package_dir(package);
        let library_path = stable_diffusion_library_path(&directory);
        let library = ggml::load_library(&library_path)
            .with_context(|| format!("failed to load {}", library_path.display()))?;

        unsafe {
            Ok(Self {
                sd_ctx_params_init: *library.get(b"sd_ctx_params_init\0")?,
                sd_set_log_callback: *library.get(b"sd_set_log_callback\0")?,
                new_sd_ctx: *library.get(b"new_sd_ctx\0")?,
                free_sd_ctx: *library.get(b"free_sd_ctx\0")?,
                sd_img_gen_params_init: *library.get(b"sd_img_gen_params_init\0")?,
                generate_image: *library.get(b"generate_image\0")?,
                sd_list_devices: *library.get(b"sd_list_devices\0")?,
                free_sd_images: *library.get(b"free_sd_images\0")?,
                sd_sample_method_name: *library.get(b"sd_sample_method_name\0")?,
                str_to_sample_method: *library.get(b"str_to_sample_method\0")?,
                sd_scheduler_name: *library.get(b"sd_scheduler_name\0")?,
                str_to_scheduler: *library.get(b"str_to_scheduler\0")?,
                sd_get_default_sample_method: *library.get(b"sd_get_default_sample_method\0")?,
                sd_get_default_scheduler: *library.get(b"sd_get_default_scheduler\0")?,
                _library: library,
            })
        }
    }

    fn list_devices(&self) -> Result<String> {
        let required = unsafe { (self.sd_list_devices)(ptr::null_mut(), 0) };
        if required == 0 {
            return Ok(String::new());
        }
        let mut buffer = vec![0u8; required + 1];
        let written =
            unsafe { (self.sd_list_devices)(buffer.as_mut_ptr().cast::<c_char>(), buffer.len()) };
        let len = written.min(required);
        Ok(String::from_utf8_lossy(&buffer[..len]).into_owned())
    }
}

fn sd_api_for_current_target() -> VmResult<&'static SdApi> {
    let package = StableDiffusionCpp::for_current_target().map_err(|err| {
        host_error(format!(
            "failed to select stable diffusion package: {err:#}"
        ))
    })?;
    sd_api(package)
        .map_err(|err| host_error(format!("failed to load stable-diffusion.cpp: {err:#}")))
}

unsafe extern "C" fn sd_log_callback(level: c_int, text: *const c_char, _data: *mut c_void) {
    if text.is_null() {
        return;
    }
    let level = match level {
        0 => "debug",
        1 => "info",
        2 => "warn",
        3 => "error",
        _ => "log",
    };
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    eprintln!("stable-diffusion.cpp [{level}] {text}");
}

fn select_package(backend: &Option<CString>) -> Result<StableDiffusionCpp> {
    ggml::select_stable_diffusion_package(
        backend.as_ref().map(|value| value.to_str().unwrap_or("")),
    )
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

fn save_image(image: &SdImage, path: &Path) -> Result<()> {
    if image.data.is_null() {
        bail!("generated image data is null");
    }
    let byte_count = image
        .width
        .checked_mul(image.height)
        .and_then(|pixels| pixels.checked_mul(image.channel))
        .context("generated image dimensions overflow")? as usize;
    let bytes = unsafe { std::slice::from_raw_parts(image.data, byte_count) }.to_vec();
    match image.channel {
        1 => ImageBuffer::<Luma<u8>, _>::from_raw(image.width, image.height, bytes)
            .context("failed to create grayscale image")?
            .save(path)?,
        3 => ImageBuffer::<Rgb<u8>, _>::from_raw(image.width, image.height, bytes)
            .context("failed to create RGB image")?
            .save(path)?,
        4 => ImageBuffer::<Rgba<u8>, _>::from_raw(image.width, image.height, bytes)
            .context("failed to create RGBA image")?
            .save(path)?,
        channels => bail!("unsupported generated image channel count {channels}"),
    }
    Ok(())
}

fn validate_positive(value: i64, label: &str) -> VmResult<()> {
    if value <= 0 {
        return Err(host_error(format!("{label} must be positive")));
    }
    Ok(())
}

fn checked_c_int(value: i64, label: &str) -> VmResult<c_int> {
    c_int::try_from(value).map_err(|_| host_error(format!("{label} is out of range")))
}

fn c_string(value: &str, label: &str) -> VmResult<CString> {
    CString::new(value).map_err(|_| host_error(format!("{label} must not contain NUL bytes")))
}

fn return_c_string(value: *const c_char, label: &str) -> VmResult<CallOutcome> {
    if value.is_null() {
        return Err(host_error(format!("{label} name pointer is null")));
    }
    let value = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    return_value(Value::String(value.into()))
}

fn optional_c_string(value: &str, label: &str) -> VmResult<Option<CString>> {
    if value.is_empty() || value.eq_ignore_ascii_case("auto") {
        Ok(None)
    } else {
        c_string(value, label).map(Some)
    }
}

fn optional_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn sd_backend_ptr(value: &Option<CString>) -> *const c_char {
    let Some(value) = value else {
        return ptr::null();
    };
    let backend = value.to_string_lossy().to_ascii_lowercase();
    if backend == "auto"
        || backend == "cpu"
        || backend == "cuda"
        || backend == "cuda12"
        || backend == "vulkan"
    {
        ptr::null()
    } else {
        value.as_ptr()
    }
}

fn parse_wtype(value: &str) -> VmResult<c_int> {
    match value.to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(SD_TYPE_COUNT),
        "f32" | "float" => Ok(SD_TYPE_F32),
        "f16" | "half" => Ok(SD_TYPE_F16),
        "bf16" => Ok(SD_TYPE_BF16),
        "q4_k" => Ok(SD_TYPE_Q4_K),
        "q5_k" => Ok(SD_TYPE_Q5_K),
        "q6_k" => Ok(SD_TYPE_Q6_K),
        "q8_0" | "q8" => Ok(SD_TYPE_Q8_0),
        other => Err(host_error(format!(
            "unknown stable diffusion weight type '{other}'"
        ))),
    }
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
