use std::collections::HashSet;
use std::time::Instant;

use pd_host_function::pd_host_function;

use crate::{CallOutcome, Value, VmResult};

use super::{host_error, return_int, return_value, with_context};

fn non_negative_index(value: i64, label: &str) -> VmResult<usize> {
    usize::try_from(value).map_err(|_| host_error(format!("{label} must be non-negative")))
}

/// Returns one host input tensor handle.
#[pd_host_function(name = "flint::runtime::input")]
pub(super) fn runtime_input_impl(index: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let index = non_negative_index(index, "input index")?;
        let handle = *context
            .inputs
            .get(index)
            .ok_or_else(|| host_error(format!("unknown input index {index}")))?;
        return_int(handle)
    })
}

/// Returns a string argument passed to the script runner.
#[pd_host_function(name = "flint::runtime::arg")]
pub(super) fn runtime_arg_impl(index: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let index = non_negative_index(index, "argument index")?;
        let value = context
            .args
            .get(index)
            .ok_or_else(|| host_error(format!("unknown runtime argument {index}")))?
            .clone();
        return_value(Value::String(value.into()))
    })
}

/// Parses one script runner argument as an integer.
#[pd_host_function(name = "flint::runtime::arg_int")]
pub(super) fn runtime_arg_int_impl(index: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let index = non_negative_index(index, "argument index")?;
        let value = context
            .args
            .get(index)
            .ok_or_else(|| host_error(format!("unknown runtime argument {index}")))?
            .parse::<i64>()
            .map_err(|err| host_error(format!("runtime argument {index} is not an int: {err}")))?;
        return_int(value)
    })
}

/// Parses one script runner argument as an integer with a fallback.
#[pd_host_function(name = "flint::runtime::arg_int_or")]
pub(super) fn runtime_arg_int_or_impl(index: i64, default: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let index = non_negative_index(index, "argument index")?;
        let Some(value) = context.args.get(index) else {
            return return_int(default);
        };
        let value = value
            .parse::<i64>()
            .map_err(|err| host_error(format!("runtime argument {index} is not an int: {err}")))?;
        return_int(value)
    })
}

/// Parses one script runner argument as a float with a fallback.
#[pd_host_function(name = "flint::runtime::arg_float_or")]
pub(super) fn runtime_arg_float_or_impl(index: i64, default: f64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let index = non_negative_index(index, "argument index")?;
        let Some(value) = context.args.get(index) else {
            return return_value(Value::Float(default));
        };
        let value = value
            .parse::<f64>()
            .map_err(|err| host_error(format!("runtime argument {index} is not a float: {err}")))?;
        return_value(Value::Float(value))
    })
}

/// Returns one script runner argument with a fallback.
#[pd_host_function(name = "flint::runtime::arg_or")]
pub(super) fn runtime_arg_or_impl(index: i64, default: &str) -> VmResult<CallOutcome> {
    with_context(|context| {
        let index = non_negative_index(index, "argument index")?;
        let value = context
            .args
            .get(index)
            .cloned()
            .unwrap_or_else(|| default.to_owned());
        return_value(Value::String(value.into()))
    })
}

/// Publishes a tensor handle as the script output.
#[pd_host_function(name = "flint::runtime::set_output")]
pub(super) fn runtime_set_output_impl(tensor: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        context.tensor(tensor)?;
        context.output = Some(tensor);
        return_value(Value::Bool(true))
    })
}

/// Publishes generated text as the script output.
#[pd_host_function(name = "flint::runtime::set_text_output")]
pub(super) fn runtime_set_text_output_impl(text: &str) -> VmResult<CallOutcome> {
    with_context(|context| {
        context.text_output = Some(text.to_owned());
        return_value(Value::Bool(true))
    })
}

/// Starts wall-clock accounting for generated output.
#[pd_host_function(name = "flint::runtime::start_timer")]
pub(super) fn runtime_start_timer_impl() -> VmResult<CallOutcome> {
    with_context(|context| {
        context.generation_started_at = Some(Instant::now());
        context.generated_tokens = None;
        context.decode_started_at = None;
        context.decode_tokens = None;
        context.generated_token_tensors.clear();
        return_value(Value::Bool(true))
    })
}

/// Starts wall-clock accounting for decode steps.
#[pd_host_function(name = "flint::runtime::start_decode_timer")]
pub(super) fn runtime_start_decode_timer_impl() -> VmResult<CallOutcome> {
    with_context(|context| {
        context.decode_started_at = Some(Instant::now());
        context.decode_tokens = None;
        return_value(Value::Bool(true))
    })
}

/// Stores the number of generated tokens.
#[pd_host_function(name = "flint::runtime::set_token_count")]
pub(super) fn runtime_set_token_count_impl(generated_tokens: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        if generated_tokens < 0 {
            return Err(host_error("generated token count must be non-negative"));
        }
        context.generated_tokens = Some(generated_tokens);
        return_value(Value::Bool(true))
    })
}

/// Stores the number of decode-step tokens.
#[pd_host_function(name = "flint::runtime::set_decode_token_count")]
pub(super) fn runtime_set_decode_token_count_impl(decode_tokens: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        if decode_tokens < 0 {
            return Err(host_error("decode token count must be non-negative"));
        }
        context.decode_tokens = Some(decode_tokens);
        return_value(Value::Bool(true))
    })
}

/// Retains two tensor handles and runtime-owned roots, dropping other handles.
#[pd_host_function(name = "flint::runtime::compact2")]
pub(super) fn runtime_compact2_impl(first: i64, second: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let mut keep = HashSet::new();
        keep.insert(first);
        keep.insert(second);
        keep.extend(context.inputs.iter().copied());
        keep.extend(context.weight_handles.values().copied());
        if let Some(output) = context.output {
            keep.insert(output);
        }
        context.tensors.retain(|handle, _| keep.contains(handle));
        context.pairs.clear();
        context
            .weight_handles
            .retain(|_, handle| context.tensors.contains_key(handle));
        let count = i64::try_from(context.tensors.len())
            .map_err(|_| host_error("tensor count exceeds RustScript integer range"))?;
        return_int(count)
    })
}
