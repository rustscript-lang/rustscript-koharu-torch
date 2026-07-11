use std::collections::HashMap;

use pd_host_function::pd_host_function;

use crate::{CallOutcome, Value, VmResult};

use super::{host_error, return_int, return_value, with_context};

type VmArrayRef<'a> = &'a [Value];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    String,
    Int,
    Float,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Store,
    StoreOption,
    StoreTrue,
    StoreFalse,
}

impl Action {
    fn parse(value: &str) -> VmResult<Self> {
        match value {
            "Store" => Ok(Self::Store),
            "StoreOption" => Ok(Self::StoreOption),
            "StoreTrue" => Ok(Self::StoreTrue),
            "StoreFalse" => Ok(Self::StoreFalse),
            _ => Err(host_error(format!(
                "unknown CLI action '{value}'; expected Store, StoreOption, StoreTrue, or StoreFalse"
            ))),
        }
    }

    fn takes_value(self) -> bool {
        matches!(self, Self::Store | Self::StoreOption)
    }
}

#[derive(Clone, Debug)]
pub(super) struct CliReference {
    parser: i64,
    kind: ValueKind,
    value: Value,
    names: Vec<String>,
    action: Option<Action>,
    help: String,
    required: bool,
    positional: bool,
    metavar: Option<String>,
    seen: bool,
}

#[derive(Clone, Debug, Default)]
pub(super) struct CliParser {
    description: String,
    references: Vec<i64>,
    parsed: bool,
}

fn names_from_values(values: &[Value]) -> VmResult<Vec<String>> {
    if values.is_empty() {
        return Err(host_error("option names must not be empty"));
    }
    values
        .iter()
        .map(|value| match value {
            Value::String(value) => Ok(value.as_str().to_owned()),
            _ => Err(host_error("option names must be strings")),
        })
        .collect()
}

fn value_kind(value: &Value) -> VmResult<ValueKind> {
    match value {
        Value::String(_) => Ok(ValueKind::String),
        Value::Int(_) => Ok(ValueKind::Int),
        Value::Float(_) => Ok(ValueKind::Float),
        Value::Bool(_) => Ok(ValueKind::Bool),
        _ => Err(host_error(
            "CLI references support string, int, float, and bool values",
        )),
    }
}

fn insert_reference(parser: i64, value: Value) -> VmResult<CallOutcome> {
    with_context(|context| {
        if !context.cli_parsers.contains_key(&parser) {
            return Err(host_error(format!("unknown CLI parser handle {parser}")));
        }
        let handle = context.next_cli_handle;
        context.next_cli_handle += 1;
        let kind = value_kind(&value)?;
        context.cli_references.insert(
            handle,
            CliReference {
                parser,
                kind,
                value,
                names: Vec::new(),
                action: None,
                help: String::new(),
                required: false,
                positional: false,
                metavar: None,
                seen: false,
            },
        );
        context
            .cli_parsers
            .get_mut(&parser)
            .expect("parser was checked above")
            .references
            .push(handle);
        return_int(handle)
    })
}

fn set_value(reference: &mut CliReference, raw: &str) -> VmResult<()> {
    reference.value = match reference.kind {
        ValueKind::String => Value::string(raw),
        ValueKind::Int => Value::Int(
            raw.parse::<i64>()
                .map_err(|error| host_error(format!("invalid integer '{}': {error}", raw)))?,
        ),
        ValueKind::Float => Value::Float(
            raw.parse::<f64>()
                .map_err(|error| host_error(format!("invalid float '{}': {error}", raw)))?,
        ),
        ValueKind::Bool => Value::Bool(
            raw.parse::<bool>()
                .map_err(|error| host_error(format!("invalid boolean '{}': {error}", raw)))?,
        ),
    };
    reference.seen = true;
    Ok(())
}

fn format_help(parser: &CliParser, references: &HashMap<i64, CliReference>) -> String {
    let mut output = parser.description.clone();
    if !output.is_empty() {
        output.push_str("\n\n");
    }
    output.push_str("Options:\n  -h, --help\tShow this help message\n");
    for handle in &parser.references {
        let Some(reference) = references.get(handle) else {
            continue;
        };
        output.push_str("  ");
        output.push_str(&reference.names.join(", "));
        if reference.action.is_some_and(Action::takes_value) {
            output.push(' ');
            output.push_str(reference.metavar.as_deref().unwrap_or("VALUE"));
        }
        if reference.required {
            output.push_str(" (required)");
        }
        if !reference.help.is_empty() {
            output.push('\t');
            output.push_str(&reference.help);
        }
        output.push('\n');
    }
    output
}

/// Creates an argument parser, matching argparse::ArgumentParser::new.
#[pd_host_function(name = "flint::cli::argument_parser")]
pub(super) fn cli_argument_parser_impl() -> VmResult<CallOutcome> {
    with_context(|context| {
        let handle = context.next_cli_handle;
        context.next_cli_handle += 1;
        context.cli_parsers.insert(handle, CliParser::default());
        return_int(handle)
    })
}

/// Sets the parser description.
#[pd_host_function(name = "flint::cli::set_description")]
pub(super) fn cli_set_description_impl(parser: i64, description: &str) -> VmResult<CallOutcome> {
    with_context(|context| {
        let parser = context
            .cli_parsers
            .get_mut(&parser)
            .ok_or_else(|| host_error(format!("unknown CLI parser handle {parser}")))?;
        parser.description = description.to_owned();
        return_value(Value::Bool(true))
    })
}

/// Creates a typed value reference attached to a parser.
#[pd_host_function(name = "flint::cli::refer")]
pub(super) fn cli_refer_impl(parser: i64, initial: Value) -> VmResult<CallOutcome> {
    insert_reference(parser, initial)
}

/// Adds named options and an argparse action to a reference.
#[pd_host_function(name = "flint::cli::add_option")]
pub(super) fn cli_add_option_impl(
    reference: i64,
    names: VmArrayRef<'_>,
    action: &str,
    help: &str,
) -> VmResult<CallOutcome> {
    let names = names_from_values(names)?;
    if names
        .iter()
        .any(|name| !name.starts_with('-') || name.len() < 2)
    {
        return Err(host_error("option names must begin with '-' or '--'"));
    }
    let action = Action::parse(action)?;
    with_context(|context| {
        let entry = context
            .cli_references
            .get_mut(&reference)
            .ok_or_else(|| host_error(format!("unknown CLI reference handle {reference}")))?;
        if matches!(action, Action::StoreTrue | Action::StoreFalse) && entry.kind != ValueKind::Bool
        {
            return Err(host_error(
                "StoreTrue and StoreFalse require a bool reference",
            ));
        }
        entry.names = names;
        entry.action = Some(action);
        entry.help = help.to_owned();
        return_int(reference)
    })
}

/// Adds a positional argument and an argparse action to a reference.
#[pd_host_function(name = "flint::cli::add_argument")]
pub(super) fn cli_add_argument_impl(
    reference: i64,
    name: &str,
    action: &str,
    help: &str,
) -> VmResult<CallOutcome> {
    let action = Action::parse(action)?;
    if !action.takes_value() {
        return Err(host_error(
            "positional arguments require Store or StoreOption",
        ));
    }
    with_context(|context| {
        let entry = context
            .cli_references
            .get_mut(&reference)
            .ok_or_else(|| host_error(format!("unknown CLI reference handle {reference}")))?;
        entry.names = vec![name.to_owned()];
        entry.action = Some(action);
        entry.help = help.to_owned();
        entry.positional = true;
        return_int(reference)
    })
}

/// Marks a reference as required.
#[pd_host_function(name = "flint::cli::required")]
pub(super) fn cli_required_impl(reference: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let entry = context
            .cli_references
            .get_mut(&reference)
            .ok_or_else(|| host_error(format!("unknown CLI reference handle {reference}")))?;
        entry.required = true;
        return_int(reference)
    })
}

/// Sets the value placeholder displayed in usage text.
#[pd_host_function(name = "flint::cli::metavar")]
pub(super) fn cli_metavar_impl(reference: i64, metavar: &str) -> VmResult<CallOutcome> {
    with_context(|context| {
        let entry = context
            .cli_references
            .get_mut(&reference)
            .ok_or_else(|| host_error(format!("unknown CLI reference handle {reference}")))?;
        entry.metavar = Some(metavar.to_owned());
        return_int(reference)
    })
}

/// Parses the current script arguments into all references attached to a parser.
#[pd_host_function(name = "flint::cli::parse_args")]
pub(super) fn cli_parse_args_impl(parser_handle: i64) -> VmResult<CallOutcome> {
    with_context(|context| {
        let parser = context
            .cli_parsers
            .get(&parser_handle)
            .cloned()
            .ok_or_else(|| host_error(format!("unknown CLI parser handle {parser_handle}")))?;
        if parser.parsed {
            return Err(host_error("CLI parser has already parsed its arguments"));
        }

        let mut options = HashMap::<String, i64>::new();
        let mut positionals = Vec::<i64>::new();
        for handle in &parser.references {
            let reference = context
                .cli_references
                .get(handle)
                .ok_or_else(|| host_error(format!("unknown CLI reference handle {handle}")))?;
            if reference.action.is_none() || reference.names.is_empty() {
                return Err(host_error(format!(
                    "CLI reference {handle} has no option or argument"
                )));
            }
            if reference.positional {
                positionals.push(*handle);
            } else {
                for name in &reference.names {
                    if options.insert(name.clone(), *handle).is_some() {
                        return Err(host_error(format!("duplicate CLI option '{name}'")));
                    }
                }
            }
        }

        let args = context.args.clone();
        let mut index = 0;
        let mut positional_index = 0;
        let mut options_enabled = true;
        while index < args.len() {
            let current = &args[index];
            if options_enabled && current == "--" {
                options_enabled = false;
                index += 1;
                continue;
            }
            if options_enabled && matches!(current.as_str(), "-h" | "--help") {
                return Err(host_error(format_help(&parser, &context.cli_references)));
            }

            let (name, inline_value) = if options_enabled && current.starts_with("--") {
                current
                    .split_once('=')
                    .map_or((current.as_str(), None), |(name, value)| {
                        (name, Some(value))
                    })
            } else {
                (current.as_str(), None)
            };

            if options_enabled && name.starts_with('-') {
                let handle = *options
                    .get(name)
                    .ok_or_else(|| host_error(format!("unrecognized option '{name}'")))?;
                let reference = context
                    .cli_references
                    .get_mut(&handle)
                    .expect("option lookup points at an existing reference");
                let action = reference.action.expect("action was checked above");
                match action {
                    Action::Store | Action::StoreOption => {
                        let raw = match inline_value {
                            Some(value) => value,
                            None => {
                                index += 1;
                                args.get(index).map(String::as_str).ok_or_else(|| {
                                    host_error(format!("option '{name}' requires a value"))
                                })?
                            }
                        };
                        set_value(reference, raw)?;
                    }
                    Action::StoreTrue => {
                        if inline_value.is_some() {
                            return Err(host_error(format!(
                                "option '{name}' does not accept a value"
                            )));
                        }
                        reference.value = Value::Bool(true);
                        reference.seen = true;
                    }
                    Action::StoreFalse => {
                        if inline_value.is_some() {
                            return Err(host_error(format!(
                                "option '{name}' does not accept a value"
                            )));
                        }
                        reference.value = Value::Bool(false);
                        reference.seen = true;
                    }
                }
            } else {
                let handle = *positionals.get(positional_index).ok_or_else(|| {
                    host_error(format!("unexpected positional argument '{current}'"))
                })?;
                let reference = context
                    .cli_references
                    .get_mut(&handle)
                    .expect("positional lookup points at an existing reference");
                set_value(reference, current)?;
                positional_index += 1;
            }
            index += 1;
        }

        for handle in &parser.references {
            let reference = context
                .cli_references
                .get(handle)
                .expect("parser points at an existing reference");
            if reference.required && !reference.seen {
                let name = reference
                    .names
                    .first()
                    .map(String::as_str)
                    .unwrap_or("VALUE");
                return Err(host_error(format!("required argument '{name}' is missing")));
            }
        }
        context
            .cli_parsers
            .get_mut(&parser_handle)
            .expect("parser was checked above")
            .parsed = true;
        return_value(Value::Bool(true))
    })
}

fn reference_value(reference: i64) -> VmResult<Value> {
    with_context(|context| {
        let reference = context
            .cli_references
            .get(&reference)
            .ok_or_else(|| host_error(format!("unknown CLI reference handle {reference}")))?;
        let parser = context
            .cli_parsers
            .get(&reference.parser)
            .expect("reference points at an existing parser");
        if !parser.parsed {
            return Err(host_error(
                "parse_args must be called before reading a reference",
            ));
        }
        Ok(reference.value.clone())
    })
}

/// Reads a parsed typed reference.
#[pd_host_function(name = "flint::cli::get")]
pub(super) fn cli_get_impl(reference: i64) -> VmResult<CallOutcome> {
    return_value(reference_value(reference)?)
}
