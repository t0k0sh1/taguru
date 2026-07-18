use serde_json::Value;

/// Percent-encodes text into the unreserved-only RFC 3986 set — safe
/// both as one URL path segment and as a query-string value.
pub(super) fn segment(name: &str) -> String {
    name.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

/// Pulls a required string argument, telling an absent one apart from a
/// present-but-wrong-typed one — folding both into "missing" sends a
/// caller who passed `{"name": 42}` hunting for an argument they did
/// supply.
pub(super) fn need<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    match arguments.get(key) {
        Some(Value::String(text)) => Ok(text),
        Some(Value::Null) | None => Err(format!("missing required argument '{key}'")),
        Some(_) => Err(format!("argument '{key}' must be a string")),
    }
}

/// `need`'s missing/null-counts-as-missing rule for a required
/// argument that isn't a string — an array or object body field, like
/// `add_associations`'s `associations`. Type checking past "present"
/// stays server-side, same as it already does for every argument
/// `pick` copies through untyped.
pub(super) fn need_present<'a>(arguments: &'a Value, key: &str) -> Result<&'a Value, String> {
    match arguments.get(key) {
        Some(value) if !value.is_null() => Ok(value),
        _ => Err(format!("missing required argument '{key}'")),
    }
}

/// Pulls an optional boolean argument, falling back to `default` when
/// absent — but, like `need`, refusing to silently coerce a
/// present-but-wrong-typed value. `.and_then(Value::as_bool).unwrap_or(default)`
/// treats a caller's typo'd `"false"` (a JSON string, not a bool) exactly
/// like never passing the argument at all, silently keeping `default`
/// instead of surfacing the mistake — most dangerous for a knob whose
/// default is `true`, where the caller's intent was plainly to turn it off.
pub(super) fn optional_bool(arguments: &Value, key: &str, default: bool) -> Result<bool, String> {
    match arguments.get(key) {
        Some(Value::Bool(value)) => Ok(*value),
        Some(Value::Null) | None => Ok(default),
        Some(_) => Err(format!("argument '{key}' must be a boolean")),
    }
}

/// Copies the listed keys into a request body, skipping absent ones.
pub(super) fn pick(arguments: &Value, keys: &[&str]) -> Value {
    let mut body = serde_json::Map::new();
    for &key in keys {
        if let Some(value) = arguments.get(key)
            && !value.is_null()
        {
            body.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(body)
}

/// Builds a `?a=1&b=2` query string from the listed keys, skipping
/// absent/null ones — the GET-request counterpart of `pick`, for tools
/// that carry no body. Numbers pass through their JSON text; strings
/// are percent-encoded. Like `need`/`optional_bool`, a present value of
/// the wrong type is refused rather than silently dropped: erasing the
/// key here would remove it before any request is built, so unlike
/// `pick`'s untyped copy-through, nothing downstream ever gets a chance
/// to reject the mistake.
pub(super) fn query_string(arguments: &Value, keys: &[&str]) -> Result<String, String> {
    let mut pairs = Vec::new();
    for &key in keys {
        let Some(value) = arguments.get(key).filter(|value| !value.is_null()) else {
            continue;
        };
        let text = match value {
            Value::String(text) => segment(text),
            Value::Number(number) => number.to_string(),
            Value::Bool(boolean) => boolean.to_string(),
            _ => {
                return Err(format!(
                    "argument '{key}' must be a string, number, or boolean"
                ));
            }
        };
        pairs.push(format!("{key}={text}"));
    }
    Ok(if pairs.is_empty() {
        String::new()
    } else {
        format!("?{}", pairs.join("&"))
    })
}

/// Like `pick`, but a value under `alias` counts for `canonical` when
/// `canonical` itself is absent — request-side back-compat for an argument
/// renamed after clients had already adopted the old name.
pub(super) fn pick_with_alias(
    arguments: &Value,
    keys: &[&str],
    canonical: &str,
    alias: &str,
) -> Value {
    let mut body = pick(arguments, keys);
    if let Value::Object(map) = &mut body
        && !map.contains_key(canonical)
        && let Some(value) = arguments.get(alias)
        && !value.is_null()
    {
        map.insert(canonical.to_string(), value.clone());
    }
    body
}
