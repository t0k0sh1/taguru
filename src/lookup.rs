//! `taguru contexts` — find a context's id from its display name
//! (issue #967, ADR 0045 §2.4: finding by name is a listing operation).
//!
//! Every CLI verb that points at an existing context takes its id
//! (`--context ID`); the id is minted by the server and a person only
//! knows the name they gave it. This verb is the bridge: it lists the
//! directory (`GET /contexts`) and, with `--name`, keeps the rows whose
//! name matches exactly. Names are not unique, so zero, one, or several
//! rows can come back — the caller picks, nothing here guesses.

use std::path::PathBuf;

use serde::Serialize;

use crate::config::{load_config, subcommand_usage_error};
use crate::registry::DirectoryEntry;
use crate::remote::{Api, default_base_url};

const USAGE: &str =
    "usage: taguru contexts [--name NAME] [--json] [--config FILE] [--url URL] [URL]

Lists the contexts of a RUNNING server as `ID<TAB>NAME` lines, sorted by
name — the id column is what every other verb's --context takes. With
--name, only the contexts whose name is exactly NAME are listed (names
are not unique: several lines can come back, and then you pick by id).

    taguru contexts --name sake                  # → the id(s) of 'sake'
    taguru extract --context \"$(taguru contexts --name sake | cut -f1)\" ...

--json prints [{\"id\", \"name\", \"description\"}, ...] instead. Auth rides
the same variables the server reads: TAGURU_API_TOKEN, or the first key
of TAGURU_API_TOKENS; a scoped key lists only what its grant reaches.
--url and the positional URL are aliases — name the target either way,
never both; unnamed, it defaults to TAGURU_ADDR after --config applies,
exactly like `taguru health`.

exit codes: 0 listed (with --name: at least one match) · 1 the server
could not be read, or --name matched nothing · 2 usage error
";

/// One listed context — `--json`'s exact element shape.
#[derive(Debug, PartialEq, Serialize)]
struct Row {
    id: String,
    name: String,
    description: String,
}

#[derive(Debug, PartialEq)]
struct Args {
    name: Option<String>,
    json: bool,
    config: Option<PathBuf>,
    url: Option<String>,
}

#[derive(Debug, PartialEq)]
enum Parsed {
    Help,
    Run(Args),
}

fn parse_args(args: &[String]) -> Result<Parsed, String> {
    let mut parsed = Args {
        name: None,
        json: false,
        config: None,
        url: None,
    };
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(Parsed::Help),
            "--name" => match rest.next() {
                Some(name) if parsed.name.is_none() => parsed.name = Some(name.clone()),
                Some(_) => return Err("--name given twice".to_string()),
                None => return Err("--name needs a context name".to_string()),
            },
            "--config" => match rest.next() {
                Some(path) if parsed.config.is_none() => parsed.config = Some(PathBuf::from(path)),
                Some(_) => return Err("--config given twice".to_string()),
                None => return Err("--config needs a file path".to_string()),
            },
            "--url" => match rest.next() {
                Some(url) if parsed.url.is_none() && !url.starts_with('-') => {
                    parsed.url = Some(url.trim_end_matches('/').to_string());
                }
                Some(_) if parsed.url.is_none() => {
                    return Err("--url needs a server URL".to_string());
                }
                Some(_) => return Err("either --url or a positional URL, not both".to_string()),
                None => return Err("--url needs a server URL".to_string()),
            },
            "--json" => parsed.json = true,
            flag if flag.starts_with('-') => return Err(format!("unknown argument '{flag}'")),
            url => {
                if parsed
                    .url
                    .replace(url.trim_end_matches('/').to_string())
                    .is_some()
                {
                    return Err("either --url or a positional URL, not both".to_string());
                }
            }
        }
    }
    Ok(Parsed::Run(parsed))
}

/// The rows to show: every entry, or with `name` only those whose name
/// equals it exactly (byte-for-byte — no case folding, no trimming: a
/// lookup that guessed would hand back the wrong id).
fn select(entries: Vec<DirectoryEntry>, name: Option<&str>) -> Vec<Row> {
    entries
        .into_iter()
        .filter(|entry| name.is_none_or(|wanted| entry.name == wanted))
        .map(|entry| Row {
            id: entry.id,
            name: entry.name,
            description: entry.description,
        })
        .collect()
}

/// `ID<TAB>NAME` per row. A name is free-form UTF-8, so control
/// characters (a tab or newline would split the line and misalign
/// `cut -f1`) are shown as spaces — `--json` carries the exact text.
fn render_lines(rows: &[Row]) -> String {
    rows.iter()
        .map(|row| {
            let name: String = row
                .name
                .chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect();
            format!("{}\t{name}\n", row.id)
        })
        .collect()
}

pub fn run(args: &[String]) -> i32 {
    let usage = |message: &str| subcommand_usage_error("contexts", message);
    let args = match parse_args(args) {
        Ok(Parsed::Help) => {
            print!("{USAGE}");
            return 0;
        }
        Ok(Parsed::Run(args)) => args,
        Err(message) => return usage(&message),
    };

    let config = args
        .config
        .or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match args.url {
        Some(url) => url,
        None => match default_base_url() {
            Ok(url) => url,
            Err(error) => {
                eprintln!("taguru: contexts: {error}");
                return 2;
            }
        },
    };
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(&base) {
        return usage(&message);
    }

    let entries = match Api::new(base).list_context_entries() {
        Ok(entries) => entries,
        Err(message) => {
            eprintln!("taguru: contexts: {message}");
            return 1;
        }
    };
    let rows = select(entries, args.name.as_deref());
    if let Some(wanted) = &args.name
        && rows.is_empty()
    {
        eprintln!("taguru: contexts: no context is named '{wanted}'");
        return 1;
    }
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).expect("rows serialize")
        );
    } else {
        print!("{}", render_lines(&rows));
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| arg.to_string()).collect()
    }

    fn entry(id: &str, name: &str) -> DirectoryEntry {
        serde_json::from_value(serde_json::json!({
            "id": id, "name": name, "description": format!("about {name}"),
            "pinned": false, "loaded": false,
            "dice_floor": null, "semantic_floor": null,
            "stats": {
                "concepts": 0, "labels": 0, "associations": 0, "sources": 0,
                "dead_edges": 0, "arena_slack": 0, "top_concepts": [], "labels_sample": []
            },
            "usage": {
                "reads": 0, "empty_reads": 0, "writes": 0,
                "last_read_at": null, "last_write_at": null
            },
            "schema_mode": null
        }))
        .expect("a directory entry")
    }

    #[test]
    fn parse_takes_every_flag_and_the_positional_url() {
        let parsed = parse_args(&strings(&[
            "--name",
            "sake",
            "--json",
            "--config",
            "c.env",
            "http://h:1/",
        ]))
        .unwrap();
        assert_eq!(
            parsed,
            Parsed::Run(Args {
                name: Some("sake".to_string()),
                json: true,
                config: Some(PathBuf::from("c.env")),
                url: Some("http://h:1".to_string()),
            })
        );
        assert_eq!(
            parse_args(&strings(&["--url", "http://h:2/"])).unwrap(),
            Parsed::Run(Args {
                name: None,
                json: false,
                config: None,
                url: Some("http://h:2".to_string()),
            })
        );
        assert_eq!(parse_args(&strings(&["-h"])).unwrap(), Parsed::Help);
    }

    #[test]
    fn parse_refuses_repeats_missing_values_and_unknown_flags() {
        for (args, expected) in [
            (vec!["--name", "a", "--name", "b"], "--name given twice"),
            (vec!["--name"], "--name needs a context name"),
            (
                vec!["--config", "a", "--config", "b"],
                "--config given twice",
            ),
            (vec!["--config"], "--config needs a file path"),
            (vec!["--url"], "--url needs a server URL"),
            (vec!["--url", "--json"], "--url needs a server URL"),
            (
                vec!["--url", "http://a", "--url", "http://b"],
                "either --url or a positional URL, not both",
            ),
            (
                vec!["--url", "http://a", "http://b"],
                "either --url or a positional URL, not both",
            ),
            (
                vec!["http://a", "--url", "http://b"],
                "either --url or a positional URL, not both",
            ),
            (vec!["--bogus"], "unknown argument '--bogus'"),
        ] {
            assert_eq!(
                parse_args(&strings(&args)).unwrap_err(),
                expected,
                "{args:?}"
            );
        }
    }

    #[test]
    fn select_matches_the_name_exactly_and_keeps_every_twin() {
        let entries = vec![
            entry("11111111-1111-4111-8111-111111111111", "sake"),
            entry("22222222-2222-4222-8222-222222222222", "Sake"),
            entry("33333333-3333-4333-8333-333333333333", "sake"),
            entry("44444444-4444-4444-8444-444444444444", "sake "),
        ];
        let rows = select(entries, Some("sake"));
        assert_eq!(
            rows.iter().map(|row| row.id.as_str()).collect::<Vec<_>>(),
            [
                "11111111-1111-4111-8111-111111111111",
                "33333333-3333-4333-8333-333333333333"
            ]
        );
        assert_eq!(rows[0].description, "about sake");
    }

    #[test]
    fn select_without_a_name_lists_everything_in_order() {
        let entries = vec![
            entry("11111111-1111-4111-8111-111111111111", "a"),
            entry("22222222-2222-4222-8222-222222222222", "b"),
        ];
        assert_eq!(select(entries, None).len(), 2);
        assert!(select(vec![], Some("a")).is_empty());
    }

    #[test]
    fn lines_are_id_tab_name_with_control_characters_flattened() {
        let rows = vec![
            Row {
                id: "11111111-1111-4111-8111-111111111111".to_string(),
                name: "酒".to_string(),
                description: String::new(),
            },
            Row {
                id: "22222222-2222-4222-8222-222222222222".to_string(),
                name: "a\tb\nc\rd".to_string(),
                description: String::new(),
            },
        ];
        assert_eq!(
            render_lines(&rows),
            "11111111-1111-4111-8111-111111111111\t酒\n\
             22222222-2222-4222-8222-222222222222\ta b c d\n"
        );
        assert_eq!(render_lines(&[]), "");
    }
}
