//! `taguru contexts` / `taguru groups` — find a context's or a group's
//! id from its display name (issues #967 and #997, ADR 0045 §2.4:
//! finding by name is a listing operation).
//!
//! Every CLI verb that points at an existing context takes its id
//! (`--context ID`), and every group reference takes the group's id
//! (`--group ID`, `group_ids`); the id is minted by the server and a
//! person only knows the name they gave it. These verbs are the bridge:
//! they list the directory (`GET /contexts` or `GET /groups`) and, with
//! `--name`, keep the rows whose name matches exactly. Names are not
//! unique, so zero, one, or several rows can come back — the caller
//! picks, nothing here guesses.

use std::path::PathBuf;

use serde::Serialize;

use crate::config::{load_config, subcommand_usage_error};
use crate::remote::{Api, default_base_url};

const CONTEXTS_USAGE: &str =
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

const GROUPS_USAGE: &str =
    "usage: taguru groups [--name NAME] [--json] [--config FILE] [--url URL] [URL]

Lists the groups of a RUNNING server as `ID<TAB>NAME` lines, sorted by
name — the id column is what /groups/{id}, every group_ids list, and
`taguru communities --group` take. With --name, only the groups whose
name is exactly NAME are listed (names are not unique: several lines
can come back, and then you pick by id).

    taguru groups --name papers                  # → the id(s) of 'papers'
    taguru communities --group \"$(taguru groups --name papers | cut -f1)\"

--json prints [{\"id\", \"name\", \"description\"}, ...] instead. Auth rides
the same variables the server reads: TAGURU_API_TOKEN, or the first key
of TAGURU_API_TOKENS. --url and the positional URL are aliases — name
the target either way, never both; unnamed, it defaults to TAGURU_ADDR
after --config applies, exactly like `taguru health`.

exit codes: 0 listed (with --name: at least one match) · 1 the server
could not be read, or --name matched nothing · 2 usage error
";

/// Which directory a lookup lists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Directory {
    Contexts,
    Groups,
}

impl Directory {
    /// The verb, as typed and as every message's prefix.
    fn verb(self) -> &'static str {
        match self {
            Self::Contexts => "contexts",
            Self::Groups => "groups",
        }
    }

    /// One row's kind, for the messages that name it.
    fn noun(self) -> &'static str {
        match self {
            Self::Contexts => "context",
            Self::Groups => "group",
        }
    }

    fn usage(self) -> &'static str {
        match self {
            Self::Contexts => CONTEXTS_USAGE,
            Self::Groups => GROUPS_USAGE,
        }
    }

    /// The whole directory as rows, in the server's `(name, id)` order.
    fn list(self, api: &Api) -> Result<Vec<Row>, String> {
        Ok(match self {
            Self::Contexts => api
                .list_context_entries()?
                .into_iter()
                .map(|entry| Row {
                    id: entry.id,
                    name: entry.name,
                    description: entry.description,
                })
                .collect(),
            Self::Groups => api
                .list_group_rows()?
                .into_iter()
                .map(|row| Row {
                    id: row.id,
                    name: row.name,
                    description: row.description,
                })
                .collect(),
        })
    }
}

/// One listed context or group — `--json`'s exact element shape.
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

fn parse_args(directory: Directory, args: &[String]) -> Result<Parsed, String> {
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
                None => return Err(format!("--name needs a {} name", directory.noun())),
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

/// The rows to show: every row, or with `name` only those whose name
/// equals it exactly (byte-for-byte — no case folding, no trimming: a
/// lookup that guessed would hand back the wrong id).
fn select(rows: Vec<Row>, name: Option<&str>) -> Vec<Row> {
    rows.into_iter()
        .filter(|row| name.is_none_or(|wanted| row.name == wanted))
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

pub fn run(directory: Directory, args: &[String]) -> i32 {
    let verb = directory.verb();
    let usage = |message: &str| subcommand_usage_error(verb, message);
    let args = match parse_args(directory, args) {
        Ok(Parsed::Help) => {
            print!("{}", directory.usage());
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
                eprintln!("taguru: {verb}: {error}");
                return 2;
            }
        },
    };
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(&base) {
        return usage(&message);
    }
    if let Err(message) = crate::remote::reject_unusable_base(&base) {
        return usage(&message);
    }

    let rows = match directory.list(&Api::new(base)) {
        Ok(rows) => rows,
        Err(message) => {
            eprintln!("taguru: {verb}: {message}");
            return 1;
        }
    };
    let rows = select(rows, args.name.as_deref());
    if let Some(wanted) = &args.name
        && rows.is_empty()
    {
        eprintln!(
            "taguru: {verb}: no {} is named '{wanted}'",
            directory.noun()
        );
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

    fn row(id: &str, name: &str) -> Row {
        Row {
            id: id.to_string(),
            name: name.to_string(),
            description: format!("about {name}"),
        }
    }

    /// `--help` prints the verb's own usage — a `groups` user must not
    /// read the `contexts` page, which points at `--context`.
    #[test]
    fn each_directory_prints_its_own_usage() {
        assert!(
            Directory::Contexts
                .usage()
                .starts_with("usage: taguru contexts [--name NAME]")
        );
        let groups = Directory::Groups.usage();
        assert!(groups.starts_with("usage: taguru groups [--name NAME]"));
        assert!(groups.contains("taguru communities --group"), "{groups}");
    }

    #[test]
    fn parse_takes_every_flag_and_the_positional_url() {
        let parsed = parse_args(
            Directory::Contexts,
            &strings(&[
                "--name",
                "sake",
                "--json",
                "--config",
                "c.env",
                "http://h:1/",
            ]),
        )
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
            parse_args(Directory::Groups, &strings(&["--url", "http://h:2/"])).unwrap(),
            Parsed::Run(Args {
                name: None,
                json: false,
                config: None,
                url: Some("http://h:2".to_string()),
            })
        );
        assert_eq!(
            parse_args(Directory::Contexts, &strings(&["-h"])).unwrap(),
            Parsed::Help
        );
    }

    #[test]
    fn parse_refuses_repeats_missing_values_and_unknown_flags() {
        for (directory, args, expected) in [
            (
                Directory::Contexts,
                vec!["--name", "a", "--name", "b"],
                "--name given twice",
            ),
            (
                Directory::Contexts,
                vec!["--name"],
                "--name needs a context name",
            ),
            (
                Directory::Groups,
                vec!["--name"],
                "--name needs a group name",
            ),
            (
                Directory::Contexts,
                vec!["--config", "a", "--config", "b"],
                "--config given twice",
            ),
            (
                Directory::Contexts,
                vec!["--config"],
                "--config needs a file path",
            ),
            (
                Directory::Contexts,
                vec!["--url"],
                "--url needs a server URL",
            ),
            (
                Directory::Contexts,
                vec!["--url", "--json"],
                "--url needs a server URL",
            ),
            (
                Directory::Contexts,
                vec!["--url", "http://a", "--url", "http://b"],
                "either --url or a positional URL, not both",
            ),
            (
                Directory::Contexts,
                vec!["--url", "http://a", "http://b"],
                "either --url or a positional URL, not both",
            ),
            (
                Directory::Contexts,
                vec!["http://a", "--url", "http://b"],
                "either --url or a positional URL, not both",
            ),
            (
                Directory::Contexts,
                vec!["--bogus"],
                "unknown argument '--bogus'",
            ),
        ] {
            assert_eq!(
                parse_args(directory, &strings(&args)).unwrap_err(),
                expected,
                "{args:?}"
            );
        }
    }

    #[test]
    fn select_matches_the_name_exactly_and_keeps_every_twin() {
        let rows = vec![
            row("11111111-1111-4111-8111-111111111111", "sake"),
            row("22222222-2222-4222-8222-222222222222", "Sake"),
            row("33333333-3333-4333-8333-333333333333", "sake"),
            row("44444444-4444-4444-8444-444444444444", "sake "),
        ];
        let rows = select(rows, Some("sake"));
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
        let rows = vec![
            row("11111111-1111-4111-8111-111111111111", "a"),
            row("22222222-2222-4222-8222-222222222222", "b"),
        ];
        assert_eq!(select(rows, None).len(), 2);
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
