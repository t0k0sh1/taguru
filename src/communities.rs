//! `taguru communities`: derives a community-summaries artifact from a
//! RUNNING server's context (issue #166) — the orchestration half of
//! the community verbs.
//!
//! The flow is deliberately client-side and HTTP-only, calibrate's
//! model: the data-directory lock is exclusive with no read-only mode,
//! so a verb that must coexist with a live server (or a replica, or a
//! router) has exactly one door. Detection runs ON the server
//! (`GET /contexts/{name}/communities` — the graph never crosses the
//! wire raw); this command diffs the result against the previous
//! artifact, asks the LLM for summaries of what actually changed, and
//! writes the artifact back through `POST /import` as an ordinary
//! context.
//!
//! Incremental by fingerprint, not by wish: a community whose
//! fingerprint matches the previous manifest reuses its stored summary
//! verbatim — zero LLM calls on an unchanged graph — and only changed
//! or new communities are summarized. Vanished community sources are
//! retracted afterwards, and the manifest batch goes LAST so a torn
//! run leaves a manifest that honestly mismatches (and re-derives)
//! rather than one that claims batches that never landed.
//!
//! Summaries ride the extract provider (`TAGURU_EXTRACT_URL/MODEL/
//! API_KEY` — one wire shape for every LLM the system talks to), which
//! is only contacted when something actually needs summarizing:
//! `--dry-run` and no-change re-runs work with no extract env at all.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::api::communities::{
    COMMUNITIES_FORMAT, COMMUNITY_SOURCE_PREFIX, CONTAINS_LABEL, CommunitiesManifest,
    INCLUDES_LABEL, MANIFEST_SOURCE, ManifestCommunity, derived_context_name,
};
use crate::config::{load_config, subcommand_usage_error};
use crate::registry::ContextRevision;
use crate::remote::default_base_url;
use crate::remote::{Api, ApiFailure};

const COMMUNITIES_USAGE: &str = "usage: taguru communities --context NAME [--into NAME] [--dry-run]
                           [--json] [--config FILE] [--url URL] [URL]
       taguru communities --group NAME [--dry-run] [--json] [--config FILE]
                           [--url URL] [URL]

Derives (or refreshes) a community-summaries artifact from a RUNNING
server's context: the server detects communities on the association
graph, this command summarizes each one with the extract LLM and
writes the result back as an ordinary context (default
'NAME::communities') — membership and hierarchy as associations,
summaries as passages, and a manifest recording the source revision
the artifact was derived from. `POST /contexts/{name}/communities/search`
then serves ranked summaries with an honest staleness verdict.

Incremental: a community whose content fingerprint is unchanged reuses
its stored summary — a re-run over an unchanged graph makes no LLM
calls at all. --dry-run reports what would change without calling the
LLM or writing anything.

--group NAME derives one artifact per member context (child groups
included, transitively) — there is no cross-context graph merge.

Summaries use the extract provider: TAGURU_EXTRACT_URL,
TAGURU_EXTRACT_MODEL, TAGURU_EXTRACT_API_KEY (docs/extract.html) —
required only when something actually needs summarizing. Auth rides
the same variables the server reads: TAGURU_API_TOKEN, or the first
key of TAGURU_API_TOKENS. --url and the positional URL are aliases —
name the target either way, never both; unnamed, it defaults to
TAGURU_ADDR after --config applies, exactly like `taguru health`.

exit codes: 0 artifact up to date (or dry-run report produced) ·
1 derivation failed · 2 usage error
";

/// Members quoted in a leaf summary prompt — the strongest carry the
/// theme; the tail would only dilute the prompt.
const PROMPT_MEMBERS: usize = 40;

/// Characters of each child summary quoted in a parent prompt.
const PROMPT_CHILD_EXCERPT: usize = 500;

/// A membership edge for a member with no intra-community strength (a
/// singleton community): weight 0.0 would read as a netted-out,
/// contested claim, which membership is not.
const SINGLETON_MEMBER_WEIGHT: f64 = 1.0;

pub fn run(args: &[String]) -> i32 {
    let usage = |message: &str| subcommand_usage_error("communities", message);
    let mut context: Option<String> = None;
    let mut group: Option<String> = None;
    let mut into: Option<String> = None;
    let mut config: Option<PathBuf> = None;
    let mut dry_run = false;
    let mut as_json = false;
    let mut explicit_url: Option<String> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{COMMUNITIES_USAGE}");
                return 0;
            }
            "--context" => match rest.next() {
                Some(name) if context.is_none() => context = Some(name.clone()),
                Some(_) => return usage("--context given twice"),
                None => return usage("--context needs a name"),
            },
            "--group" => match rest.next() {
                Some(name) if group.is_none() => group = Some(name.clone()),
                Some(_) => return usage("--group given twice"),
                None => return usage("--group needs a name"),
            },
            "--into" => match rest.next() {
                Some(name) if into.is_none() => into = Some(name.clone()),
                Some(_) => return usage("--into given twice"),
                None => return usage("--into needs a context name"),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => return usage("--config given twice"),
                None => return usage("--config needs a file path"),
            },
            "--url" => match rest.next() {
                Some(url) if explicit_url.is_none() && !url.starts_with('-') => {
                    explicit_url = Some(url.trim_end_matches('/').to_string());
                }
                Some(_) if explicit_url.is_none() => {
                    return usage("--url needs a server URL");
                }
                Some(_) => {
                    return usage("either --url or a positional URL, not both");
                }
                None => return usage("--url needs a server URL"),
            },
            "--dry-run" => dry_run = true,
            "--json" => as_json = true,
            flag if flag.starts_with('-') => {
                return usage(&format!("unknown argument '{flag}'"));
            }
            url => {
                if explicit_url
                    .replace(url.trim_end_matches('/').to_string())
                    .is_some()
                {
                    return usage("either --url or a positional URL, not both");
                }
            }
        }
    }
    match (&context, &group) {
        (Some(_), Some(_)) => return usage("--context and --group are mutually exclusive"),
        (None, None) => return usage("--context NAME (or --group NAME) is required"),
        _ => {}
    }
    if into.is_some() && group.is_some() {
        return usage("--into names one artifact; with --group each member gets its own");
    }

    // The config file first, then the URL default off the (possibly
    // just-loaded) environment — the order every client verb resolves
    // in, so one --config deployment file aims them all at one port.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match explicit_url {
        Some(url) => url,
        None => match default_base_url() {
            Ok(url) => url,
            Err(error) => {
                eprintln!("taguru: communities: {error}");
                return 2;
            }
        },
    };
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(&base) {
        return crate::config::subcommand_usage_error("communities", &message);
    }
    // `reject_userinfo` deliberately leaves an unparsable `base` alone;
    // refuse it outright here instead (issue #248 item 8, matching
    // `evaluate`/`benchmark search`/`calibrate`'s own fix, issue #289 /
    // #281 / #288) rather than letting a malformed string reach `Api`.
    if url::Url::parse(&base).is_err() {
        return crate::config::subcommand_usage_error(
            "communities",
            "the URL could not be parsed as a URL",
        );
    }
    let api = Api::new(base);

    let contexts = match &group {
        None => vec![context.expect("checked above")],
        Some(group) => match group_members(&api, group) {
            Ok(members) if members.is_empty() => {
                eprintln!("taguru: communities: group '{group}' reaches no contexts");
                return 1;
            }
            Ok(members) => members,
            Err(error) => {
                eprintln!("taguru: communities: {error}");
                return 1;
            }
        },
    };

    let mut failed = false;
    let mut reports = Vec::new();
    for name in &contexts {
        let derived = into.clone().unwrap_or_else(|| derived_context_name(name));
        match derive(&api, name, &derived, dry_run) {
            Ok(report) => {
                if !as_json {
                    print!("{}", report.render());
                }
                reports.push(report);
            }
            Err(error) => {
                // A group run reports every member; one failure does
                // not hide the rest, but it does fail the run.
                eprintln!("taguru: communities: {name}: {error}");
                failed = true;
            }
        }
    }
    if as_json {
        match serde_json::to_string_pretty(&reports) {
            Ok(text) => println!("{text}"),
            Err(error) => {
                eprintln!("taguru: communities: report did not serialize: {error}");
                return 1;
            }
        }
    }
    if failed { 1 } else { 0 }
}

/// One context's derivation, start to finish.
fn derive(api: &Api, name: &str, derived: &str, dry_run: bool) -> Result<Report, String> {
    // The analysis stream carries the revision snapshot the server cut
    // it at — that, not a separately-read revision, is what the
    // manifest records.
    let stream = api.get_raw(&["contexts", name, "communities"])?;
    let analysis = parse_analysis(&stream)?;

    // The previous manifest, if an artifact exists: the fingerprint
    // ledger this run diffs against. An algorithm change invalidates
    // every fingerprint — incomparable digests must not "match".
    let previous = read_manifest(api, derived)?;
    let comparable = previous
        .as_ref()
        .is_some_and(|manifest| manifest.algorithm == analysis.header.algorithm);
    let mut previous_by_fingerprint: BTreeMap<&str, &ManifestCommunity> = BTreeMap::new();
    if let Some(manifest) = previous.as_ref().filter(|_| comparable) {
        for community in &manifest.communities {
            previous_by_fingerprint.insert(community.fingerprint.as_str(), community);
        }
    }

    // Split reuse from fresh work. Reused summaries are fetched in one
    // lookup; fresh ones go to the LLM, leaves before parents so a
    // parent prompt can quote its children.
    let mut reused_sources: Vec<String> = Vec::new();
    let mut fresh = 0usize;
    for community in &analysis.communities {
        match previous_by_fingerprint.get(community.fingerprint.as_str()) {
            Some(old) => reused_sources.push(format!("{COMMUNITY_SOURCE_PREFIX}{}", old.id)),
            None => fresh += 1,
        }
    }
    // Vanished = a previous id no NEW community carries. Ids present
    // in both runs are rewritten wholesale by their import batch
    // (replace-per-source), whatever their fingerprints did — judging
    // by fingerprint here would retract a freshly rewritten source.
    let new_ids: BTreeSet<&str> = analysis
        .communities
        .iter()
        .map(|community| community.id.as_str())
        .collect();
    let vanished: Vec<String> = previous
        .as_ref()
        .map(|manifest| {
            manifest
                .communities
                .iter()
                .filter(|community| !new_ids.contains(community.id.as_str()))
                .map(|community| community.id.clone())
                .collect()
        })
        .unwrap_or_default();

    let mut report = Report {
        context: name.to_string(),
        derived: derived.to_string(),
        dry_run,
        algorithm: analysis.header.algorithm.clone(),
        revision_graph: analysis.header.revision.graph,
        concept_count: analysis.header.concept_count,
        edge_count: analysis.header.edge_count,
        levels: analysis.header.levels,
        communities: analysis.communities.len(),
        summaries_generated: fresh,
        summaries_reused: reused_sources.len(),
        retracted: vanished.len(),
        rebuilt_from_scratch: previous.is_some() && !comparable,
    };
    if dry_run {
        return Ok(report);
    }

    // Old summaries for the reused set, one round trip. A torn earlier
    // run can leave the manifest promising summaries the store no
    // longer holds — those re-summarize below, so they need the LLM
    // exactly like fresh communities do.
    let mut old_texts: BTreeMap<String, String> = BTreeMap::new();
    if !reused_sources.is_empty() {
        old_texts = lookup_passages(api, derived, &reused_sources)?;
    }
    let torn = reused_sources
        .iter()
        .filter(|source| !old_texts.contains_key(*source))
        .count();

    // Summaries, leaves first: parents quote their children. The chat
    // client exists exactly when something below will call it — a
    // clean unchanged graph still re-runs with no extract env at all.
    let chat = if fresh + torn > 0 {
        Some(
            crate::extract::ChatClient::from_env()
                .map_err(|error| format!("{error} (only --dry-run works without it)"))?,
        )
    } else {
        None
    };
    let mut summaries: BTreeMap<&str, String> = BTreeMap::new();
    let mut ordered: Vec<&AnalysisCommunity> = analysis.communities.iter().collect();
    ordered.sort_by_key(|community| community.level);
    for community in ordered {
        let text = match previous_by_fingerprint.get(community.fingerprint.as_str()) {
            Some(old) => {
                let old_source = format!("{COMMUNITY_SOURCE_PREFIX}{}", old.id);
                match old_texts.get(&old_source) {
                    Some(text) => text.clone(),
                    // The manifest promised a summary the store no
                    // longer holds (a torn earlier run) — summarize
                    // fresh rather than write an empty passage.
                    None => {
                        report.summaries_reused -= 1;
                        report.summaries_generated += 1;
                        summarize(chat.as_ref(), community, &summaries)?
                    }
                }
            }
            None => summarize(chat.as_ref(), community, &summaries)?,
        };
        summaries.insert(community.id.as_str(), text);
    }

    // The import stream: one batch per community, the manifest LAST.
    let manifest = CommunitiesManifest {
        taguru_communities: COMMUNITIES_FORMAT,
        algorithm: analysis.header.algorithm.clone(),
        source_context: name.to_string(),
        revision: analysis.header.revision,
        levels: analysis.header.levels,
        communities: analysis
            .communities
            .iter()
            .map(|community| ManifestCommunity {
                id: community.id.clone(),
                level: community.level,
                fingerprint: community.fingerprint.clone(),
                concept_count: community.concept_count,
                parent: community.parent.clone(),
            })
            .collect(),
    };
    let batches = render_batches(name, derived, &analysis, &summaries, &manifest)?;
    for chunk in crate::remote::pack_import_chunks(&batches) {
        api.import(&chunk)?;
    }

    // Vanished communities go last: at every earlier failure point the
    // old sources still exist and the (old or new) manifest accounts
    // for them.
    for id in &vanished {
        retract_source(api, derived, &format!("{COMMUNITY_SOURCE_PREFIX}{id}"))?;
    }
    Ok(report)
}

/// One community's summary: leaves from their members and strongest
/// relations, parents from their children's summaries. `chat` is only
/// `None` when nothing needs the LLM — no fresh communities and no
/// torn reuse — and then this is never called.
fn summarize(
    chat: Option<&crate::extract::ChatClient>,
    community: &AnalysisCommunity,
    summaries: &BTreeMap<&str, String>,
) -> Result<String, String> {
    let chat = chat.expect("fresh + torn counted before the loop");
    let system = "You are summarizing one cluster of a knowledge graph so it can be \
                  found and read later. Answer in the language most of the facts are \
                  written in. Prose only — no preamble, no headings, no lists.";
    let user = if community.level == 0 {
        let members: Vec<&str> = community
            .members
            .iter()
            .take(PROMPT_MEMBERS)
            .map(|member| member.name.as_str())
            .collect();
        let relations: Vec<String> = community
            .top_associations
            .iter()
            .map(|association| {
                format!(
                    "{} —{}→ {} (weight {:.2}, ×{})",
                    association.subject,
                    association.label,
                    association.object,
                    association.weight,
                    association.count,
                )
            })
            .collect();
        format!(
            "One community of {} concepts. Central concepts, strongest first: {}.\n\
             Strongest relations:\n{}\n\
             Write 2-4 short paragraphs. The first sentence names the community's \
             theme. Then state what the relations establish — a negative weight is a \
             contested or negated claim and must be reported as such, not dropped.",
            community.concept_count,
            members.join(", "),
            relations.join("\n"),
        )
    } else {
        let children: Vec<String> = community
            .children
            .iter()
            .filter_map(|child| summaries.get(child.as_str()))
            .map(|summary| {
                let excerpt: String = summary.chars().take(PROMPT_CHILD_EXCERPT).collect();
                format!("- {excerpt}")
            })
            .collect();
        format!(
            "One community grouping {} sub-communities ({} concepts in all). Their \
             summaries:\n{}\n\
             Write 1-3 short paragraphs. The first sentence names the shared theme; \
             then what distinguishes the subgroups.",
            community.children.len(),
            community.concept_count,
            children.join("\n"),
        )
    };
    let response = chat.complete(
        &[
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user}),
        ],
        &crate::extract::RequestOptions::default(),
    )?;
    let text = response.content.trim();
    if text.is_empty() {
        return Err(format!(
            "the model answered an empty summary for community {}",
            community.id
        ));
    }
    Ok(text.to_string())
}

/// The artifact as an import stream: one batch per community (summary
/// passage + membership/hierarchy edges), the manifest batch last.
fn render_batches(
    name: &str,
    derived: &str,
    analysis: &Analysis,
    summaries: &BTreeMap<&str, String>,
    manifest: &CommunitiesManifest,
) -> Result<Vec<String>, String> {
    let render = |value: &Value| -> Result<String, String> {
        serde_json::to_string(value).map_err(|error| format!("batch line: {error}"))
    };
    let mut batches = Vec::with_capacity(analysis.communities.len() + 1);
    let mut first = true;
    for community in &analysis.communities {
        let source = format!("{COMMUNITY_SOURCE_PREFIX}{}", community.id);
        let mut header = json!({
            "taguru_batch": 1,
            "context": derived,
            "source": source,
        });
        if first {
            // Applied only if the artifact context does not exist yet.
            header["create"] = json!({
                "description": format!(
                    "Community analysis of '{name}' (taguru communities)"
                ),
            });
            first = false;
        }
        let mut lines = vec![render(&header)?];
        let summary = summaries
            .get(community.id.as_str())
            .ok_or_else(|| format!("no summary for community {}", community.id))?;
        lines.push(render(&json!({"passage": summary}))?);
        for member in &community.members {
            let weight = if member.strength > 0.0 {
                member.strength.min(1e6)
            } else {
                SINGLETON_MEMBER_WEIGHT
            };
            lines.push(render(&json!({
                "subject": source,
                "label": CONTAINS_LABEL,
                "object": member.name,
                "weight": weight,
            }))?);
        }
        for child in &community.children {
            lines.push(render(&json!({
                "subject": source,
                "label": INCLUDES_LABEL,
                "object": format!("{COMMUNITY_SOURCE_PREFIX}{child}"),
                "weight": 1.0,
            }))?);
        }
        batches.push(lines.join("\n"));
    }
    let manifest_text =
        serde_json::to_string(manifest).map_err(|error| format!("manifest: {error}"))?;
    batches.push(
        [
            render(&json!({
                "taguru_batch": 1,
                "context": derived,
                "source": MANIFEST_SOURCE,
            }))?,
            render(&json!({"passage": manifest_text}))?,
        ]
        .join("\n"),
    );
    Ok(batches)
}

/// The previous artifact's manifest: `None` when the artifact context
/// or its manifest record does not exist yet (a first run), an error
/// only for a manifest that exists but does not parse — that needs a
/// human, not a silent full rebuild.
fn read_manifest(api: &Api, derived: &str) -> Result<Option<CommunitiesManifest>, String> {
    let body = json!({"sources": [MANIFEST_SOURCE]});
    let result = match api.post_envelope(&["contexts", derived, "sources", "lookup"], &body) {
        Ok(result) => result,
        Err(ApiFailure::NotFound { .. }) => return Ok(None),
        Err(ApiFailure::Other(error)) => return Err(error),
    };
    let Some(text) = result["passages"][MANIFEST_SOURCE].as_str() else {
        return Ok(None);
    };
    serde_json::from_str(text).map(Some).map_err(|error| {
        format!(
            "the previous '{MANIFEST_SOURCE}' record in '{derived}' does not parse \
                 ({error}) — delete the artifact context to rebuild from scratch"
        )
    })
}

/// A group's transitive member contexts, child groups included —
/// depth is server-capped, and cycles are refused at write time, so a
/// plain recursion cannot run away.
fn group_members(api: &Api, group: &str) -> Result<Vec<String>, String> {
    let mut contexts = BTreeSet::new();
    let mut pending = vec![group.to_string()];
    let mut seen = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let entry = api
            .get_envelope(&["groups", &name])
            .map_err(|failure| failure.into_message())?;
        for context in entry["contexts"].as_array().into_iter().flatten() {
            if let Some(context) = context.as_str() {
                contexts.insert(context.to_string());
            }
        }
        for child in entry["groups"].as_array().into_iter().flatten() {
            if let Some(child) = child.as_str() {
                pending.push(child.to_string());
            }
        }
    }
    Ok(contexts.into_iter().collect())
}

/// What one derivation did — one line of the human report each, the
/// whole struct under --json.
#[derive(Serialize)]
struct Report {
    context: String,
    derived: String,
    dry_run: bool,
    algorithm: String,
    revision_graph: u64,
    concept_count: usize,
    edge_count: usize,
    levels: usize,
    communities: usize,
    summaries_generated: usize,
    summaries_reused: usize,
    retracted: usize,
    /// True when a previous artifact existed but its algorithm differs
    /// — every fingerprint was incomparable and everything re-derived.
    rebuilt_from_scratch: bool,
}

impl Report {
    fn render(&self) -> String {
        let verb = if self.dry_run {
            "would generate"
        } else {
            "generated"
        };
        let mut text = format!(
            "context '{}': {} communities across {} levels ({} concepts, {} edges)\n  \
             summaries: {} {verb}, {} reused",
            self.context,
            self.communities,
            self.levels,
            self.concept_count,
            self.edge_count,
            self.summaries_generated,
            self.summaries_reused,
        );
        if self.retracted > 0 {
            text.push_str(&format!(
                " · {} vanished communit{} {}",
                self.retracted,
                if self.retracted == 1 { "y" } else { "ies" },
                if self.dry_run {
                    "would retract"
                } else {
                    "retracted"
                },
            ));
        }
        if self.rebuilt_from_scratch {
            text.push_str(" · algorithm changed: previous fingerprints incomparable");
        }
        text.push_str(&format!(
            "\n  artifact '{}' {} at graph revision {}\n",
            self.derived,
            if self.dry_run {
                "would update"
            } else {
                "updated"
            },
            self.revision_graph,
        ));
        text
    }
}

/// The analysis stream, parsed: header line plus one community per
/// line (the wire shape of `GET /contexts/{name}/communities`).
struct Analysis {
    header: AnalysisHeader,
    communities: Vec<AnalysisCommunity>,
}

#[derive(Deserialize)]
struct AnalysisHeader {
    taguru_communities: u64,
    algorithm: String,
    revision: ContextRevision,
    concept_count: usize,
    edge_count: usize,
    levels: usize,
    communities: usize,
}

#[derive(Deserialize)]
struct AnalysisCommunity {
    id: String,
    level: usize,
    #[serde(default)]
    parent: Option<String>,
    fingerprint: String,
    concept_count: usize,
    #[serde(default)]
    members: Vec<AnalysisMember>,
    #[serde(default)]
    children: Vec<String>,
    #[serde(default)]
    top_associations: Vec<AnalysisAssociation>,
}

#[derive(Deserialize)]
struct AnalysisMember {
    name: String,
    strength: f64,
}

#[derive(Deserialize)]
struct AnalysisAssociation {
    subject: String,
    label: String,
    object: String,
    weight: f64,
    count: u64,
}

fn parse_analysis(stream: &str) -> Result<Analysis, String> {
    let mut lines = stream.lines().filter(|line| !line.trim().is_empty());
    let header: AnalysisHeader = match lines.next() {
        None => return Err("the analysis stream is empty".to_string()),
        Some(line) => serde_json::from_str(line)
            .map_err(|error| format!("analysis header unreadable: {error}"))?,
    };
    if header.taguru_communities != COMMUNITIES_FORMAT {
        return Err(format!(
            "analysis format {} is newer than this taguru understands ({}) — upgrade the CLI",
            header.taguru_communities, COMMUNITIES_FORMAT,
        ));
    }
    let communities: Vec<AnalysisCommunity> = lines
        .map(|line| {
            serde_json::from_str(line).map_err(|error| format!("analysis line unreadable: {error}"))
        })
        .collect::<Result<_, _>>()?;
    if communities.len() != header.communities {
        return Err(format!(
            "the analysis stream is torn: header names {} communities, body carries {}",
            header.communities,
            communities.len(),
        ));
    }
    Ok(Analysis {
        header,
        communities,
    })
}

/// This verb's domain helpers over the shared HTTP door — the
/// `contexts/{}/sources/...` path knowledge lives here, not in
/// `Api` itself.
fn lookup_passages(
    api: &Api,
    context: &str,
    sources: &[String],
) -> Result<BTreeMap<String, String>, String> {
    let body = json!({"sources": sources});
    let result = api
        .post_envelope(&["contexts", context, "sources", "lookup"], &body)
        .map_err(|failure| failure.into_message())?;
    let mut passages = BTreeMap::new();
    if let Some(map) = result["passages"].as_object() {
        for (source, text) in map {
            if let Some(text) = text.as_str() {
                passages.insert(source.clone(), text.to_string());
            }
        }
    }
    Ok(passages)
}

fn retract_source(api: &Api, context: &str, source: &str) -> Result<(), String> {
    api.post_envelope(
        &["contexts", context, "sources", "retract"],
        &json!({"source": source}),
    )
    .map(|_| ())
    .map_err(|failure| failure.into_message())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_analysis_refuses_a_torn_stream_and_a_newer_format() {
        let header = r#"{"taguru_communities":1,"context":"c","algorithm":"louvain-cc/1","revision":{"graph":3,"passages":0,"config":0},"concept_count":2,"edge_count":1,"levels":1,"communities":1}"#;
        let line = r#"{"id":"L0-0","level":0,"fingerprint":"00","concept_count":2}"#;

        let parsed = parse_analysis(&format!("{header}\n{line}\n")).unwrap();
        assert_eq!(parsed.header.revision.graph, 3);
        assert_eq!(parsed.communities.len(), 1);
        assert!(parse_analysis(header).is_err());

        let newer = header.replace("\"taguru_communities\":1", "\"taguru_communities\":2");
        assert!(parse_analysis(&format!("{newer}\n{line}\n")).is_err());
    }

    #[test]
    fn reports_render_the_dry_run_conditionals() {
        let report = Report {
            context: "sake".to_string(),
            derived: "sake::communities".to_string(),
            dry_run: true,
            algorithm: "louvain-cc/1".to_string(),
            revision_graph: 7,
            concept_count: 10,
            edge_count: 20,
            levels: 2,
            communities: 3,
            summaries_generated: 2,
            summaries_reused: 1,
            retracted: 1,
            rebuilt_from_scratch: false,
        };
        let text = report.render();
        assert!(text.contains("would generate"));
        assert!(text.contains("would retract"));
        assert!(text.contains("would update"));
        assert!(text.contains("graph revision 7"));
    }

    /// Issue #753: a member with no intra-community strength (a
    /// singleton) gets [`SINGLETON_MEMBER_WEIGHT`] — weight 0.0 would
    /// read as a netted-out claim, which membership is not — while a
    /// runaway positive strength is capped at 1e6.
    #[test]
    fn a_zero_strength_member_lands_at_the_singleton_weight() {
        let header = r#"{"taguru_communities":1,"context":"c","algorithm":"louvain-cc/1","revision":{"graph":1,"passages":0,"config":0},"concept_count":2,"edge_count":1,"levels":1,"communities":1}"#;
        let line = r#"{"id":"L0-0","level":0,"fingerprint":"00","concept_count":2,"members":[{"name":"solo","strength":0.0},{"name":"heavy","strength":2e7}]}"#;
        let analysis = parse_analysis(&format!("{header}\n{line}\n")).unwrap();
        let summaries = BTreeMap::from([("L0-0", "要約".to_string())]);
        let manifest = CommunitiesManifest {
            taguru_communities: COMMUNITIES_FORMAT,
            algorithm: "louvain-cc/1".to_string(),
            source_context: "c".to_string(),
            revision: ContextRevision::default(),
            levels: 1,
            communities: Vec::new(),
        };
        let batches = render_batches("c", "c::communities", &analysis, &summaries, &manifest)
            .expect("render");
        let community_batch = &batches[0];
        let weights: Vec<(String, f64)> = community_batch
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .filter(|value| value["label"] == json!(CONTAINS_LABEL))
            .map(|value| {
                (
                    value["object"].as_str().unwrap().to_string(),
                    value["weight"].as_f64().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            weights,
            [
                ("solo".to_string(), SINGLETON_MEMBER_WEIGHT),
                ("heavy".to_string(), 1e6),
            ]
        );
    }
}
