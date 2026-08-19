//! `RouteMap`: parses and queries the context→shard map that
//! `TAGURU_ROUTE_MAP` names.

use super::*;

/// The context→shard map, parsed from `TAGURU_ROUTE_MAP`. Shards are
/// deduped by URL in file order; `fallback` is the `*` entry.
#[derive(Debug)]
pub(crate) struct RouteMap {
    pub(super) shards: Vec<String>,
    pub(super) contexts: BTreeMap<String, usize>,
    pub(super) fallback: Option<usize>,
}

impl RouteMap {
    /// One `context = url` per line (`*` for the fallback), `#`
    /// comments and blank lines ignored — the same boring dialect as
    /// the config file. Refused whole on the first malformed line:
    /// a route map with a silent hole misroutes forever.
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let mut shards: Vec<String> = Vec::new();
        let mut contexts = BTreeMap::new();
        let mut fallback = None;
        let shard_index = |url: &str, shards: &mut Vec<String>| -> usize {
            let url = url.trim_end_matches('/').to_string();
            match shards.iter().position(|known| *known == url) {
                Some(index) => index,
                None => {
                    shards.push(url);
                    shards.len() - 1
                }
            }
        };
        for (number, line) in text.lines().enumerate() {
            let number = number + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((name, url)) = line.split_once('=') else {
                return Err(format!(
                    "line {number}: expected 'context = shard-url' (or '* = shard-url')"
                ));
            };
            let (name, url) = (name.trim(), url.trim());
            if name.is_empty() {
                return Err(format!("line {number}: the context name is empty"));
            }
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(format!(
                    "line {number}: '{url}' is not an http(s) shard URL"
                ));
            }
            // Shard URLs surface verbatim in logs, error messages, and
            // /metrics labels, so a URL carrying userinfo would leak
            // its credentials there — refused at the door instead.
            // The message deliberately does NOT echo the URL: the boot
            // and reload paths print this error.
            let authority = &url[url.find("//").expect("scheme checked above") + 2..];
            let head = authority.split(['/', '?', '#']).next().unwrap_or_default();
            if head.contains('@') {
                return Err(format!(
                    "line {number}: the shard URL carries userinfo ('user@host') — shard \
                     URLs appear in logs and metrics labels, so credentials are refused"
                ));
            }
            // The HOST alone: an IPv6 literal keeps its brackets,
            // anything else ends at the port colon — so a bare scheme
            // ('http://', 'http:///path') AND a port-only authority
            // ('http://:8248') both refuse here, with the line number,
            // instead of storing a shard no dial can ever reach. The
            // message deliberately does not echo the URL: a hostless
            // spelling can still carry a secret in its query, and this
            // error prints at boot and reload (the same posture as the
            // userinfo refusal above).
            let host = match head.strip_prefix('[') {
                Some(rest) => rest.split(']').next().unwrap_or_default(),
                None => head.split(':').next().unwrap_or_default(),
            };
            if host.is_empty() {
                return Err(format!("line {number}: the shard URL names no host"));
            }
            if name == "*" {
                if fallback.is_some() {
                    return Err(format!("line {number}: '*' fallback given twice"));
                }
                fallback = Some(shard_index(url, &mut shards));
            } else if contexts
                .insert(name.to_string(), shard_index(url, &mut shards))
                .is_some()
            {
                return Err(format!("line {number}: context '{name}' is mapped twice"));
            }
        }
        if shards.is_empty() {
            return Err("the route map names no shards".to_string());
        }
        Ok(Self {
            shards,
            contexts,
            fallback,
        })
    }

    pub(super) fn shard_of(&self, context: &str) -> Option<usize> {
        self.contexts.get(context).copied().or(self.fallback)
    }

    pub(super) fn all(&self) -> impl Iterator<Item = usize> {
        0..self.shards.len()
    }

    pub(super) fn url(&self, shard: usize) -> &str {
        &self.shards[shard]
    }

    /// The map's member-list projection for one shard — what a group
    /// write sends there. A member no shard owns keeps flowing to the
    /// owning-shard check downstream, which refuses it exactly as a
    /// single instance refuses a nonexistent member.
    pub(super) fn project<'a>(
        &self,
        members: impl IntoIterator<Item = &'a str>,
        shard: usize,
    ) -> Vec<String> {
        members
            .into_iter()
            .filter(|name| self.shard_of(name) == Some(shard))
            .map(str::to_string)
            .collect()
    }
}
