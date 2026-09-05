//! ADR 0038 §3.1–3.2 (#881): the deterministic sensitive-content
//! judgment `taguru extract --redact` masks with and `taguru import
//! --refuse-sensitive` refuses on. Pure functions over text — no
//! model, no dictionary, no I/O — so the same document yields the
//! same matches and the same placeholders on every run (manifests
//! and checkpoints key on that, §3.5).
//!
//! - [`RuleSet`]: the versioned built-in rule set `redact1` in two
//!   selectable groups (`secrets`, `pii`), in the order ADR 0038 §3.1
//!   lists them, optionally extended by a user's rules (#884).
//! - [`scan`]: every accepted match, paragraph by paragraph (ADR 0003
//!   §7's splitter, so no match spans a paragraph separator),
//!   ordered by start offset, then length descending, then rule
//!   order, and accepted greedily — an overlapping later candidate is
//!   dropped, so one span is one match. A value-only rule (a
//!   credential assignment, an `Authorization` header, URL userinfo)
//!   masks its value but owns its whole match: another rule's
//!   candidate inside that match is shadowed, which is how a bearer
//!   token that is also e-mail-shaped is one `authorization_header`
//!   redaction and `user:secret@host` loses its secret, not its host.
//! - [`mask`]: the redacted text and one [`Redaction`] per accepted
//!   match. The placeholder is `«redacted <rule> <hex>»` — the rule's
//!   name and a prefix of `SHA-256(document_sha256 ‖ matched bytes)`,
//!   four hex digits unless two distinct matches of one rule collide
//!   on that prefix, in which case every placeholder of that rule in
//!   the document uses the shortest prefix that tells them all apart.
//!   A placeholder the input already carries is recognised as rule
//!   `preexisting`, left byte for byte, and counted apart.
//!
//! What `redact1` deliberately does not judge (§3.1): high-entropy
//! strings (every SHA-256 in a technical document is one), names,
//! postal addresses, dates of birth, account numbers without a check
//! digit, IP addresses, and hostnames.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use regex::Regex;

/// The built-in rule set's version — a manifest input (ADR 0038 §3.5):
/// changing any built-in pattern is a new version, so an already
/// extracted document re-extracts instead of being reused.
pub(crate) const RULESET_VERSION: &str = "redact1";

/// The rule name the scanner gives a placeholder the input already
/// carries (§3.2) — never a built-in's or a user rule's name.
pub(crate) const PREEXISTING_RULE: &str = "preexisting";

/// The shortest digest prefix a placeholder carries (§3.2).
const MIN_DIGEST_HEX: usize = 4;

/// Which built-in groups are on (§3.1): `--redact` alone means both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Groups {
    pub(crate) secrets: bool,
    pub(crate) pii: bool,
}

impl Groups {
    pub(crate) const BOTH: Groups = Groups {
        secrets: true,
        pii: true,
    };

    /// The `--redact [secrets|pii]` value: absent means both.
    pub(crate) fn parse(spec: Option<&str>) -> Result<Groups, String> {
        match spec {
            None | Some("") => Ok(Groups::BOTH),
            Some("secrets") => Ok(Groups {
                secrets: true,
                pii: false,
            }),
            Some("pii") => Ok(Groups {
                secrets: false,
                pii: true,
            }),
            Some(other) => Err(format!(
                "--redact takes `secrets` or `pii` (or nothing, for both), not '{other}'"
            )),
        }
    }

    /// The manifest value's group suffix: nothing for both, `:secrets`
    /// or `:pii` for one (§3.5).
    pub(crate) fn version_suffix(self) -> &'static str {
        match (self.secrets, self.pii) {
            (true, true) => "",
            (true, false) => ":secrets",
            (false, true) => ":pii",
            (false, false) => ":none",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Group {
    Secrets,
    Pii,
}

/// One rule: what it matches, which part of the match is the secret
/// (`value_group` — a credential assignment masks its value only, so
/// `password = «…»` still reads as a configuration line), and an
/// optional check the regex cannot express (a check digit).
struct Rule {
    name: String,
    regex: Regex,
    value_group: Option<usize>,
    validate: Option<fn(&str) -> bool>,
}

struct Builtin {
    name: &'static str,
    group: Group,
    pattern: &'static str,
    value_group: Option<usize>,
    validate: Option<fn(&str) -> bool>,
}

/// The built-ins, in ADR 0038 §3.1's order — the tie-break order
/// [`scan`] applies between candidates starting at the same offset
/// with the same length.
const BUILTINS: &[Builtin] = &[
    Builtin {
        name: "aws_access_key",
        group: Group::Secrets,
        pattern: r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "github_token",
        group: Group::Secrets,
        pattern: r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "openai_key",
        group: Group::Secrets,
        pattern: r"\bsk-[A-Za-z0-9_-]{20,}\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "slack_token",
        group: Group::Secrets,
        pattern: r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "google_api_key",
        group: Group::Secrets,
        pattern: r"\bAIza[0-9A-Za-z_-]{35}\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "private_key",
        group: Group::Secrets,
        pattern: r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        value_group: None,
        validate: None,
    },
    Builtin {
        // Three base64url segments; `eyJ` is base64url for `{"`, so a
        // header and a payload that start with it decode to a JSON
        // object — the "first two decoding to `{`" test, by prefix.
        name: "jwt",
        group: Group::Secrets,
        pattern: r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "credential_assignment",
        group: Group::Secrets,
        pattern: r#"(?i)\b(?:password|passwd|secret|token|api[_-]?key)\s*[=:]\s*["']?([^\s"',;]+)"#,
        value_group: Some(1),
        validate: None,
    },
    Builtin {
        name: "authorization_header",
        group: Group::Secrets,
        pattern: r"(?i)\bAuthorization:\s*(?:Bearer|Basic)\s+(\S+)",
        value_group: Some(1),
        validate: None,
    },
    Builtin {
        // The secret only; the whole `scheme://user:secret@` match is
        // owned, so the e-mail shape `secret@host` never takes the host.
        name: "url_userinfo",
        group: Group::Secrets,
        pattern: r"\b[a-z][a-z0-9+.-]*://[^\s/:@]+:([^\s/@]+)@",
        value_group: Some(1),
        validate: None,
    },
    Builtin {
        name: "email",
        group: Group::Pii,
        pattern: r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        value_group: None,
        validate: None,
    },
    Builtin {
        // Japanese fixed (`03-1234-5678`, `(03) 1234-5678`), mobile
        // (`090-1234-5678`), and toll-free (`0120-123-456`) forms with
        // their separators, and international `+81 3 1234 5678` forms.
        // A bare digit run never matches: it is a date, a law article,
        // a price.
        name: "phone",
        group: Group::Pii,
        pattern: r"(?:\b0\d{1,4}[-‐]\d{1,4}[-‐]\d{3,4}\b|\(0\d{1,4}\)\s?\d{1,4}[-‐]\d{3,4}\b|\+\d{1,3}[- ]\d{1,4}[- ]\d{2,4}[- ]\d{3,4}\b)",
        value_group: None,
        validate: None,
    },
    Builtin {
        name: "my_number",
        group: Group::Pii,
        pattern: r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b",
        value_group: None,
        validate: Some(my_number_check_digit_holds),
    },
    Builtin {
        name: "payment_card",
        group: Group::Pii,
        pattern: r"\b\d(?:[ -]?\d){12,18}\b",
        value_group: None,
        validate: Some(payment_card_holds),
    },
];

/// The rule set a run judges with: the built-ins of the selected
/// groups, in order, then the user's rules in file order (#884).
pub(crate) struct RuleSet {
    rules: Vec<Rule>,
}

impl RuleSet {
    /// The built-ins of `groups`, compiled.
    pub(crate) fn builtin(groups: Groups) -> RuleSet {
        let rules = BUILTINS
            .iter()
            .filter(|builtin| match builtin.group {
                Group::Secrets => groups.secrets,
                Group::Pii => groups.pii,
            })
            .map(|builtin| Rule {
                name: builtin.name.to_string(),
                regex: Regex::new(builtin.pattern).expect("built-in patterns are valid"),
                value_group: builtin.value_group,
                validate: builtin.validate,
            })
            .collect();
        RuleSet { rules }
    }

    /// Whether `name` is a built-in's (a user rule may not repeat one,
    /// §3.1) — judged against every built-in, whichever groups are on.
    /// The consumer is `--redact-rules` (#884).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn is_builtin_name(name: &str) -> bool {
        BUILTINS.iter().any(|builtin| builtin.name == name)
    }

    /// The names of the rules in this set, in order.
    #[cfg(test)]
    pub(crate) fn names(&self) -> Vec<&str> {
        self.rules.iter().map(|rule| rule.name.as_str()).collect()
    }
}

/// One accepted match in the document: the rule, the paragraph it is
/// in, and the byte span that is masked (the value group's span for a
/// value-only rule). `preexisting` marks a placeholder the input
/// already carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Match {
    pub(crate) rule: String,
    pub(crate) paragraph: u32,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) preexisting: bool,
}

/// One candidate span within a paragraph, before ordering: the span
/// that would be masked, the rule's order, and — for a value-only
/// rule — the whole match it owns.
struct Candidate {
    start: usize,
    end: usize,
    order: usize,
    rule: String,
    preexisting: bool,
    owns: Option<(usize, usize)>,
}

/// Whether `text` carries a placeholder anywhere — the mechanical
/// pass's "a placeholder is not an entity" rule (ADR 0038 §3.3).
pub(crate) fn is_placeholder_bearing(text: &str) -> bool {
    placeholder_regex().is_match(text)
}

/// The placeholder form, matched to recognise a pre-existing one.
fn placeholder_regex() -> &'static Regex {
    static PLACEHOLDER: OnceLock<Regex> = OnceLock::new();
    PLACEHOLDER.get_or_init(|| {
        Regex::new(r"«redacted [a-z0-9_]+ [0-9a-f]{4,64}»").expect("valid placeholder pattern")
    })
}

/// Every accepted match in `text` (ADR 0038 §3.1's "per paragraph, in
/// a fixed order, and non-overlapping"): the scanner runs every rule
/// over one paragraph at a time, lets each value-only rule shadow the
/// other rules' candidates inside its whole match, orders what is
/// left by start offset ascending, then length descending, then rule
/// order (the pre-existing placeholder form first, so what is already
/// masked is never re-masked), and accepts greedily. Offsets are into
/// `text`.
pub(crate) fn scan(text: &str, rules: &RuleSet) -> Vec<Match> {
    let mut accepted = Vec::new();
    for span in crate::paragraph::split(text) {
        let (start, end) = (span.start as usize, span.end as usize);
        let paragraph = &text[start..end];
        let mut candidates: Vec<Candidate> = Vec::new();
        for found in placeholder_regex().find_iter(paragraph) {
            candidates.push(Candidate {
                start: found.start(),
                end: found.end(),
                order: 0,
                rule: PREEXISTING_RULE.to_string(),
                preexisting: true,
                owns: None,
            });
        }
        for (order, rule) in rules.rules.iter().enumerate() {
            for captures in rule.regex.captures_iter(paragraph) {
                let whole = captures.get(0).expect("group 0 always participates");
                let (value, owns) = match rule.value_group {
                    Some(group) => match captures.get(group) {
                        Some(value) => (value, Some((whole.start(), whole.end()))),
                        None => continue,
                    },
                    None => (whole, None),
                };
                if let Some(validate) = rule.validate
                    && !validate(value.as_str())
                {
                    continue;
                }
                candidates.push(Candidate {
                    start: value.start(),
                    end: value.end(),
                    order,
                    rule: rule.name.clone(),
                    preexisting: false,
                    owns,
                });
            }
        }
        // A value-only rule owns its whole match: another rule's
        // candidate inside it is shadowed (a pre-existing placeholder
        // never is — what is already masked stays as it is). An owner
        // is never shadowed by another owner: two owned spans that
        // overlap (`token: https://user:secret@host` — the assignment's
        // value is the URL, the URL's owned span holds the password)
        // would otherwise remove each other's candidate and leave the
        // secret in plain text; between owners the ordering below and
        // the greedy accept decide, and the earlier, longer value wins.
        let owned: Vec<(String, usize, usize)> = candidates
            .iter()
            .filter_map(|candidate| {
                candidate
                    .owns
                    .map(|(from, to)| (candidate.rule.clone(), from, to))
            })
            .collect();
        candidates.retain(|candidate| {
            candidate.preexisting
                || candidate.owns.is_some()
                || !owned.iter().any(|(rule, from, to)| {
                    *rule != candidate.rule && candidate.start < *to && candidate.end > *from
                })
        });
        // Start ascending, then length descending — which, once the
        // starts tie, is end descending — then rule order (a
        // pre-existing placeholder shares order 0 with the first rule,
        // but the two can never tie on a span: the placeholder form
        // matches no rule).
        candidates.sort_by(|a, b| {
            a.start
                .cmp(&b.start)
                .then(b.end.cmp(&a.end))
                .then(a.order.cmp(&b.order))
        });
        let mut taken_until = 0usize;
        for candidate in candidates {
            if candidate.start < taken_until {
                continue;
            }
            if candidate.end <= candidate.start {
                continue;
            }
            taken_until = candidate.end;
            accepted.push(Match {
                rule: candidate.rule,
                paragraph: span.index,
                start: start + candidate.start,
                end: start + candidate.end,
                preexisting: candidate.preexisting,
            });
        }
    }
    accepted
}

/// One redaction the mask made, for the accounting (ADR 0038 §3.6):
/// rule, paragraph, the placeholder written, and the matched length —
/// never the matched text.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct Redaction {
    pub(crate) rule: String,
    pub(crate) paragraph: u32,
    pub(crate) placeholder: String,
    pub(crate) bytes: usize,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub(crate) preexisting: bool,
}

/// The redacted text and its redactions (ADR 0038 §3.2). Each accepted
/// match is replaced by `«redacted <rule> <hex>»`, the hex a prefix of
/// `SHA-256(document_sha256 ‖ matched bytes)`: four digits, or per
/// rule the shortest prefix that tells that rule's distinct matches
/// apart, so the same secret reads as one placeholder throughout and
/// two secrets never share one. A pre-existing placeholder is kept
/// byte for byte and recorded with itself as the placeholder. The
/// replacement holds no newline, so paragraph numbering is unchanged.
pub(crate) fn mask(text: &str, document_sha256: &str, rules: &RuleSet) -> (String, Vec<Redaction>) {
    let matches = scan(text, rules);
    // Digest every distinct matched string per rule, then choose the
    // prefix length per rule from the complete set.
    let mut digests: BTreeMap<&str, BTreeMap<&str, String>> = BTreeMap::new();
    for found in matches.iter().filter(|found| !found.preexisting) {
        let matched = &text[found.start..found.end];
        digests
            .entry(found.rule.as_str())
            .or_default()
            .entry(matched)
            .or_insert_with(|| {
                let mut salted = document_sha256.as_bytes().to_vec();
                salted.extend_from_slice(matched.as_bytes());
                crate::sha256::sha256_hex(&salted)
            });
    }
    let prefix_len: BTreeMap<&str, usize> = digests
        .iter()
        .map(|(rule, by_match)| {
            let all: Vec<&str> = by_match.values().map(String::as_str).collect();
            (*rule, distinguishing_prefix(&all))
        })
        .collect();

    let mut redacted = String::with_capacity(text.len());
    let mut redactions = Vec::with_capacity(matches.len());
    let mut cursor = 0usize;
    for found in &matches {
        redacted.push_str(&text[cursor..found.start]);
        let matched = &text[found.start..found.end];
        let placeholder = if found.preexisting {
            matched.to_string()
        } else {
            let digest = &digests[found.rule.as_str()][matched];
            format!(
                "«redacted {} {}»",
                found.rule,
                &digest[..prefix_len[found.rule.as_str()]]
            )
        };
        redacted.push_str(&placeholder);
        redactions.push(Redaction {
            rule: found.rule.clone(),
            paragraph: found.paragraph,
            placeholder,
            bytes: matched.len(),
            preexisting: found.preexisting,
        });
        cursor = found.end;
    }
    redacted.push_str(&text[cursor..]);
    (redacted, redactions)
}

/// The shortest prefix length, at least [`MIN_DIGEST_HEX`], under
/// which every digest in `digests` is distinct — up to the full
/// digest, which is distinct by construction for distinct inputs.
fn distinguishing_prefix(digests: &[&str]) -> usize {
    let full = digests
        .iter()
        .map(|digest| digest.len())
        .max()
        .unwrap_or(MIN_DIGEST_HEX);
    for len in MIN_DIGEST_HEX..full {
        let mut seen: Vec<&str> = digests.iter().map(|digest| &digest[..len]).collect();
        seen.sort_unstable();
        if seen.windows(2).all(|pair| pair[0] != pair[1]) {
            return len;
        }
    }
    full
}

/// 個人番号's check digit (総務省令): with P1..P11 the first eleven
/// digits counted from the right and Qn = n+1 for n ≤ 6, n−5 for
/// n ≥ 7, the twelfth digit is 11 − (Σ Pn·Qn mod 11), or 0 when that
/// is 10 or 11.
fn my_number_check_digit_holds(candidate: &str) -> bool {
    let digits: Vec<u32> = candidate.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() != 12 {
        return false;
    }
    let sum: u32 = digits[..11]
        .iter()
        .rev()
        .enumerate()
        .map(|(index, digit)| {
            let n = index as u32 + 1;
            let q = if n <= 6 { n + 1 } else { n - 5 };
            digit * q
        })
        .sum();
    let remainder = sum % 11;
    let expected = if remainder <= 1 { 0 } else { 11 - remainder };
    digits[11] == expected
}

/// A payment card number: 13–19 digits in a known issuer range (Visa
/// 4, Mastercard 51–55 and 2221–2720, American Express 34/37, JCB
/// 3528–3589, Discover 6011/65) whose Luhn check digit holds.
fn payment_card_holds(candidate: &str) -> bool {
    let digits: Vec<u32> = candidate.chars().filter_map(|c| c.to_digit(10)).collect();
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    let lead = |n: usize| -> u32 { digits[..n].iter().fold(0, |acc, d| acc * 10 + d) };
    let known_issuer = digits[0] == 4
        || (51..=55).contains(&lead(2))
        || (2221..=2720).contains(&lead(4))
        || matches!(lead(2), 34 | 37)
        || (3528..=3589).contains(&lead(4))
        || lead(4) == 6011
        || lead(2) == 65;
    if !known_issuer {
        return false;
    }
    let luhn: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(index, digit)| {
            if index % 2 == 1 {
                // The doubled digit with its own digits summed.
                const DOUBLED: [u32; 10] = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9];
                DOUBLED[*digit as usize]
            } else {
                *digit
            }
        })
        .sum();
    luhn.is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn names_of(matches: &[Match]) -> Vec<&str> {
        matches.iter().map(|found| found.rule.as_str()).collect()
    }

    fn matched<'a>(text: &'a str, found: &Match) -> &'a str {
        &text[found.start..found.end]
    }

    #[test]
    fn groups_parse_and_version_suffix() {
        assert_eq!(Groups::parse(None).unwrap(), Groups::BOTH);
        assert_eq!(Groups::parse(Some("")).unwrap(), Groups::BOTH);
        assert_eq!(
            Groups::parse(Some("secrets")).unwrap(),
            Groups {
                secrets: true,
                pii: false
            }
        );
        assert_eq!(
            Groups::parse(Some("pii")).unwrap(),
            Groups {
                secrets: false,
                pii: true
            }
        );
        assert!(Groups::parse(Some("all")).unwrap_err().contains("'all'"));
        assert_eq!(Groups::BOTH.version_suffix(), "");
        assert_eq!(
            Groups::parse(Some("secrets")).unwrap().version_suffix(),
            ":secrets"
        );
        assert_eq!(Groups::parse(Some("pii")).unwrap().version_suffix(), ":pii");
        assert_eq!(RULESET_VERSION, "redact1");
    }

    #[test]
    fn the_built_in_set_is_in_the_adrs_order_and_filters_by_group() {
        assert_eq!(
            RuleSet::builtin(Groups::BOTH).names(),
            vec![
                "aws_access_key",
                "github_token",
                "openai_key",
                "slack_token",
                "google_api_key",
                "private_key",
                "jwt",
                "credential_assignment",
                "authorization_header",
                "url_userinfo",
                "email",
                "phone",
                "my_number",
                "payment_card",
            ]
        );
        assert_eq!(
            RuleSet::builtin(Groups::parse(Some("pii")).unwrap()).names(),
            vec!["email", "phone", "my_number", "payment_card"]
        );
        assert_eq!(
            RuleSet::builtin(Groups::parse(Some("secrets")).unwrap())
                .names()
                .len(),
            10
        );
        assert!(RuleSet::is_builtin_name("email"));
        assert!(!RuleSet::is_builtin_name("employee_id"));
        assert!(!RuleSet::is_builtin_name(PREEXISTING_RULE));
    }

    #[test]
    fn secrets_match_by_shape_and_value_only_rules_mask_the_value() {
        let rules = RuleSet::builtin(Groups::BOTH);
        let text = "key AKIAIOSFODNN7EXAMPLE and ghp_abcdefghijklmnopqrstuvwxyz0123456789 \
                    and sk-abcdefghijklmnopqrstuvwxyz and xoxb-1234567890-abcdef \
                    and AIzaSyA1234567890abcdefghijklmnopqrstuv";
        let found = scan(text, &rules);
        assert_eq!(
            names_of(&found),
            vec![
                "aws_access_key",
                "github_token",
                "openai_key",
                "slack_token",
                "google_api_key"
            ]
        );
        assert_eq!(matched(text, &found[0]), "AKIAIOSFODNN7EXAMPLE");

        let text = "password = \"hunter2\"; api_key: abc123def; token=t0ken,\n\
                    Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig_sig_sig\n\
                    https://user:s3cr3t@example.com/path";
        let found = scan(text, &rules);
        let pairs: Vec<(&str, &str)> = found
            .iter()
            .map(|f| (f.rule.as_str(), matched(text, f)))
            .collect();
        // The bearer value is JWT-shaped and the userinfo secret is
        // e-mail-shaped: the value-only rules own their whole match,
        // so each is one redaction under the header's / the URL's
        // rule, and the URL keeps its host.
        assert_eq!(
            pairs,
            vec![
                ("credential_assignment", "hunter2"),
                ("credential_assignment", "abc123def"),
                ("credential_assignment", "t0ken"),
                (
                    "authorization_header",
                    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sig_sig_sig"
                ),
                ("url_userinfo", "s3cr3t"),
            ]
        );
        let text = "Authorization: Basic dXNlcjpwYXNz\nhttps://user:p4ss@example.com/";
        let found = scan(text, &rules);
        let pairs: Vec<(&str, &str)> = found
            .iter()
            .map(|f| (f.rule.as_str(), matched(text, f)))
            .collect();
        assert_eq!(
            pairs,
            vec![
                ("authorization_header", "dXNlcjpwYXNz"),
                ("url_userinfo", "p4ss"),
            ]
        );
        // A JWT on its own, and a private-key block as one match.
        let text = "token is eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghij and\n\
                    -----BEGIN RSA PRIVATE KEY-----\nMIIB\nAAAA\n-----END RSA PRIVATE KEY-----\ndone";
        let found = scan(text, &rules);
        // `token is …` has no `=`/`:`, so it is not an assignment — the
        // JWT shape names it; the key block is one match.
        assert_eq!(names_of(&found), vec!["jwt", "private_key"]);
        assert!(
            matched(text, &found[1]).starts_with("-----BEGIN")
                && matched(text, &found[1]).ends_with("KEY-----")
        );
        let text = "bare eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghij here";
        assert_eq!(names_of(&scan(text, &rules)), vec!["jwt"]);
    }

    #[test]
    fn pii_rules_need_a_shape_or_a_check_digit() {
        let rules = RuleSet::builtin(Groups::parse(Some("pii")).unwrap());
        let text = "mail a.b+c@example.co.jp tel 03-1234-5678 or (03) 1234-5678 or 090-1234-5678 \
                    or 0120-123-456 or +81 3 1234 5678";
        let found = scan(text, &rules);
        assert_eq!(
            names_of(&found),
            vec!["email", "phone", "phone", "phone", "phone", "phone"]
        );
        // Bare digit runs are not phones: a date, a law article, a price.
        let text = "2026-09-05 第二条 12345678 ¥1234567890 0312345678";
        assert!(scan(text, &rules).is_empty(), "{:?}", scan(text, &rules));
        // Known-good card numbers pass Luhn; one digit off fails; an
        // unknown issuer range fails even with a valid Luhn.
        for card in [
            "4111 1111 1111 1111",
            "5555-5555-5555-4444",
            "378282246310005",
            "3530111333300000",
            "6011111111111117",
            "6500000000000002",
        ] {
            assert_eq!(
                names_of(&scan(card, &rules)),
                vec!["payment_card"],
                "{card}"
            );
        }
        // (A 12-digit tail of a card-like run can still be a valid
        // 個人番号 shape, so the assertion is "no card", not "nothing".)
        assert!(!names_of(&scan("4111 1111 1111 1112", &rules)).contains(&"payment_card"));
        assert!(
            !names_of(&scan("9111111111111110", &rules)).contains(&"payment_card"),
            "no issuer range"
        );
        // 個人番号: the check digit decides; the same digits with a
        // wrong last digit are left alone.
        let valid = valid_my_number("12345678901");
        assert_eq!(names_of(&scan(&valid, &rules)), vec!["my_number"]);
        let spaced = format!("{} {} {}", &valid[..4], &valid[4..8], &valid[8..]);
        assert_eq!(names_of(&scan(&spaced, &rules)), vec!["my_number"]);
        let last = valid.chars().last().unwrap().to_digit(10).unwrap();
        let wrong = format!("{}{}", &valid[..11], (last + 1) % 10);
        assert!(scan(&wrong, &rules).is_empty(), "{wrong}");
    }

    /// Appends the check digit the 総務省令 formula yields.
    fn valid_my_number(eleven: &str) -> String {
        for check in 0..10u32 {
            let candidate = format!("{eleven}{check}");
            if my_number_check_digit_holds(&candidate) {
                return candidate;
            }
        }
        unreachable!("one of ten digits is the check digit")
    }

    #[test]
    fn my_number_check_digit_follows_the_formula() {
        // P1..P11 from the right of 12345678901: 1,0,9,8,7,6,5,4,3,2,1;
        // Q = 2,3,4,5,6,7,2,3,4,5,6 → Σ = 2+0+36+40+42+42+10+12+12+10+6 = 212;
        // 212 mod 11 = 3 → check = 11 − 3 = 8.
        assert!(my_number_check_digit_holds("123456789018"));
        assert!(!my_number_check_digit_holds("123456789017"));
        assert!(!my_number_check_digit_holds("12345678901"));
        // A remainder of 0 or 1 yields a check digit of 0, not 11 or 10.
        let zero = valid_my_number("00000000000");
        assert_eq!(zero, "000000000000");
    }

    #[test]
    fn matching_is_per_paragraph_ordered_and_non_overlapping() {
        let rules = RuleSet::builtin(Groups::BOTH);
        // A bearer token that is also e-mail-shaped is ONE
        // authorization redaction (start offset ties: longer first,
        // then rule order — and the value span of the header rule
        // starts where the e-mail does).
        let text = "Authorization: Bearer a@b.co";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["authorization_header"]);
        assert_eq!(matched(text, &found[0]), "a@b.co");
        // A key block a blank line has split is two matches, one per
        // paragraph, and neither crosses the separator.
        let text = "-----BEGIN PRIVATE KEY-----\nAAAA\n\n-----END PRIVATE KEY-----";
        let found = scan(text, &rules);
        assert!(
            found.is_empty(),
            "the split block has no BEGIN…END within one paragraph: {found:?}"
        );
        let text = "x\n\n-----BEGIN PRIVATE KEY-----\nAAAA\n-----END PRIVATE KEY-----\n\ny";
        let found = scan(text, &rules);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].paragraph, 1);
        // Paragraph numbering: a match in the third paragraph says so.
        let text = "one\n\ntwo\n\nmail me@example.com";
        assert_eq!(scan(text, &rules)[0].paragraph, 2);
    }

    /// Two value-only rules whose owned spans overlap do not shadow
    /// each other into nothing: the earlier, longer value is masked,
    /// and it holds the other's secret.
    #[test]
    fn value_only_rules_never_shadow_each_other_away() {
        let rules = RuleSet::builtin(Groups::parse(Some("secrets")).unwrap());
        // The assignment's value is the whole URL; the URL's userinfo
        // owns a span inside it.
        let text = "token: https://user:s3cr3t@host";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["credential_assignment"]);
        assert_eq!(matched(text, &found[0]), "https://user:s3cr3t@host");
        // The header's value is an assignment.
        let text = "Authorization: Bearer api_key=xxx";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["authorization_header"]);
        assert_eq!(matched(text, &found[0]), "api_key=xxx");
        // Masked, neither secret survives.
        let (masked, _) = mask("token: https://user:s3cr3t@host", "d", &rules);
        assert!(!masked.contains("s3cr3t"), "{masked}");
    }

    /// A value-only rule's whole match shadows what lies INSIDE it and
    /// nothing that merely touches its edges: a candidate starting
    /// exactly where the owned match ends, or ending exactly where it
    /// begins, is still its own match.
    #[test]
    fn shadowing_covers_the_owned_span_but_not_its_edges() {
        let rules = RuleSet::builtin(Groups::parse(Some("secrets")).unwrap());
        // Inside: the key in the userinfo secret is the secret, once.
        let text = "https://u:AKIAIOSFODNN7EXAMPLE@h";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["url_userinfo"]);
        assert_eq!(matched(text, &found[0]), "AKIAIOSFODNN7EXAMPLE");
        // Starting exactly at the owned match's end (right after `@`).
        let text = "https://u:p@AKIAIOSFODNN7EXAMPLE";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["url_userinfo", "aws_access_key"]);
        // Ending exactly at the owned match's start (the key block's
        // closing dashes run into the scheme).
        let text = "-----BEGIN PRIVATE KEY-----\nX\n-----END PRIVATE KEY-----https://u:p@h";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["private_key", "url_userinfo"]);
        assert_eq!(matched(text, &found[1]), "p");
        // Two accepted spans that touch: the key starts on the byte the
        // key block ends on, and both are taken.
        let text = "-----BEGIN PRIVATE KEY-----\nX\n-----END PRIVATE KEY-----AKIAIOSFODNN7EXAMPLE";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec!["private_key", "aws_access_key"]);
        assert_eq!(found[0].end, found[1].start);
    }

    #[test]
    fn mask_writes_stable_placeholders_and_records_no_content() {
        let rules = RuleSet::builtin(Groups::BOTH);
        let text = "k1 AKIAIOSFODNN7EXAMPLE again AKIAIOSFODNN7EXAMPLE other AKIAI44QH8DHBEXAMPLE\n\nmail me@example.com";
        let (redacted, redactions) = mask(text, DOC, &rules);
        assert!(!redacted.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(!redacted.contains("me@example.com"));
        assert_eq!(redactions.len(), 4);
        // The same key reads as the same placeholder; a different key
        // as a different one; the e-mail under its own rule.
        assert_eq!(redactions[0].placeholder, redactions[1].placeholder);
        assert_ne!(redactions[0].placeholder, redactions[2].placeholder);
        assert!(
            redactions[0]
                .placeholder
                .starts_with("«redacted aws_access_key ")
        );
        assert!(redactions[3].placeholder.starts_with("«redacted email "));
        assert_eq!(redactions[3].paragraph, 1);
        assert_eq!(redactions[0].bytes, "AKIAIOSFODNN7EXAMPLE".len());
        assert!(!redactions[0].preexisting);
        // Four hex digits when nothing collides; no newline inside.
        assert_eq!(
            redactions[0].placeholder.len(),
            "«redacted aws_access_key 0000»".len()
        );
        assert!(!redactions.iter().any(|r| r.placeholder.contains('\n')));
        // Paragraph structure is intact.
        assert_eq!(crate::paragraph::split(&redacted).len(), 2);
        // Deterministic: the same document hash and text give the same
        // output; a different document hash gives different digits.
        assert_eq!(mask(text, DOC, &rules), mask(text, DOC, &rules));
        let other = "1111111111111111111111111111111111111111111111111111111111111111";
        assert_ne!(
            mask(text, other, &rules).1[0].placeholder,
            redactions[0].placeholder
        );
        // The record never carries the matched text.
        let json = serde_json::to_string(&redactions).unwrap();
        assert!(!json.contains("EXAMPLE") && !json.contains("example.com"));
        assert!(!json.contains("preexisting"), "false is omitted: {json}");
    }

    #[test]
    fn a_same_rule_prefix_collision_extends_every_placeholder_of_that_rule() {
        let rules = RuleSet::builtin(Groups::parse(Some("secrets")).unwrap());
        // Search for two distinct AWS keys whose salted digests share
        // their first four hex digits under DOC — deterministic, so the
        // search always lands on the same pair.
        let base = "AKIAIOSFODNN7EXAMPLE";
        let salted = |key: &str| {
            let mut bytes = DOC.as_bytes().to_vec();
            bytes.extend_from_slice(key.as_bytes());
            crate::sha256::sha256_hex(&bytes)
        };
        let base_prefix = salted(base)[..4].to_string();
        let alphabet: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars().collect();
        let mut colliding = None;
        'outer: for a in &alphabet {
            for b in &alphabet {
                for c in &alphabet {
                    for d in &alphabet {
                        let key = format!("AKIA{a}{b}{c}{d}IOSFODNN7EXA");
                        if key != base && salted(&key)[..4] == base_prefix {
                            colliding = Some(key);
                            break 'outer;
                        }
                    }
                }
            }
        }
        let colliding = colliding.expect("a 16-bit collision within 1.7M candidates");
        let text = format!("{base} and {colliding} and {base}");
        let (redacted, redactions) = mask(&text, DOC, &rules);
        assert_eq!(redactions.len(), 3);
        assert_ne!(redactions[0].placeholder, redactions[1].placeholder);
        assert_eq!(redactions[0].placeholder, redactions[2].placeholder);
        let digits = |placeholder: &str| {
            placeholder
                .trim_end_matches('»')
                .rsplit(' ')
                .next()
                .unwrap()
                .len()
        };
        assert!(digits(&redactions[0].placeholder) > 4);
        assert_eq!(
            digits(&redactions[0].placeholder),
            digits(&redactions[1].placeholder)
        );
        assert!(!redacted.contains(base) && !redacted.contains(&colliding));
        // A third, non-colliding key of the same rule gets the same
        // (extended) width — the length is per rule per document.
        let text = format!("{base} {colliding} AKIAI44QH8DHBEXAMPLE");
        let (_, redactions) = mask(&text, DOC, &rules);
        assert_eq!(
            digits(&redactions[2].placeholder),
            digits(&redactions[0].placeholder)
        );
        assert_eq!(distinguishing_prefix(&["abcd1", "abce2"]), 4);
        assert_eq!(distinguishing_prefix(&["abcd1", "abcd2"]), 5);
        assert_eq!(
            distinguishing_prefix(&["abcd", "abcd"]),
            4,
            "identical digests: the full length"
        );
        assert_eq!(distinguishing_prefix(&[]), 4);
    }

    #[test]
    fn a_preexisting_placeholder_is_kept_and_counted_apart() {
        let rules = RuleSet::builtin(Groups::BOTH);
        let text = "was «redacted aws_access_key 1a2b» and is AKIAIOSFODNN7EXAMPLE";
        let found = scan(text, &rules);
        assert_eq!(names_of(&found), vec![PREEXISTING_RULE, "aws_access_key"]);
        assert!(found[0].preexisting && !found[1].preexisting);
        let (redacted, redactions) = mask(text, DOC, &rules);
        assert!(
            redacted
                .starts_with("was «redacted aws_access_key 1a2b» and is «redacted aws_access_key ")
        );
        assert_eq!(redactions[0].placeholder, "«redacted aws_access_key 1a2b»");
        assert!(redactions[0].preexisting);
        assert_eq!(redactions[0].bytes, "«redacted aws_access_key 1a2b»".len());
        // Masking the masked text again changes nothing and reports
        // every placeholder as pre-existing.
        let (again, redactions) = mask(&redacted, DOC, &rules);
        assert_eq!(again, redacted);
        assert!(redactions.iter().all(|r| r.preexisting));
        assert!(
            serde_json::to_string(&redactions[0])
                .unwrap()
                .contains("\"preexisting\":true")
        );
    }

    #[test]
    fn a_clean_document_is_returned_unchanged() {
        let rules = RuleSet::builtin(Groups::BOTH);
        let text = "青嶺酒造の杜氏は高瀬。\n\nsha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 is a hash, \
                    10.0.0.1 an address, and 2026-09-05 a date.";
        assert!(scan(text, &rules).is_empty());
        let (redacted, redactions) = mask(text, DOC, &rules);
        assert_eq!(redacted, text);
        assert!(redactions.is_empty());
    }
}
