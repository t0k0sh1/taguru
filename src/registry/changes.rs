//! The per-context change feed's ring (#422): a bounded, in-memory
//! record of recent content changes, serving `GET
//! /contexts/{name}/changes?since=` so a polling client (a local cache,
//! an external index, a communities/evidence recomputation trigger) can
//! ask "what changed since my cursor" instead of re-listing everything.
//!
//! Deliberately NOT derived from the WAL: the per-context WAL truncates
//! on every successful image flush (once a second by default), so it
//! retains no history to serve. And deliberately not persisted: the
//! ring is runtime state with an honest failure mode — a server
//! restart, a ring overflow, or a delete-and-recreate mints a new
//! `epoch`, and a cursor from another epoch answers `stale_cursor`
//! (fall back to a full resync) rather than silently pretending
//! nothing happened in between. That contract is what keeps the ring
//! small and the on-disk format untouched.
//!
//! Granularity: one event per WRITE CALL, not per association — a
//! 10k-line import must not evict the whole ring as 10k entries.
//! `associations_added` carries a count; source-level events carry the
//! source id, which is the unit a syncing client actually re-pulls.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;

use super::{AccessError, AppState};
use crate::wal::WalOp;

impl AppState {
    /// Serves one page of the context's change feed — see
    /// [`ChangeRing::read`] for the cursor contract. Touches nothing
    /// but the ring: the context itself stays cold if it was cold,
    /// which is what makes a tight polling loop affordable.
    pub fn context_changes(
        &self,
        name: &str,
        since: Option<&str>,
        limit: usize,
    ) -> Result<ChangesOutcome, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let _fence = entry.read_unless_deleted().ok_or(AccessError::NotFound)?;
        let outcome = entry.changes.lock().read(since, limit);
        self.touch(&entry);
        Ok(outcome)
    }
}

/// How many events one context retains. A steady writer overflows this
/// eventually by design — the feed serves "poll every few seconds or
/// minutes," not "replay last week" (the replication bucket is the
/// durable history). 1024 events at one event per write CALL covers
/// hours of ordinary ingestion; a client that falls further behind gets
/// `stale_cursor` and resyncs, which is also exactly what it would have
/// to do against any retention bound.
const CHANGE_RING_CAP: usize = 1024;

/// Ring identities are process-unique: seeded from the wall clock once,
/// then incremented per ring, so a context deleted and recreated under
/// the same name (same process or not) can never validate a cursor
/// minted against its predecessor's ring.
fn next_epoch() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let seed = *SEED.get_or_init(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|since| since.as_nanos() as u64)
            .unwrap_or(1)
    });
    seed.wrapping_add(NEXT.fetch_add(1, Ordering::Relaxed))
}

/// One recorded change. `seq` is per-ring, monotonic, 1-based; the
/// event kinds mirror what a syncing client acts on, not the WAL's op
/// vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChangeEvent {
    pub seq: u64,
    #[serde(flatten)]
    pub kind: ChangeKind,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChangeKind {
    /// One write call's worth of applied association assertions —
    /// aggregated so a bulk import is one event, not one per line.
    AssociationsAdded {
        count: usize,
    },
    AssociationRetracted {
        subject: String,
        label: String,
        object: String,
    },
    /// One write call's worth of alias registrations (concepts and
    /// labels combined), aggregated like `AssociationsAdded`.
    AliasesAdded {
        count: usize,
    },
    AliasesRemoved {
        count: usize,
    },
    /// A source's passage landed (stored or replaced).
    SourceStored {
        source: String,
    },
    /// A source's contributions were withdrawn (graph side; its
    /// passage, when one existed, went with it).
    SourceRetracted {
        source: String,
    },
    /// A schema document was installed or replaced.
    SchemaUpdated {
        mode: String,
    },
}

/// What one `changes` read answers.
#[derive(Debug, PartialEq)]
pub enum ChangesOutcome {
    Page {
        events: Vec<ChangeEvent>,
        /// The cursor to pass as the next `since` — always present,
        /// even (especially) when `events` is empty.
        next: String,
        /// Whether events past `limit` remain — poll again immediately
        /// rather than waiting out the interval.
        more: bool,
    },
    /// The cursor names another epoch, or history it covers has been
    /// evicted from the ring — the client's only correct move is a
    /// full resync, and saying so beats a silently incomplete page.
    Stale,
}

pub struct ChangeRing {
    epoch: u64,
    next_seq: u64,
    events: VecDeque<ChangeEvent>,
}

impl Default for ChangeRing {
    fn default() -> Self {
        Self {
            epoch: next_epoch(),
            next_seq: 1,
            events: VecDeque::new(),
        }
    }
}

impl ChangeRing {
    pub fn push(&mut self, kind: ChangeKind) {
        if self.events.len() == CHANGE_RING_CAP {
            self.events.pop_front();
        }
        let seq = self.next_seq;
        self.next_seq += 1;
        self.events.push_back(ChangeEvent { seq, kind });
    }

    pub fn extend(&mut self, kinds: impl IntoIterator<Item = ChangeKind>) {
        for kind in kinds {
            self.push(kind);
        }
    }

    /// Serves one page after `since` (`None` = "start tailing now":
    /// no events, just the cursor to poll from). `limit` must be > 0.
    pub fn read(&self, since: Option<&str>, limit: usize) -> ChangesOutcome {
        let latest = self.next_seq - 1;
        let last_seen = match since {
            None => {
                return ChangesOutcome::Page {
                    events: Vec::new(),
                    next: self.cursor(latest),
                    more: false,
                };
            }
            Some(cursor) => match parse_cursor(cursor) {
                Some((epoch, seq)) if epoch == self.epoch => seq,
                // Another ring's cursor — a restart, a recreate, or a
                // malformed string; all of them mean "your position is
                // meaningless here."
                _ => return ChangesOutcome::Stale,
            },
        };
        if last_seen > latest {
            // A cursor from this ring's future: impossible to have been
            // minted honestly, so treat it like any other lost position.
            return ChangesOutcome::Stale;
        }
        if let Some(front) = self.events.front()
            && last_seen + 1 < front.seq
        {
            // The events after the cursor were already evicted — part
            // of the answer is gone, so the whole answer is.
            return ChangesOutcome::Stale;
        }
        let events: Vec<ChangeEvent> = self
            .events
            .iter()
            .filter(|event| event.seq > last_seen)
            .take(limit)
            .cloned()
            .collect();
        let served_to = events.last().map_or(last_seen, |event| event.seq);
        ChangesOutcome::Page {
            more: served_to < latest,
            next: self.cursor(served_to),
            events,
        }
    }

    fn cursor(&self, seq: u64) -> String {
        format_cursor(self.epoch, seq)
    }
}

/// Cursors are opaque to clients, versioned for the server's own
/// benefit: `cf1-{epoch:016x}-{seq}`.
fn format_cursor(epoch: u64, seq: u64) -> String {
    format!("cf1-{epoch:016x}-{seq}")
}

fn parse_cursor(cursor: &str) -> Option<(u64, u64)> {
    let rest = cursor.strip_prefix("cf1-")?;
    let (epoch, seq) = rest.split_once('-')?;
    Some((u64::from_str_radix(epoch, 16).ok()?, seq.parse().ok()?))
}

/// The change events one applied prefix of a `logged_write` batch
/// means: `Associate`/alias ops aggregate to per-call counts, the
/// retract ops stay individual (they carry the identity a client acts
/// on). Order within the batch is preserved where it matters —
/// aggregates are emitted after the individual events of the same
/// batch would only reorder within one write call, so aggregates come
/// first for determinism.
pub fn events_of_ops(ops: &[WalOp]) -> Vec<ChangeKind> {
    let mut associations = 0usize;
    let mut aliases_added = 0usize;
    let mut aliases_removed = 0usize;
    let mut individual = Vec::new();
    for op in ops {
        match op {
            WalOp::Associate(_) => associations += 1,
            WalOp::AliasConcept { .. } | WalOp::AliasLabel { .. } => aliases_added += 1,
            WalOp::UnaliasConcept { .. } | WalOp::UnaliasLabel { .. } => aliases_removed += 1,
            WalOp::RetractSource { source } => individual.push(ChangeKind::SourceRetracted {
                source: source.clone(),
            }),
            WalOp::RetractAssociation {
                subject,
                label,
                object,
            } => individual.push(ChangeKind::AssociationRetracted {
                subject: subject.clone(),
                label: label.clone(),
                object: object.clone(),
            }),
        }
    }
    let mut events = Vec::new();
    if associations > 0 {
        events.push(ChangeKind::AssociationsAdded {
            count: associations,
        });
    }
    if aliases_added > 0 {
        events.push(ChangeKind::AliasesAdded {
            count: aliases_added,
        });
    }
    if aliases_removed > 0 {
        events.push(ChangeKind::AliasesRemoved {
            count: aliases_removed,
        });
    }
    events.extend(individual);
    events
}

#[cfg(test)]
mod tests {
    use super::*;

    fn added(count: usize) -> ChangeKind {
        ChangeKind::AssociationsAdded { count }
    }

    #[test]
    fn tailing_without_a_cursor_returns_only_the_position() {
        let mut ring = ChangeRing::default();
        ring.push(added(3));
        let ChangesOutcome::Page { events, next, more } = ring.read(None, 10) else {
            panic!("tailing is never stale");
        };
        assert!(events.is_empty());
        assert!(!more);
        // The position resumes AFTER everything that already happened.
        let ChangesOutcome::Page { events, .. } = ring.read(Some(&next), 10) else {
            panic!("a freshly minted cursor is never stale");
        };
        assert!(events.is_empty());
    }

    #[test]
    fn reads_resume_exactly_after_the_cursor() {
        let mut ring = ChangeRing::default();
        ring.push(added(1));
        ring.push(ChangeKind::SourceStored {
            source: "a.md".into(),
        });
        let ChangesOutcome::Page { events, next, more } = ring.read(None, 10) else {
            panic!();
        };
        assert!(events.is_empty() && !more);
        let before = next;

        ring.push(added(2));
        let ChangesOutcome::Page { events, next, more } = ring.read(Some(&before), 10) else {
            panic!("a live cursor is never stale");
        };
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, added(2));
        assert!(!more);
        // The advanced cursor sees nothing further.
        let ChangesOutcome::Page { events, .. } = ring.read(Some(&next), 10) else {
            panic!();
        };
        assert!(events.is_empty());
    }

    #[test]
    fn limit_pages_and_reports_more() {
        let mut ring = ChangeRing::default();
        let ChangesOutcome::Page { next: start, .. } = ring.read(None, 10) else {
            panic!();
        };
        for count in 1..=5 {
            ring.push(added(count));
        }
        let ChangesOutcome::Page { events, next, more } = ring.read(Some(&start), 2) else {
            panic!();
        };
        assert_eq!(events.len(), 2);
        assert!(more, "three events remain past the limit");
        let ChangesOutcome::Page { events, more, .. } = ring.read(Some(&next), 10) else {
            panic!();
        };
        assert_eq!(events.len(), 3);
        assert!(!more);
    }

    #[test]
    fn an_evicted_position_is_stale_not_silently_partial() {
        let mut ring = ChangeRing::default();
        let ChangesOutcome::Page { next: start, .. } = ring.read(None, 10) else {
            panic!();
        };
        for count in 0..(CHANGE_RING_CAP + 10) {
            ring.push(added(count + 1));
        }
        assert_eq!(ring.read(Some(&start), 10), ChangesOutcome::Stale);
        // The oldest STILL-HELD event is reachable from the cursor just
        // before it.
        let front = ring.events.front().unwrap().seq;
        let cursor = format_cursor(ring.epoch, front - 1);
        let ChangesOutcome::Page { events, .. } = ring.read(Some(&cursor), 10) else {
            panic!("the ring's own edge must still serve");
        };
        assert_eq!(events.first().unwrap().seq, front);
    }

    #[test]
    fn foreign_future_and_malformed_cursors_are_stale() {
        let mut ring = ChangeRing::default();
        ring.push(added(1));
        let foreign = format_cursor(ring.epoch.wrapping_add(1), 1);
        assert_eq!(ring.read(Some(&foreign), 10), ChangesOutcome::Stale);
        let future = format_cursor(ring.epoch, 99);
        assert_eq!(ring.read(Some(&future), 10), ChangesOutcome::Stale);
        assert_eq!(ring.read(Some("garbage"), 10), ChangesOutcome::Stale);
        assert_eq!(ring.read(Some("cf1-zz-1"), 10), ChangesOutcome::Stale);
    }

    #[test]
    fn two_rings_never_validate_each_others_cursors() {
        let mut first = ChangeRing::default();
        let second = ChangeRing::default();
        first.push(added(1));
        let ChangesOutcome::Page { next, .. } = first.read(None, 10) else {
            panic!();
        };
        assert_eq!(second.read(Some(&next), 10), ChangesOutcome::Stale);
    }

    #[test]
    fn events_of_ops_aggregates_bulk_ops_and_keeps_retracts_individual() {
        use crate::registry::AssocOp;
        let assoc = |subject: &str| {
            WalOp::Associate(AssocOp {
                subject: subject.into(),
                label: "r".into(),
                object: "o".into(),
                weight: 1.0,
                source: None,
                paragraph: None,
            })
        };
        let ops = [
            assoc("a"),
            assoc("b"),
            WalOp::RetractSource {
                source: "old.md".into(),
            },
            WalOp::AliasConcept {
                alias: "x".into(),
                canonical: "y".into(),
            },
            assoc("c"),
        ];
        assert_eq!(
            events_of_ops(&ops),
            vec![
                ChangeKind::AssociationsAdded { count: 3 },
                ChangeKind::AliasesAdded { count: 1 },
                ChangeKind::SourceRetracted {
                    source: "old.md".into()
                },
            ]
        );
    }
}
