#!/usr/bin/env python3
"""Aggregate `taguru extract`'s trace records into metric tables (#792).

Reads the per-document records extract writes beside every batch
(ADR 0023-0029: `<out>/.extract-trace/<batch>.jsonl` and
`<batch stem>.attempts.jsonl`) and rolls the #784 metrics up through
the granularities: document -> context -> group -> run.

    python3 scripts/extract_metrics.py OUT_DIR [OUT_DIR ...]
        [--ledger ledger.json]        # source -> context/groups mapping
        [--price-in N] [--price-out N]  # cost per 1M tokens (default 0)
        [--anchoring report.json]     # `taguru anchoring --json` output (#793)
        [--json out.json] [--markdown out.md]  # default: markdown to stdout
        [--compare baseline.json]     # per-document deltas vs an earlier run

The ledger (the scenario-set's own record, #780) maps each document's
`source` (exactly as the trace's `document` record spells it) to its
context and group memberships:

    {"sources": {"docs/a.md": {"context": "ch1", "groups": ["book", "law"]}}}

An unmapped source lands in context `(unassigned)` and in one implicit
group named after its `--out` directory, so the tables are useful with
no ledger at all.

Metric definitions (ADR 0024 SS3.5, ADR 0026, ADR 0028 SS4, ADR 0029):

- loss rate, per item kind and reason: losses / (kept + losses) over
  the trace's `item` and `loss` records.
- paragraph coverage: covered / total `paragraph` records, and the
  byte-weighted variant.
- correction success: corrective attempts (those carrying `corrects`)
  that ended `stop_valid`, over all corrective attempts; the items
  those still removed are counted beside it (`removed_instead`).
- side rates over attempt records: stop_valid / length_limited /
  timeout shares, transport retries.
- ladder moves: counts per `move` kind.
- label stats over kept associations: top-1 share, singleton share,
  Shannon entropy (bits); the steering record's offered list size.
- graph shape over kept associations: isolated-concept share
  (degree 1), connected components.
- cost: seconds (sum, per call, lost to non-`stop_valid` attempts,
  per KB of chunk text), input/output tokens (and the tokens those
  lost attempts spent), and money at the given per-1M-token prices.
- failed: documents with an attempts log but no trace (ADR 0025 keeps
  the log of a document that failed) — counted apart from `documents`,
  with their attempts, moves, and cost rolled into every scope's sums
  and their quality metrics left empty (#807).

Standard library only, matching the repo's other python3 tooling.
`--self-test` builds a synthetic out-directory and checks the sums.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

TRACE_DIR = ".extract-trace"


# ---------------------------------------------------------------- loading


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_documents(out_dirs: list[Path]) -> list[dict]:
    """One entry per document the out-dirs hold records for: its
    records, split by kind.

    A traced document (a `<batch>.jsonl` trace, written only when the
    batch was) carries its attempts log beside it. An attempts log
    with NO trace is a document that never produced a batch — it
    failed, or the run was interrupted before it finished — kept
    exactly so its cost is visible (ADR 0025 §3.2); such an entry has
    an empty `trace` and `failed: True`, and the attempts log's own
    `document` record names the source (#807).
    """
    documents = []
    for out_dir in out_dirs:
        trace_dir = out_dir / TRACE_DIR
        if not trace_dir.is_dir():
            print(f"warning: {trace_dir} does not exist; skipped", file=sys.stderr)
            continue
        traced: set[str] = set()
        for trace_path in sorted(trace_dir.iterdir()):
            name = trace_path.name
            if not name.endswith(".jsonl") or name.endswith(".attempts.jsonl"):
                continue
            trace = read_jsonl(trace_path)
            header = next((r for r in trace if r.get("kind") == "document"), None)
            if header is None:
                print(f"warning: {trace_path} has no document record; skipped", file=sys.stderr)
                continue
            stem = name[: -len(".jsonl")]
            traced.add(stem)
            attempts_path = trace_dir / f"{stem}.attempts.jsonl"
            attempts_log = read_jsonl(attempts_path) if attempts_path.is_file() else []
            documents.append(
                {
                    "source": header["source"],
                    "out_dir": out_dir,
                    "trace": trace,
                    "attempts_log": attempts_log,
                    "failed": False,
                }
            )
        for attempts_path in sorted(trace_dir.iterdir()):
            name = attempts_path.name
            if not name.endswith(".attempts.jsonl"):
                continue
            stem = name[: -len(".attempts.jsonl")]
            if stem in traced:
                continue
            attempts_log = read_jsonl(attempts_path)
            header = next((r for r in attempts_log if r.get("kind") == "document"), None)
            if header is None:
                print(
                    f"warning: {attempts_path} has no trace and no document record; skipped",
                    file=sys.stderr,
                )
                continue
            documents.append(
                {
                    "source": header["source"],
                    "out_dir": out_dir,
                    "trace": [],
                    "attempts_log": attempts_log,
                    "failed": True,
                }
            )
    return documents


# ------------------------------------------------------------- per-document


def entropy_bits(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts if n)


def blank_metrics() -> dict:
    return {
        "documents": 0,
        # attempts-only documents (no batch, no trace): counted apart
        # from `documents` so the quality rates stay over what landed
        # while cost/attempts/moves include what it took to fail.
        "failed": 0,
        "kept": Counter(),
        "losses": Counter(),  # (kind, reason) -> n
        "paragraphs": Counter(),  # total/covered/bytes/covered_bytes
        "corrections": Counter(),  # total/resolved/unresolved/removed_instead/flagged
        "attempts": Counter(),  # total/by-state/transport_retries
        "moves": Counter(),
        "labels": Counter(),  # label -> uses (kept associations)
        "offered_labels": 0,
        "edges": [],  # (subject, object) pairs for graph shape
        "seconds": 0.0,
        "lost_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "lost_input_tokens": 0,
        "lost_output_tokens": 0,
        "chunk_bytes": 0,
    }


def absorb_document(bucket: dict, document: dict) -> None:
    if document.get("failed"):
        bucket["failed"] += 1
    else:
        bucket["documents"] += 1
    for record in document["trace"]:
        kind = record.get("kind")
        if kind == "item":
            bucket["kept"][record["item"]] += 1
            if record["item"] == "association":
                bucket["labels"][record["label"]] += 1
                bucket["edges"].append((record["subject"], record["object"]))
        elif kind == "loss":
            bucket["losses"][(record["item"], record["reason"])] += 1
        elif kind == "paragraph":
            bucket["paragraphs"]["total"] += 1
            bucket["paragraphs"]["bytes"] += record["bytes"]
            if record["covered"]:
                bucket["paragraphs"]["covered"] += 1
                bucket["paragraphs"]["covered_bytes"] += record["bytes"]
        elif kind == "chunk":
            bucket["chunk_bytes"] += record["chunk_bytes"]
        elif kind == "steering":
            bucket["offered_labels"] = max(
                bucket["offered_labels"], len(record.get("vocabulary") or [])
            )
    by_id = {}
    for record in document["attempts_log"]:
        if record.get("kind") == "attempt":
            by_id[(record["run_id"], record["attempt_seq"])] = record
    for record in document["attempts_log"]:
        kind = record.get("kind")
        if kind == "move":
            bucket["moves"][record["move"]] += 1
        elif kind == "attempt":
            # A `--replay` run re-emits an `attempt` record for a
            # completion satisfied from an earlier one instead of a
            # live call — `replayed_from` names that original, already
            # counted attempt (ADR 0031 §3.2, #823). Its own state/
            # transport_retries/elapsed_seconds/tokens describe the
            # replay itself (no model call, so time is near zero and
            # the tokens are the original's own, restated) — counting
            # it too would double every one of these for a replayed
            # completion while zeroing out its time, so it is skipped
            # here entirely rather than partially.
            if record.get("replayed_from"):
                continue
            bucket["attempts"]["total"] += 1
            bucket["attempts"][record["state"]] += 1
            bucket["attempts"]["transport_retries"] += record.get("transport_retries", 0)
            seconds = record.get("elapsed_seconds", 0.0)
            tokens_in = record.get("input_tokens") or 0
            tokens_out = record.get("output_tokens") or 0
            bucket["seconds"] += seconds
            bucket["input_tokens"] += tokens_in
            bucket["output_tokens"] += tokens_out
            if record["state"] != "stop_valid":
                bucket["lost_seconds"] += seconds
                bucket["lost_input_tokens"] += tokens_in
                bucket["lost_output_tokens"] += tokens_out
            corrects = record.get("corrects")
            if corrects:
                bucket["corrections"]["total"] += 1
                target = by_id.get((corrects["run_id"], corrects["attempt_seq"]))
                flagged = 1
                if target is not None and target.get("validation_issues"):
                    flagged = len(target["validation_issues"])
                bucket["corrections"]["flagged"] += flagged
                if record["state"] == "stop_valid":
                    bucket["corrections"]["resolved"] += 1
                    bucket["corrections"]["removed_instead"] += len(
                        record.get("removed_items") or []
                    )
                else:
                    bucket["corrections"]["unresolved"] += 1


def merge_into(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, Counter):
            target[key].update(value)
        elif isinstance(value, list):
            target[key].extend(value)
        elif key == "offered_labels":
            target[key] = max(target[key], value)
        else:
            target[key] += value


# --------------------------------------------------------------- summarize


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def graph_shape(edges: list[tuple[str, str]]) -> dict:
    degree: Counter = Counter()
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for subject, obj in edges:
        for node in (subject, obj):
            degree[node] += 1
            parent.setdefault(node, node)
        parent[find(subject)] = find(obj)
    concepts = len(degree)
    isolated = sum(1 for n in degree.values() if n == 1)
    components = len({find(node) for node in parent})
    return {
        "concepts": concepts,
        "isolated_share": ratio(isolated, concepts),
        "connected_components": components,
    }


def summarize(bucket: dict, prices: tuple[float, float]) -> dict:
    kept, losses = bucket["kept"], bucket["losses"]
    loss_rates = {}
    for item_kind in ("association", "concept", "label", "question", "alias"):
        lost = sum(n for (kind, _), n in losses.items() if kind == item_kind)
        # trace `item` records call alias records concept/label; `loss`
        # records call both "alias" — fold them for the rate.
        have = kept[item_kind]
        if item_kind == "alias":
            have = kept["concept"] + kept["label"]
        if lost == 0 and have == 0:
            continue
        loss_rates[item_kind] = {
            "kept": have,
            "lost": lost,
            "rate": ratio(lost, have + lost),
            "by_reason": {
                reason: n for (kind, reason), n in sorted(losses.items()) if kind == item_kind
            },
        }
    paragraphs = bucket["paragraphs"]
    corrections = bucket["corrections"]
    attempts = bucket["attempts"]
    labels = bucket["labels"]
    label_total = sum(labels.values())
    price_in, price_out = prices
    money = (
        bucket["input_tokens"] * price_in + bucket["output_tokens"] * price_out
    ) / 1_000_000
    lost_money = (
        bucket["lost_input_tokens"] * price_in + bucket["lost_output_tokens"] * price_out
    ) / 1_000_000
    return {
        "documents": bucket["documents"],
        "failed": bucket["failed"],
        "loss": loss_rates,
        "coverage": {
            "paragraphs": paragraphs["total"],
            "covered_rate": ratio(paragraphs["covered"], paragraphs["total"]),
            "covered_byte_rate": ratio(paragraphs["covered_bytes"], paragraphs["bytes"]),
        },
        "corrections": {
            "attempted": corrections["total"],
            "resolved": corrections["resolved"],
            "unresolved": corrections["unresolved"],
            "removed_instead": corrections["removed_instead"],
            "flagged_issues": corrections["flagged"],
            "success_rate": ratio(corrections["resolved"], corrections["total"]),
        },
        "attempts": {
            "total": attempts["total"],
            "stop_valid_rate": ratio(attempts["stop_valid"], attempts["total"]),
            "length_limited_rate": ratio(attempts["length_limited"], attempts["total"]),
            "timeout_rate": ratio(attempts["timeout"], attempts["total"]),
            "transport_retries": attempts["transport_retries"],
            "by_state": {
                state: n
                for state, n in sorted(attempts.items())
                if state not in ("total", "transport_retries")
            },
        },
        "moves": dict(sorted(bucket["moves"].items())),
        "labels": {
            "distinct": len(labels),
            "uses": label_total,
            "top1_share": ratio(max(labels.values(), default=0), label_total),
            "singleton_share": ratio(
                sum(1 for n in labels.values() if n == 1), len(labels)
            ),
            "entropy_bits": entropy_bits(list(labels.values())),
            "offered": bucket["offered_labels"],
        },
        "graph": graph_shape(bucket["edges"]),
        "cost": {
            "seconds": round(bucket["seconds"], 3),
            "seconds_per_call": ratio(bucket["seconds"], attempts["total"]),
            "lost_seconds": round(bucket["lost_seconds"], 3),
            "seconds_per_kb": ratio(bucket["seconds"], bucket["chunk_bytes"] / 1024),
            "input_tokens": bucket["input_tokens"],
            "output_tokens": bucket["output_tokens"],
            "lost_input_tokens": bucket["lost_input_tokens"],
            "lost_output_tokens": bucket["lost_output_tokens"],
            "money": round(money, 6),
            "lost_money": round(lost_money, 6),
        },
    }


def aggregate(documents: list[dict], ledger: dict, prices: tuple[float, float]) -> dict:
    per_document, contexts, groups, run = {}, {}, {}, blank_metrics()
    for document in documents:
        bucket = blank_metrics()
        absorb_document(bucket, document)
        source = document["source"]
        entry = ledger.get(source, {})
        context = entry.get("context", "(unassigned)")
        memberships = entry.get("groups") or [document["out_dir"].name]
        # Two out-dirs can hold the same source (the same document
        # extracted twice); the run/context/group sums count both, so
        # the document table must too — disambiguate by directory
        # rather than silently overwriting. The rule is deterministic,
        # so --compare keys still match when both runs collide alike.
        key = source
        if key in per_document:
            key = f"{source} ({document['out_dir'].name})"
            print(
                f"warning: {source} appears in more than one OUT_DIR; "
                f"listed again as {key!r}",
                file=sys.stderr,
            )
        per_document[key] = {
            "context": context,
            "groups": memberships,
            "failed": bool(document.get("failed")),
            "metrics": summarize(bucket, prices),
        }
        for name, store in [(context, contexts)] + [(g, groups) for g in memberships]:
            merge_into(store.setdefault(name, blank_metrics()), bucket)
        merge_into(run, bucket)
    return {
        "run": summarize(run, prices),
        "contexts": {name: summarize(b, prices) for name, b in sorted(contexts.items())},
        "groups": {name: summarize(b, prices) for name, b in sorted(groups.items())},
        "documents": per_document,
    }


# ---------------------------------------------------------------- anchoring

ANCHOR_COUNT_KEYS = (
    "associations",
    "anchored_strict",
    "anchored_with_aliases",
    "cited",
    "locator_valid",
)


def anchoring_rates(counts: dict) -> dict:
    rates = dict(counts)
    rates["rate_strict"] = ratio(counts["anchored_strict"], counts["associations"])
    rates["rate_with_aliases"] = ratio(
        counts["anchored_with_aliases"], counts["associations"]
    )
    rates["alias_dependent_rate"] = ratio(
        counts["anchored_with_aliases"] - counts["anchored_strict"], counts["associations"]
    )
    rates["locator_validity"] = ratio(counts["locator_valid"], counts["cited"])
    return rates


def attach_anchoring(report: dict, anchoring: dict) -> None:
    """Folds a `taguru anchoring --json` report into the tables, using
    the context/group assignments the trace aggregation already made
    (so both metric families roll up identically)."""
    run = Counter()
    contexts: dict[str, Counter] = defaultdict(Counter)
    groups: dict[str, Counter] = defaultdict(Counter)
    for source, row in anchoring.get("documents", {}).items():
        counts = {key: row.get(key, 0) for key in ANCHOR_COUNT_KEYS}
        entry = report["documents"].get(source)
        if entry is None:
            print(
                f"warning: anchoring report covers {source!r}, which the trace "
                "aggregation does not; skipped",
                file=sys.stderr,
            )
            continue
        entry["metrics"]["anchoring"] = anchoring_rates(counts)
        run.update(counts)
        contexts[entry["context"]].update(counts)
        for group in entry["groups"]:
            groups[group].update(counts)
    if not run:
        return
    report["run"]["anchoring"] = anchoring_rates(dict(run))
    for name, counter in contexts.items():
        report["contexts"][name]["anchoring"] = anchoring_rates(dict(counter))
    for name, counter in groups.items():
        report["groups"][name]["anchoring"] = anchoring_rates(dict(counter))


# ----------------------------------------------------------------- compare

# metric path -> direction ("down" = lower is better)
COMPARE_KEYS = [
    (("loss", "association", "rate"), "down", "assoc loss"),
    (("anchoring", "rate_strict"), "up", "anchoring (strict)"),
    (("anchoring", "rate_with_aliases"), "up", "anchoring (aliases)"),
    (("anchoring", "locator_validity"), "up", "locator validity"),
    (("coverage", "covered_byte_rate"), "up", "coverage (bytes)"),
    (("corrections", "success_rate"), "up", "correction success"),
    (("attempts", "stop_valid_rate"), "up", "stop_valid"),
    (("cost", "seconds"), "down", "seconds"),
    (("cost", "output_tokens"), "down", "output tokens"),
]


def dig(mapping: dict, path: tuple) -> float | None:
    for key in path:
        if not isinstance(mapping, dict) or key not in mapping:
            return None
        mapping = mapping[key]
    return mapping


def compare(current: dict, baseline: dict) -> dict:
    shared = sorted(set(current["documents"]) & set(baseline["documents"]))
    only_current = sorted(set(current["documents"]) - set(baseline["documents"]))
    only_baseline = sorted(set(baseline["documents"]) - set(current["documents"]))
    rows, verdicts = {}, {}
    for path, direction, label in COMPARE_KEYS:
        improved = worsened = unchanged = 0
        deltas = {}
        for source in shared:
            now = dig(current["documents"][source]["metrics"], path)
            was = dig(baseline["documents"][source]["metrics"], path)
            if now is None or was is None:
                continue
            delta = now - was
            deltas[source] = {"baseline": was, "current": now, "delta": round(delta, 6)}
            if abs(delta) < 1e-9:
                unchanged += 1
            elif (delta < 0) == (direction == "down"):
                improved += 1
            else:
                worsened += 1
        rows[label] = deltas
        verdicts[label] = {
            "improved": improved,
            "worsened": worsened,
            "unchanged": unchanged,
        }
    return {
        "shared_documents": len(shared),
        "only_in_current": only_current,
        "only_in_baseline": only_baseline,
        "verdicts": verdicts,
        "per_document": rows,
    }


# ---------------------------------------------------------------- markdown


def fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def metric_row(name: str, metrics: dict) -> str:
    loss = dig(metrics, ("loss", "association", "rate"))
    return (
        f"| {name} | {metrics['documents']} | {metrics.get('failed', 0)} | {fmt(loss)} "
        f"| {fmt(dig(metrics, ('coverage', 'covered_byte_rate')))} "
        f"| {fmt(dig(metrics, ('corrections', 'success_rate')))} "
        f"| {fmt(dig(metrics, ('attempts', 'stop_valid_rate')))} "
        f"| {sum(metrics['moves'].values())} "
        f"| {metrics['cost']['input_tokens']} | {metrics['cost']['output_tokens']} "
        f"| {fmt(metrics['cost']['seconds'])} | {fmt(metrics['cost']['lost_seconds'])} |"
    )


HEADER = (
    "| scope | docs | failed | assoc loss | coverage(B) | corr. success | stop_valid "
    "| moves | in tok | out tok | secs | lost secs |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def markdown(report: dict) -> str:
    lines = ["# extract metrics", "", "## Run", "", HEADER, metric_row("run", report["run"])]
    for title, section in [("Contexts", "contexts"), ("Groups", "groups")]:
        if report[section]:
            lines += ["", f"## {title}", "", HEADER]
            lines += [metric_row(name, m) for name, m in report[section].items()]
    lines += ["", "## Documents", "", HEADER]
    lines += [
        metric_row(f"{source} (failed)" if entry.get("failed") else source, entry["metrics"])
        for source, entry in report["documents"].items()
    ]
    if any(entry.get("failed") for entry in report["documents"].values()):
        lines += [
            "",
            "`failed`: documents with an attempts log but no trace — no batch was "
            "written (the extraction failed, or the run stopped before it finished); "
            "quality columns are `-`, cost/attempts/moves are what it took.",
        ]
    anchored = [("run", report["run"])] + [
        (name, m) for section in ("contexts", "groups") for name, m in report[section].items()
    ] + [(source, entry["metrics"]) for source, entry in report["documents"].items()]
    anchored = [(name, m["anchoring"]) for name, m in anchored if "anchoring" in m]
    if anchored:
        lines += [
            "",
            "## Anchoring",
            "",
            "| scope | assocs | strict | with aliases | alias-dependent | locator validity |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {name} | {a['associations']} | {fmt(a['rate_strict'])} "
            f"| {fmt(a['rate_with_aliases'])} | {fmt(a['alias_dependent_rate'])} "
            f"| {fmt(a['locator_validity'])} |"
            for name, a in anchored
        ]
    if "compare" in report:
        cmp = report["compare"]
        lines += [
            "",
            "## Compared to baseline",
            "",
            f"shared documents: {cmp['shared_documents']}"
            + (
                f"; only in current: {', '.join(cmp['only_in_current'])}"
                if cmp["only_in_current"]
                else ""
            )
            + (
                f"; only in baseline: {', '.join(cmp['only_in_baseline'])}"
                if cmp["only_in_baseline"]
                else ""
            ),
            "",
            "| metric | improved | worsened | unchanged |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {label} | {v['improved']} | {v['worsened']} | {v['unchanged']} |"
            for label, v in cmp["verdicts"].items()
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- self-test


def self_test() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        trace_dir = out / TRACE_DIR
        trace_dir.mkdir(parents=True)
        run_id = "0" * 16
        trace = [
            {"kind": "document", "run_id": run_id, "source": "a.md", "document_sha256": "d" * 64,
             "batch_path": "out/a.jsonl", "chunk_total": 1},
            {"kind": "steering", "chunk_index": None, "candidates": [],
             "vocabulary": [{"label": "rel", "count": 2}], "context_names": [], "schema": None},
            {"kind": "chunk", "chunk_index": 0, "chunk_total": 1, "chunk_sha256": "c" * 64,
             "chunk_bytes": 2048, "paragraph_first": 0, "paragraph_last": 1},
            {"kind": "piece", "piece_id": "c" * 64, "chunk_index": 0, "chunk_sha256": "c" * 64,
             "piece_bytes": 2048, "paragraph_first": 0, "paragraph_last": 1, "reused": False,
             "attempt": {"run_id": run_id, "attempt_seq": 2}},
            {"kind": "item", "item": "association", "subject": "A", "label": "rel",
             "object": "B", "piece_id": "c" * 64},
            {"kind": "item", "item": "association", "subject": "A", "label": "rel",
             "object": "C", "piece_id": "c" * 64},
            {"kind": "item", "item": "association", "subject": "D", "label": "uses",
             "object": "E", "piece_id": "c" * 64},
            {"kind": "loss", "item": "association", "reason": "removed", "rule": "r",
             "path": "associations[3]", "raw": {}, "piece_id": "c" * 64,
             "attempt": {"run_id": run_id, "attempt_seq": 2}, "paragraph": 0, "text": "t"},
            {"kind": "paragraph", "paragraph": 0, "bytes": 1000, "items": 2, "covered": True},
            {"kind": "paragraph", "paragraph": 1, "bytes": 3000, "items": 0, "covered": False,
             "text": "t"},
        ]
        attempts = [
            {"kind": "document", "run_id": run_id, "source": "a.md",
             "document_sha256": "d" * 64, "resumed": False},
            {"kind": "system", "sha256": "s" * 64, "bytes": 3, "content": "sys"},
            {"kind": "attempt", "run_id": run_id, "attempt_seq": 1, "piece_id": "c" * 64,
             "source": "a.md", "chunk_index": 0, "stage": "item", "attempt": 1,
             "max_attempts": 2, "state": "stop_malformed", "length_limited": False,
             "transport_retries": 2, "elapsed_seconds": 2.0, "requested_max_tokens": None,
             "finish_reason": "stop", "input_tokens": 100, "output_tokens": 50,
             "messages": [], "answer": "bad", "parse_error": "e",
             "validation_issues": ["i1", "i2"], "removed_items": None},
            {"kind": "move", "move": "escalate", "run_id": run_id, "piece_id": "c" * 64,
             "chunk_index": 0, "reason": "cap", "from_max_tokens": 512, "to_max_tokens": 1024},
            {"kind": "attempt", "run_id": run_id, "attempt_seq": 2, "piece_id": "c" * 64,
             "source": "a.md", "chunk_index": 0, "stage": "item", "attempt": 2,
             "max_attempts": 2, "state": "stop_valid", "length_limited": False,
             "transport_retries": 0, "elapsed_seconds": 3.0, "requested_max_tokens": None,
             "finish_reason": "stop", "input_tokens": 200, "output_tokens": 150,
             "messages": [], "answer": "good", "parse_error": None,
             "validation_issues": None,
             "removed_items": ["associations[3]: r"],
             "corrects": {"run_id": run_id, "attempt_seq": 1}},
            # A later --replay run's re-emitted record for attempt 2 —
            # huge fake seconds/tokens that must never reach the cost
            # rollup, proving replayed_from is what excludes it (#823).
            {"kind": "attempt", "run_id": "1" * 16, "attempt_seq": 1, "piece_id": "c" * 64,
             "source": "a.md", "chunk_index": 0, "stage": "item", "attempt": 1,
             "max_attempts": 2, "state": "stop_valid", "length_limited": False,
             "transport_retries": 9, "elapsed_seconds": 1000.0, "requested_max_tokens": None,
             "finish_reason": "stop", "input_tokens": 999999, "output_tokens": 999999,
             "messages": [], "answer": "good", "parse_error": None,
             "validation_issues": None, "removed_items": None,
             "replayed_from": {"run_id": run_id, "attempt_seq": 2}},
            # A replayed *failed* completion (answer: null, ADR 0025
            # §3.3's timeout/transport shape) — must be excluded the
            # same way a replayed success is, not just counted with
            # zeroed-out tokens.
            {"kind": "attempt", "run_id": "1" * 16, "attempt_seq": 2, "piece_id": "d" * 64,
             "source": "a.md", "chunk_index": 0, "stage": "item", "attempt": 1,
             "max_attempts": 2, "state": "timeout", "length_limited": False,
             "transport_retries": 9, "elapsed_seconds": 1000.0, "requested_max_tokens": None,
             "finish_reason": None, "input_tokens": None, "output_tokens": None,
             "messages": [], "answer": None, "parse_error": "e",
             "validation_issues": None, "removed_items": None,
             "replayed_from": {"run_id": run_id, "attempt_seq": 2}},
        ]
        (trace_dir / "a.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in trace), encoding="utf-8"
        )
        (trace_dir / "a.attempts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in attempts), encoding="utf-8"
        )
        # A document that FAILED: an attempts log and no trace (#807).
        # Two length-limited rounds, a split, then the run gave up —
        # every second and token of it must reach the run's sums.
        failed_attempts = [
            {"kind": "document", "run_id": "2" * 16, "source": "f.md",
             "document_sha256": "f" * 64, "resumed": False},
            {"kind": "system", "sha256": "s" * 64, "bytes": 3, "content": "sys"},
            {"kind": "attempt", "run_id": "2" * 16, "attempt_seq": 1, "piece_id": "e" * 64,
             "source": "f.md", "chunk_index": 0, "stage": "item", "attempt": 1,
             "max_attempts": 2, "state": "length_limited", "length_limited": True,
             "transport_retries": 1, "elapsed_seconds": 10.0, "requested_max_tokens": 300,
             "finish_reason": "length", "input_tokens": 400, "output_tokens": 300,
             "messages": [], "answer": "cut", "parse_error": "e",
             "validation_issues": None, "removed_items": None},
            {"kind": "move", "move": "escalate", "run_id": "2" * 16, "piece_id": "e" * 64,
             "chunk_index": 0, "reason": "cap", "from_max_tokens": 300, "to_max_tokens": 600},
            {"kind": "attempt", "run_id": "2" * 16, "attempt_seq": 2, "piece_id": "e" * 64,
             "source": "f.md", "chunk_index": 0, "stage": "item", "attempt": 1,
             "max_attempts": 2, "state": "length_limited", "length_limited": True,
             "transport_retries": 0, "elapsed_seconds": 20.0, "requested_max_tokens": 600,
             "finish_reason": "length", "input_tokens": 400, "output_tokens": 600,
             "messages": [], "answer": "cut", "parse_error": "e",
             "validation_issues": None, "removed_items": None},
            {"kind": "move", "move": "split", "run_id": "2" * 16, "piece_id": "e" * 64,
             "chunk_index": 0, "reason": "cap", "piece_bytes": 4096, "split_cap": 2048,
             "sub_pieces": 2},
        ]
        (trace_dir / "f.attempts.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in failed_attempts), encoding="utf-8"
        )
        # An attempts log with no document record is neither: skipped.
        (trace_dir / "g.attempts.jsonl").write_text(
            json.dumps({"kind": "system", "sha256": "s" * 64, "bytes": 3, "content": "sys"})
            + "\n",
            encoding="utf-8",
        )
        # The same source in a second out-dir must not overwrite the
        # first document-table row.
        twin = Path(tmp) / "twin"
        twin_trace = twin / TRACE_DIR
        twin_trace.mkdir(parents=True)
        for name in ("a.jsonl", "a.attempts.jsonl"):
            twin_trace.joinpath(name).write_text(
                trace_dir.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8"
            )
        report = aggregate(
            load_documents([out, twin]),
            {"sources": {"a.md": {"context": "ch1", "groups": ["book"]}}}["sources"],
            (100.0, 200.0),
        )
        if sorted(report["documents"]) != ["a.md", "a.md (twin)", "f.md"]:
            print(f"self-test collision keys failed: {sorted(report['documents'])}")
            return 1
        if report["run"]["documents"] != 2:
            print("self-test collision run count failed")
            return 1
        report = aggregate(
            load_documents([out]),
            {"sources": {"a.md": {"context": "ch1", "groups": ["book"]},
                         "f.md": {"context": "ch2", "groups": ["book"]}}}["sources"],
            (100.0, 200.0),
        )
        run = report["run"]
        failed = report["documents"]["f.md"]
        checks = [
            (run["documents"], 1),
            # The failed document (#807): apart from `documents`, its
            # attempts/moves/cost in the sums, its quality metrics empty.
            (run["failed"], 1),
            (failed["failed"], True),
            (failed["context"], "ch2"),
            (failed["metrics"]["documents"], 0),
            (failed["metrics"]["failed"], 1),
            (failed["metrics"]["loss"], {}),
            (failed["metrics"]["coverage"]["covered_byte_rate"], None),
            (failed["metrics"]["attempts"]["total"], 2),
            (failed["metrics"]["moves"], {"escalate": 1, "split": 1}),
            (failed["metrics"]["cost"]["seconds"], 30.0),
            (failed["metrics"]["cost"]["lost_seconds"], 30.0),
            (failed["metrics"]["cost"]["lost_output_tokens"], 900),
            (report["contexts"]["ch2"]["documents"], 0),
            (report["contexts"]["ch2"]["failed"], 1),
            (report["groups"]["book"]["documents"], 1),
            (report["groups"]["book"]["failed"], 1),
            (report["groups"]["book"]["cost"]["seconds"], 35.0),
            (run["loss"]["association"]["kept"], 3),
            (run["loss"]["association"]["lost"], 1),
            (round(run["loss"]["association"]["rate"], 4), 0.25),
            (run["coverage"]["covered_rate"], 0.5),
            (run["coverage"]["covered_byte_rate"], 0.25),
            (run["corrections"]["attempted"], 1),
            (run["corrections"]["resolved"], 1),
            (run["corrections"]["flagged_issues"], 2),
            (run["corrections"]["removed_instead"], 1),
            (run["corrections"]["success_rate"], 1.0),
            # A replayed --replay re-emission of attempt 2 sits in the
            # fixture (huge fake seconds/tokens/transport_retries) —
            # every count below must be exactly as if it were absent.
            (run["attempts"]["total"], 4),
            (run["attempts"]["stop_valid_rate"], 0.25),
            (run["attempts"]["transport_retries"], 3),
            (run["moves"], {"escalate": 2, "split": 1}),
            (run["labels"]["distinct"], 2),
            (run["labels"]["top1_share"], 2 / 3),
            (run["labels"]["singleton_share"], 0.5),
            (run["labels"]["offered"], 1),
            (run["graph"]["concepts"], 5),
            (run["graph"]["connected_components"], 2),
            (run["graph"]["isolated_share"], 4 / 5),
            (run["cost"]["seconds"], 35.0),
            (run["cost"]["lost_seconds"], 32.0),
            (run["cost"]["input_tokens"], 1100),
            (run["cost"]["output_tokens"], 1100),
            (run["cost"]["lost_input_tokens"], 900),
            (run["cost"]["lost_output_tokens"], 950),
            # 1100 in * 100/1M + 1100 out * 200/1M
            (run["cost"]["money"], 0.33),
            (run["cost"]["seconds_per_kb"], 17.5),
            (report["contexts"]["ch1"]["documents"], 1),
            (report["groups"]["book"]["documents"], 1),
            (report["documents"]["a.md"]["context"], "ch1"),
        ]
        for index, (got, want) in enumerate(checks):
            if got != want:
                print(f"self-test check {index} failed: got {got!r}, want {want!r}")
                return 1
        # anchoring attachment: counts roll up with the same assignments.
        attach_anchoring(
            report,
            {"documents": {"a.md": {"context": "ch1", "associations": 4,
                                    "anchored_strict": 2, "anchored_with_aliases": 3,
                                    "cited": 3, "locator_valid": 3},
                           "ghost.md": {"context": "x", "associations": 1,
                                        "anchored_strict": 0,
                                        "anchored_with_aliases": 0,
                                        "cited": 0, "locator_valid": 0}}},
        )
        anchor = report["run"]["anchoring"]
        anchor_checks = [
            (anchor["rate_strict"], 0.5),
            (anchor["rate_with_aliases"], 0.75),
            (anchor["alias_dependent_rate"], 0.25),
            (anchor["locator_validity"], 1.0),
            (report["contexts"]["ch1"]["anchoring"]["associations"], 4),
            (report["groups"]["book"]["anchoring"]["associations"], 4),
            ("anchoring" in report["documents"]["a.md"]["metrics"], True),
        ]
        for index, (got, want) in enumerate(anchor_checks):
            if got != want:
                print(f"self-test anchoring check {index} failed: got {got!r}, want {want!r}")
                return 1
        # compare: a baseline where the loss rate was worse and seconds lower
        baseline = json.loads(json.dumps(report))
        baseline["documents"]["a.md"]["metrics"]["loss"]["association"]["rate"] = 0.5
        baseline["documents"]["a.md"]["metrics"]["cost"]["seconds"] = 1.0
        verdicts = compare(report, baseline)["verdicts"]
        if verdicts["assoc loss"]["improved"] != 1 or verdicts["seconds"]["worsened"] != 1:
            print(f"self-test compare failed: {verdicts}")
            return 1
        rendered = markdown(report)
        if "| f.md (failed) | 0 | 1 |" not in rendered or "`failed`:" not in rendered:
            print(f"self-test markdown failed-row check failed:\n{rendered}")
            return 1
    print("self-test ok")
    return 0


# --------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("out_dirs", nargs="*", type=Path, metavar="OUT_DIR")
    parser.add_argument("--ledger", type=Path, help="source -> context/groups JSON")
    parser.add_argument("--price-in", type=float, default=0.0, help="per 1M input tokens")
    parser.add_argument("--price-out", type=float, default=0.0, help="per 1M output tokens")
    parser.add_argument("--anchoring", type=Path, help="a `taguru anchoring --json` report")
    parser.add_argument("--json", type=Path, help="write the full report as JSON here")
    parser.add_argument("--markdown", type=Path, help="write the tables here (default stdout)")
    parser.add_argument("--compare", type=Path, help="a --json report to diff against")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.out_dirs:
        parser.error("at least one OUT_DIR (or --self-test) is required")
    ledger = {}
    if args.ledger:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8")).get("sources", {})
    documents = load_documents(args.out_dirs)
    if not documents:
        print("no documents found (no trace, and no attempts log)", file=sys.stderr)
        return 1
    report = aggregate(documents, ledger, (args.price_in, args.price_out))
    if args.anchoring:
        attach_anchoring(
            report, json.loads(args.anchoring.read_text(encoding="utf-8"))
        )
    if args.compare:
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
        report["compare"] = compare(report, baseline)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rendered = markdown(report)
    if args.markdown:
        args.markdown.write_text(rendered, encoding="utf-8")
    if not args.json and not args.markdown:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
