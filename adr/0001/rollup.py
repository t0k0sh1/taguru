# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jsonschema>=4.21",
#   "pydantic>=2.7",
# ]
# ///
"""Aggregate results/attempts.jsonl into the CSV tables the ADR cites.

Usage: uv run adr/0001/rollup.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import common

RESULTS_DIR = common.ADR_DIR / "results"
ATTEMPTS_PATH = RESULTS_DIR / "attempts.jsonl"
CANARY_PATH = RESULTS_DIR / "canary.json"


def load() -> tuple[list[dict], list[dict]]:
    attempts, trials = [], []
    for line in ATTEMPTS_PATH.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") == "attempt":
            attempts.append(record)
        elif record.get("type") == "trial":
            trials.append(record)
    return attempts, trials


def group(records: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    table: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        table[tuple(str(record.get(field)) for field in fields)].append(record)
    return dict(sorted(table.items()))


def pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def percentile(values: list[float], p: int) -> float | str:
    if not values:
        return ""
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100 * len(ordered)) - 1)
    return round(ordered[index], 1)


def mean(values: list[float]) -> float | str:
    return round(sum(values) / len(values), 2) if values else ""


def write_csv(name: str, header: list[str], rows: list[list]) -> Path:
    path = RESULTS_DIR / name
    with path.open("w", newline="") as sink:
        writer = csv.writer(sink)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def main() -> None:
    attempts, trials = load()

    # 1. failure_rates: trial outcomes per mechanism/wire/ceiling
    SKIPPED = ("skipped_unsupported", "skipped_stall")
    rows = []
    for key, bucket in group(trials, ("model", "mechanism", "wire", "ceiling")).items():
        run = [t for t in bucket if t["failure_class"] not in SKIPPED]
        unsupported = sum(1 for t in bucket if t["failure_class"] == "skipped_unsupported")
        stalled = sum(1 for t in bucket if t["failure_class"] == "skipped_stall")
        by = lambda cls: sum(1 for t in run if t["failure_class"] == cls)
        rows.append(
            [
                *key,
                len(run),
                unsupported,
                stalled,
                pct(by("ok"), len(run)),
                pct(by("parser"), len(run)),
                pct(by("length"), len(run)),
                pct(by("timeout"), len(run)),
                pct(by("transport"), len(run)),
            ]
        )
    write_csv(
        "failure_rates.csv",
        ["model", "mechanism", "wire", "ceiling", "n_run", "n_skip_unsupported", "n_skip_stall",
         "pct_ok", "pct_parser", "pct_length", "pct_timeout", "pct_transport"],
        rows,
    )

    # 2. truncation_crosstab: finish_reason x syntax outcome, per ceiling
    rows = []
    counted = group(
        [a for a in attempts if "finish_reason" in a],
        ("model", "mechanism", "wire", "effective_ceiling", "finish_reason", "syntax_ok"),
    )
    for key, bucket in counted.items():
        rows.append([*key, len(bucket)])
    write_csv(
        "truncation_crosstab.csv",
        ["model", "mechanism", "wire", "effective_ceiling", "finish_reason", "syntax_ok", "n"],
        rows,
    )

    # 3. latency/token percentiles
    rows = []
    for key, bucket in group(
        [a for a in attempts if a.get("http_status") == 200], ("model", "mechanism", "wire", "doc")
    ).items():
        latencies = [a["latency_ms"] for a in bucket]
        tokens = [a["completion_tokens"] for a in bucket if a.get("completion_tokens") is not None]
        rows.append(
            [*key, len(bucket),
             percentile(latencies, 50), percentile(latencies, 90),
             percentile(tokens, 50), percentile(tokens, 90)]
        )
    write_csv(
        "latency_tokens_percentiles.csv",
        ["model", "mechanism", "wire", "doc", "n_calls",
         "latency_ms_p50", "latency_ms_p90", "completion_tokens_p50", "completion_tokens_p90"],
        rows,
    )

    # 4. item counts on successful parses, and how often the ceiling was hit
    length_trials: dict[tuple, set] = defaultdict(set)
    for a in attempts:
        if a.get("failure_class") == "length" or a.get("finish_reason") in ("length", "max_tokens"):
            length_trials[(a["model"], a["mechanism"], a["wire"], a["doc"], a["ceiling"])].add(a["trial_key"])
    rows = []
    for key, bucket in group(
        [a for a in attempts if a.get("syntax_ok")], ("model", "mechanism", "wire", "doc", "ceiling")
    ).items():
        lenient = [a["lenient"] for a in bucket if a.get("lenient")]
        trial_keys = {a["trial_key"] for a in bucket}
        hit = length_trials.get(tuple(key), set())
        rows.append(
            [
                *key,
                len(bucket),
                mean([l["merged_associations"] for l in lenient]),
                mean([l["merged_aliases"] for l in lenient]),
                mean([l["parsed_questions"] for l in lenient]),
                mean([l["dropped"] for l in lenient]),
                mean([l["duplicates"] for l in lenient]),
                pct(len(hit), len(trial_keys | hit)),
            ]
        )
    write_csv(
        "item_counts_and_completion.csv",
        ["model", "mechanism", "wire", "doc", "ceiling", "n_ok_attempts",
         "mean_merged_associations", "mean_merged_aliases", "mean_parsed_questions",
         "mean_dropped", "mean_duplicates", "pct_trials_hit_length"],
        rows,
    )

    # 5. corrective-retry effectiveness
    rows = []
    for key, bucket in group(trials, ("model", "mechanism", "wire")).items():
        run = [t for t in bucket if t["failure_class"] not in SKIPPED]
        if not run:
            continue
        recovered = sum(1 for t in run if t.get("recovered_by_retry"))
        rows.append(
            [*key, len(run), pct(sum(1 for t in run if t["ok"]), len(run)),
             pct(recovered, len(run)), mean([t["attempts_used"] for t in run])]
        )
    write_csv(
        "lossless_correction.csv",
        ["model", "mechanism", "wire", "n_run", "pct_ok", "pct_recovered_by_retry", "mean_attempts"],
        rows,
    )

    # 6. lenient (Rust) vs strict (SDK) parse outcomes on identical answers
    rows = []
    for key, bucket in group(
        [a for a in attempts if a.get("http_status") == 200], ("model", "mechanism", "wire", "doc")
    ).items():
        lenient_ok = sum(1 for a in bucket if a.get("syntax_ok"))
        strict_ok = sum(1 for a in bucket if a.get("strict_ok"))
        rows.append(
            [*key, len(bucket), pct(lenient_ok, len(bucket)), pct(strict_ok, len(bucket)),
             round(pct(lenient_ok, len(bucket)) - pct(strict_ok, len(bucket)), 1)]
        )
    write_csv(
        "lenient_vs_strict_delta.csv",
        ["model", "mechanism", "wire", "doc", "n_answers",
         "pct_lenient_pass", "pct_strict_pass", "delta_pp"],
        rows,
    )

    # 7. golden recall (docs that have ground truth)
    rows = []
    for key, bucket in group(
        [a for a in attempts if a.get("golden")], ("model", "mechanism", "wire", "doc", "ceiling")
    ).items():
        goldens = [a["golden"] for a in bucket]
        triple = [g["triple_recall"] for g in goldens if "triple_recall" in g]
        pair = [g["pair_recall"] for g in goldens if "pair_recall" in g]
        polarity = [
            g["polarity_ok"] / g["polarity_checked"]
            for g in goldens
            if g.get("polarity_checked")
        ]
        surface = [g["alias_surface_recall"] for g in goldens if g.get("alias_surface_recall") is not None]
        false_pos = [g["alias_false_positives"] for g in goldens if "alias_false_positives" in g]
        rows.append(
            [*key, len(bucket),
             mean(triple), round(min(triple), 4) if triple else "",
             mean(pair), mean(polarity),
             mean(surface), mean(false_pos)]
        )
    write_csv(
        "golden_recall.csv",
        ["model", "mechanism", "wire", "doc", "ceiling", "n_scored",
         "mean_triple_recall", "min_triple_recall", "mean_pair_recall",
         "mean_polarity_ok_rate", "mean_alias_surface_recall", "mean_alias_false_positives"],
        rows,
    )

    # 8. capability matrix straight from the canary
    canary = json.loads(CANARY_PATH.read_text()) if CANARY_PATH.exists() else {"probes": []}
    rows = [
        [probe["model"], probe["wire"], probe["mechanism"], probe["verdict"],
         probe.get("finish_reason", ""), probe.get("evidence", "").replace("\n", "\\n")[:200]]
        for probe in canary["probes"]
    ]
    write_csv(
        "capability_matrix.csv",
        ["model", "wire", "mechanism", "verdict", "finish_reason", "evidence"],
        rows,
    )

    # summary.md — the extremes, as candidate sentences for the ADR
    lines = ["# Harness rollup summary", ""]
    lines.append(f"- attempts: {len(attempts)}, trials: {len(trials)} "
                 f"(ok: {sum(1 for t in trials if t['ok'])}, "
                 f"skipped-unsupported: {sum(1 for t in trials if t['failure_class'] == 'skipped_unsupported')}, "
                 f"skipped-stall: {sum(1 for t in trials if t['failure_class'] == 'skipped_stall')})")
    core = [t for t in trials if t["variant"] == "core" and t["failure_class"] not in SKIPPED]
    by_mech: dict[tuple, list] = defaultdict(list)
    for t in core:
        by_mech[(t["mechanism"], t["wire"])].append(t)
    ranked = sorted(
        ((pct(sum(1 for t in bucket if t["ok"]), len(bucket)), mech, wire, len(bucket))
         for (mech, wire), bucket in by_mech.items()),
        reverse=True,
    )
    for rate, mech, wire, n in ranked:
        lines.append(f"- core trials {mech}/{wire}: {rate}% ok (n={n})")
    lengths = [a for a in attempts if a.get("finish_reason") in ("length", "max_tokens")]
    lines.append(f"- attempts finishing at the output cap: {len(lengths)}")
    recovered = sum(1 for t in trials if t.get("recovered_by_retry"))
    lines.append(f"- trials recovered by the corrective retry: {recovered}")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
