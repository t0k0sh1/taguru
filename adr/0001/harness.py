# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.27",
#   "jsonschema>=4.21",
#   "pydantic>=2.7",
# ]
# ///
"""ADR-0001 evidence harness: run the structured-extraction mechanism matrix
against a local Ollama and record raw per-attempt metadata.

Mechanisms (x2 wires: Ollama-native /api/chat, OpenAI-compatible /v1):
  A0  bare prompted JSON — exactly what `taguru extract` sends today
  A1  JSON mode          — native format:"json" / v1 response_format json_object
  B   schema-constrained — native format:<schema> / v1 response_format json_schema
  C   tool calling       — tools=[ModelOutput], v1 forces tool_choice

Every trial replays the production corrective-retry loop (max 2 attempts,
full replay of the prior bad answer, the producers' exact corrective texts,
imported from taguru_langchain._extract). Results append to
results/attempts.jsonl, which doubles as the resume journal: completed
trials are skipped on re-run. Full untruncated responses land in raw/
(gitignored); the journal stores the first 2 KiB of each answer.

Usage:
  uv run adr/0001/harness.py --canary-only     # capability probes only
  uv run adr/0001/harness.py                   # canary (if needed) + matrix
  uv run adr/0001/harness.py --limit 5         # smoke-run five trials
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

import common
from common import x  # taguru_langchain._extract, the production twin

BASE_URL_DEFAULT = "http://localhost:11434"
PRIMARY_MODEL = "taguru-extract-12b"
SECONDARY_MODEL = "qwen2.5:7b"
MECHANISMS = ["A0", "A1", "B", "C"]
WIRES = ["native", "v1"]
CHEAP_DOCS = ["short", "negation", "aliases", "table_noisy"]
EXPENSIVE_DOCS = ["long_dense", "output_limit"]
CEILINGS_EXPENSIVE = ["512", "2048", "unbounded"]
MAX_ATTEMPTS = 2  # production default (DEFAULT_MAX_ATTEMPTS)
CALL_TIMEOUT_SECS = 180
NUM_CTX = 16384  # docs/extract.html's recommendation for 24 KiB chunks
TRUNCATE_JOURNAL_BYTES = 2048

RESULTS_DIR = common.ADR_DIR / "results"
RAW_DIR = common.ADR_DIR / "raw"
ATTEMPTS_PATH = RESULTS_DIR / "attempts.jsonl"
CANARY_PATH = RESULTS_DIR / "canary.json"

# The canonical schema, and the flavor tool-calling providers expect
# (plain parameters object, no $schema/title keys).
SCHEMA = common.SCHEMA
TOOL_PARAMETERS = {k: v for k, v in SCHEMA.items() if k not in ("$schema", "title")}
TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "ModelOutput",
        "description": "Return the complete extraction as one JSON object.",
        "parameters": TOOL_PARAMETERS,
    },
}

COLORS_PROMPT = [
    {
        "role": "user",
        "content": "Name three colors and say briefly why you like each of them.",
    }
]
COLORS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["colors"],
    "properties": {"colors": {"type": "array", "items": {"type": "string"}}},
}


def trial_key(model: str, wire: str, mech: str, doc: str, ceiling: str, variant: str, rep: int) -> str:
    return f"{model}::{wire}::{mech}::{doc}::{ceiling}::{variant}::r{rep}"


def raw_path(key: str, attempt: int) -> Path:
    safe = key.replace("::", "__").replace(":", "-").replace("#", "_").replace("/", "-")
    return RAW_DIR / f"{safe}.a{attempt}.txt"


# -- request building ---------------------------------------------------------


def build_body(
    wire: str,
    model: str,
    mechanism: str,
    messages: list[dict],
    ceiling: str,
    thinking_capable: bool,
) -> dict:
    if wire == "native":
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,  # the Modelfile bakes temperature 1 — override
                "num_ctx": NUM_CTX,
                "num_predict": -1 if ceiling == "unbounded" else int(ceiling),
            },
        }
        if thinking_capable:
            body["think"] = False
        if mechanism == "A1":
            body["format"] = "json"
        elif mechanism == "B":
            body["format"] = TOOL_PARAMETERS
        elif mechanism == "C":
            body["tools"] = [TOOL_DEF]
        return body
    body = {"model": model, "messages": messages, "temperature": 0}
    if ceiling != "unbounded":
        body["max_tokens"] = int(ceiling)
    if mechanism == "A1":
        body["response_format"] = {"type": "json_object"}
    elif mechanism == "B":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "ModelOutput", "schema": SCHEMA, "strict": True},
        }
    elif mechanism == "C":
        body["tools"] = [TOOL_DEF]
        body["tool_choice"] = {"type": "function", "function": {"name": "ModelOutput"}}
    return body


def call(client: httpx.Client, base_url: str, wire: str, body: dict) -> dict:
    url = f"{base_url}/api/chat" if wire == "native" else f"{base_url}/v1/chat/completions"
    started = time.monotonic()
    try:
        response = client.post(url, json=body)
    except httpx.TimeoutException:
        return {"kind": "timeout", "latency_ms": int((time.monotonic() - started) * 1000)}
    except httpx.HTTPError as error:
        return {
            "kind": "transport",
            "error": str(error)[:200],
            "latency_ms": int((time.monotonic() - started) * 1000),
        }
    latency_ms = int((time.monotonic() - started) * 1000)
    if response.status_code != 200:
        return {
            "kind": "http_error",
            "status": response.status_code,
            "error": response.text[:300],
            "latency_ms": latency_ms,
        }
    return {"kind": "ok", "status": 200, "payload": response.json(), "latency_ms": latency_ms}


def extract_answer(wire: str, payload: dict) -> dict:
    """Normalize both wire shapes into one answer record. A tool call's
    arguments become the content under validation — the same thing the SDK
    structured path revalidates."""
    if wire == "native":
        message = payload.get("message") or {}
        finish = payload.get("done_reason")
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
    else:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish = choice.get("finish_reason")
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    had_tool_calls = bool(tool_calls)
    if had_tool_calls:
        arguments = (tool_calls[0].get("function") or {}).get("arguments")
        if isinstance(arguments, str):
            content = arguments
        elif arguments is not None:
            content = json.dumps(arguments, ensure_ascii=False)
    return {
        "content": content,
        "finish_reason": str(finish) if finish is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "had_tool_calls": had_tool_calls,
    }


# -- canary probes ------------------------------------------------------------


def canary(client: httpx.Client, base_url: str, models: list[str]) -> dict:
    report: dict = {"base_url": base_url, "models": {}, "probes": []}
    version = client.get(f"{base_url}/api/version")
    report["ollama_version"] = version.json().get("version") if version.status_code == 200 else None
    for model in models:
        show = client.post(f"{base_url}/api/show", json={"model": model})
        info = show.json() if show.status_code == 200 else {}
        report["models"][model] = {
            "capabilities": info.get("capabilities", []),
            "parameters": info.get("parameters", ""),
            "details": info.get("details", {}),
        }
    for model in models:
        thinking = "thinking" in report["models"][model]["capabilities"]
        for wire in WIRES:
            for mechanism in ["A0", "A1", "B", "C"]:
                body = build_body(wire, model, mechanism, COLORS_PROMPT, "512", thinking)
                # Canary constrains with the tiny Colors schema, not ModelOutput.
                if mechanism == "B":
                    if wire == "native":
                        body["format"] = COLORS_SCHEMA
                    else:
                        body["response_format"] = {
                            "type": "json_schema",
                            "json_schema": {"name": "Colors", "schema": COLORS_SCHEMA, "strict": True},
                        }
                elif mechanism == "C":
                    tool = {
                        "type": "function",
                        "function": {
                            "name": "Colors",
                            "description": "Report the colors.",
                            "parameters": COLORS_SCHEMA,
                        },
                    }
                    body["tools"] = [tool]
                    if wire == "v1":
                        body["tool_choice"] = {"type": "function", "function": {"name": "Colors"}}
                result = call(client, base_url, wire, body)
                probe: dict = {"model": model, "wire": wire, "mechanism": mechanism}
                if result["kind"] != "ok":
                    probe["verdict"] = f"error:{result.get('status', result['kind'])}"
                    probe["evidence"] = result.get("error", "")[:300]
                else:
                    answer = extract_answer(wire, result["payload"])
                    text = answer["content"]
                    probe["evidence"] = text[:300]
                    probe["finish_reason"] = answer["finish_reason"]
                    if mechanism == "C":
                        probe["verdict"] = "yes" if answer["had_tool_calls"] else "no"
                    elif mechanism in ("A1", "B"):
                        try:
                            parsed = json.loads(text)
                            is_json = True
                        except (json.JSONDecodeError, ValueError):
                            parsed, is_json = None, False
                        if mechanism == "B":
                            conforms = is_json and not list(
                                common.jsonschema.Draft202012Validator(COLORS_SCHEMA).iter_errors(parsed)
                            )
                            probe["verdict"] = "yes" if conforms else "no"
                        else:
                            probe["verdict"] = "yes" if is_json else "no"
                    else:  # A0 baseline: record whether prose came back, as expected
                        try:
                            json.loads(text)
                            probe["verdict"] = "baseline-json"
                        except (json.JSONDecodeError, ValueError):
                            probe["verdict"] = "baseline-prose"
                report["probes"].append(probe)
                print(
                    f"canary {model} {wire} {mechanism}: {probe['verdict']}",
                    flush=True,
                )
    return report


def gate(canary_report: dict, model: str, wire: str, mechanism: str) -> bool:
    """True when the canary says this mechanism works on this model/wire."""
    if mechanism not in ("B", "C"):
        return True
    for probe in canary_report["probes"]:
        if (probe["model"], probe["wire"], probe["mechanism"]) == (model, wire, mechanism):
            return probe["verdict"] == "yes"
    return True  # no probe recorded — do not silently skip


# -- the matrix ---------------------------------------------------------------


def build_trials() -> list[dict]:
    trials = []

    def add(model, wire, mech, doc, ceiling, variant, rep):
        trials.append(
            {
                "model": model,
                "wire": wire,
                "mechanism": mech,
                "doc": doc,
                "ceiling": ceiling,
                "variant": variant,
                "rep": rep,
                "key": trial_key(model, wire, mech, doc, ceiling, variant, rep),
            }
        )

    for wire in WIRES:
        for mech in MECHANISMS:
            for doc in CHEAP_DOCS:
                for rep in range(1, 4):
                    add(PRIMARY_MODEL, wire, mech, doc, "2048", "core", rep)
            for doc in EXPENSIVE_DOCS:
                for ceiling in CEILINGS_EXPENSIVE:
                    for rep in range(1, 3):
                        add(PRIMARY_MODEL, wire, mech, doc, ceiling, "core", rep)
    for index in range(5):  # Option D: per-paragraph units of long_dense
        for rep in range(1, 4):
            add(PRIMARY_MODEL, "native", "A0", f"long_dense#p{index}", "2048", "D", rep)
    for rep in range(1, 4):  # Option E: start at 512, escalate on `length`
        add(PRIMARY_MODEL, "native", "A0", "output_limit", "512-esc", "E", rep)
    for wire in WIRES:
        for mech in MECHANISMS:
            for doc in CHEAP_DOCS:
                add(SECONDARY_MODEL, wire, mech, doc, "2048", "secondary", 1)
    return trials


# -- trial execution ----------------------------------------------------------


def run_trial(client: httpx.Client, base_url: str, trial: dict, doc_cache: dict, canary_report: dict, journal) -> str:
    doc = doc_cache.setdefault(trial["doc"], common.load_doc(trial["doc"]))
    system, user = common.build_prompt(doc)
    base = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    thinking = "thinking" in canary_report["models"].get(trial["model"], {}).get("capabilities", [])

    prior_bad: str | None = None
    last_error = ""
    length_limited = False
    trial_started = time.monotonic()
    outcome = {"ok": False, "attempts_used": 0, "failure_class": "parser"}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        messages = list(base)
        if prior_bad is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": x.corrective_assistant_turn_content(prior_bad, None),
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": x.corrective_message(last_error, length_limited, 0),
                }
            )
        if trial["variant"] == "E":
            # Escalate the budget only on a provider-reported cutoff;
            # a plain malformed answer retries at the same 512 ceiling.
            ceiling = "512" if attempt == 1 or not length_limited else "unbounded"
        else:
            ceiling = trial["ceiling"]
        body = build_body(trial["wire"], trial["model"], trial["mechanism"], messages, ceiling, thinking)
        result = call(client, base_url, trial["wire"], body)
        outcome["attempts_used"] = attempt

        record = {
            "type": "attempt",
            "ts": time.time(),
            "trial_key": trial["key"],
            "attempt": attempt,
            "model": trial["model"],
            "wire": trial["wire"],
            "mechanism": trial["mechanism"],
            "doc": trial["doc"],
            "ceiling": trial["ceiling"],
            "effective_ceiling": ceiling,
            "variant": trial["variant"],
            "rep": trial["rep"],
            "latency_ms": result["latency_ms"],
            "http_status": result.get("status"),
        }
        if result["kind"] != "ok":
            record["failure_class"] = "timeout" if result["kind"] == "timeout" else "transport"
            record["transport_error"] = result.get("error", "")
            journal(record)
            outcome["failure_class"] = record["failure_class"]
            break

        answer = extract_answer(trial["wire"], result["payload"])
        verdict = common.validate_response(answer["content"], doc)
        raw_path(trial["key"], attempt).write_text(answer["content"])
        if verdict["syntax_ok"]:
            failure_class = "ok"
        elif x.indicates_length_limit(answer["finish_reason"]):
            failure_class = "length"
        else:
            failure_class = "parser"
        record.update(
            {
                "finish_reason": answer["finish_reason"],
                "prompt_tokens": answer["prompt_tokens"],
                "completion_tokens": answer["completion_tokens"],
                "had_tool_calls": answer["had_tool_calls"],
                "content_truncated": answer["content"][:TRUNCATE_JOURNAL_BYTES],
                "failure_class": failure_class,
                **verdict,
            }
        )
        journal(record)

        if verdict["syntax_ok"]:
            outcome["ok"] = True
            outcome["failure_class"] = "ok"
            break
        outcome["failure_class"] = failure_class
        last_error = verdict["syntax_error"]
        length_limited = x.indicates_length_limit(answer["finish_reason"])
        prior_bad = answer["content"]

    journal(
        {
            "type": "trial",
            "ts": time.time(),
            "trial_key": trial["key"],
            **{k: trial[k] for k in ("model", "wire", "mechanism", "doc", "ceiling", "variant", "rep")},
            "ok": outcome["ok"],
            "attempts_used": outcome["attempts_used"],
            "recovered_by_retry": outcome["ok"] and outcome["attempts_used"] > 1,
            "failure_class": outcome["failure_class"],
            "total_latency_ms": int((time.monotonic() - trial_started) * 1000),
        }
    )
    return outcome["failure_class"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--force-canary", action="store_true", help="re-run probes even if canary.json exists")
    parser.add_argument("--limit", type=int, default=0, help="run at most N pending trials (0 = all)")
    args = parser.parse_args()

    common.self_test()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(timeout=CALL_TIMEOUT_SECS)
    models = [PRIMARY_MODEL, SECONDARY_MODEL]

    if args.force_canary or not CANARY_PATH.exists():
        report = canary(client, args.base_url, models)
        CANARY_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"canary written: {CANARY_PATH}", flush=True)
    else:
        report = json.loads(CANARY_PATH.read_text())
    if args.canary_only:
        return

    done: set[str] = set()
    if ATTEMPTS_PATH.exists():
        for line in ATTEMPTS_PATH.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "trial":
                done.add(record["trial_key"])

    trials = [t for t in build_trials() if t["key"] not in done]
    if args.limit:
        trials = trials[: args.limit]
    total = len(trials)
    print(f"{total} trial(s) pending ({len(done)} already journaled)", flush=True)

    doc_cache: dict = {}
    # Circuit breaker: after 3 consecutive timeout trials on one
    # (model, wire, mechanism), skip the combination's remaining trials —
    # recorded as skipped_stall, never silently. The v1 wire cannot disable
    # thinking, and a thinking model there can pin every call at the 180 s
    # cap; three identical stalls prove the pathology without burning an
    # hour re-proving it.
    STALL_LIMIT = 3
    stalls: dict[tuple, int] = {}
    with ATTEMPTS_PATH.open("a") as sink:

        def journal(record: dict) -> None:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            sink.flush()

        def skip(trial: dict, index: int, failure_class: str, reason: str) -> None:
            journal(
                {
                    "type": "trial",
                    "ts": time.time(),
                    "trial_key": trial["key"],
                    **{k: trial[k] for k in ("model", "wire", "mechanism", "doc", "ceiling", "variant", "rep")},
                    "ok": False,
                    "attempts_used": 0,
                    "recovered_by_retry": False,
                    "failure_class": failure_class,
                    "total_latency_ms": 0,
                }
            )
            print(f"[{index}/{total}] {trial['key']} skipped ({reason})", flush=True)

        for index, trial in enumerate(trials, 1):
            combo = (trial["model"], trial["wire"], trial["mechanism"])
            if not gate(report, trial["model"], trial["wire"], trial["mechanism"]):
                skip(trial, index, "skipped_unsupported", "canary: unsupported")
                continue
            if stalls.get(combo, 0) >= STALL_LIMIT:
                skip(trial, index, "skipped_stall", f"{STALL_LIMIT} consecutive timeouts on {'/'.join(combo)}")
                continue
            started = time.monotonic()
            failure_class = run_trial(client, args.base_url, trial, doc_cache, report, journal)
            stalls[combo] = stalls.get(combo, 0) + 1 if failure_class == "timeout" else 0
            print(
                f"[{index}/{total}] {trial['key']} done in {time.monotonic() - started:.1f}s",
                flush=True,
            )
    print("harness complete", flush=True)


if __name__ == "__main__":
    main()
