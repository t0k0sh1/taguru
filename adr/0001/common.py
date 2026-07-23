"""Shared validation pipeline for the ADR-0001 evidence harness.

Production fidelity is the whole point of this module, and it is achieved
two different ways:

- Everything the *Python SDK producer* does — the system prompt, paragraph
  labeling, corrective texts, fence stripping, the strict (pydantic) parse,
  the canonical MODEL_OUTPUT_JSON_SCHEMA, and merge()'s business rules — is
  IMPORTED from `taguru_langchain._extract`, not copied, so it cannot
  drift. That module is itself the hand-mirrored twin of src/extract.rs
  (its docstring: "treat the two as one artifact").
- The one behavior the SDK deliberately does NOT share — the *Rust CLI's*
  lenient per-field/per-item deserialization (`lenient_vec`/`lenient_string`
  /`lenient_f64`/`lenient_u32`, src/extract.rs) — is mirrored by hand here
  in `lenient_output()`, because measuring the lenient-vs-strict delta
  between the two producer philosophies is one of the experiment's outputs.

`self_test()` guards both copies at startup: the schema must accept/reject
exactly the checked-in fixtures in tests/fixtures/model_output/, or the
harness refuses to run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import jsonschema

ADR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ADR_DIR.parent.parent


def _load_extract_module():
    """Load taguru_langchain._extract straight from its file: the package
    __init__ pulls in langchain_core, which the harness neither needs nor
    wants as a dependency, while _extract.py itself is stdlib+pydantic."""
    path = REPO_ROOT / "sdk" / "python-langchain" / "src" / "taguru_langchain" / "_extract.py"
    spec = importlib.util.spec_from_file_location("taguru_extract", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves cls.__module__ through here
    spec.loader.exec_module(module)
    return module


x = _load_extract_module()

SCHEMA = x.MODEL_OUTPUT_JSON_SCHEMA
VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)

CORPUS_DIR = ADR_DIR / "corpus"
GOLDEN_DIR = ADR_DIR / "golden"


def self_test() -> None:
    """Fail fast if the canonical schema drifted from the shared fixtures."""
    fixtures = REPO_ROOT / "tests" / "fixtures" / "model_output"
    for path in sorted((fixtures / "accepted").glob("*.json")):
        errors = list(VALIDATOR.iter_errors(json.loads(path.read_text())))
        if errors:
            raise SystemExit(
                f"schema drift: accepted fixture {path.name} fails: {errors[0].message}"
            )
    for path in sorted((fixtures / "rejected").glob("*.json")):
        if not list(VALIDATOR.iter_errors(json.loads(path.read_text()))):
            raise SystemExit(f"schema drift: rejected fixture {path.name} passes")


# -- stage 1: JSON syntax, mirroring src/extract.rs parse_model_output() ------


def _reject_constant(name: str) -> None:
    # serde_json refuses Infinity/NaN; Python's json accepts them by
    # default, which would let the mirror pass what Rust retries on.
    raise ValueError(f"non-standard JSON constant {name}")


@dataclass
class SyntaxResult:
    ok: bool
    value: dict | None
    error: str  # the text the producer would put in the corrective turn
    empty: bool


def parse_syntax(content: str) -> SyntaxResult:
    unfenced = x.strip_fences(content.strip())
    if not unfenced:
        return SyntaxResult(
            False,
            None,
            "the answer was empty — thinking-mode models can burn their whole "
            "budget on reasoning before any text (docs/extract.html: turn "
            "thinking off)",
            True,
        )
    first_error: str
    try:
        value = json.loads(unfenced, parse_constant=_reject_constant)
        if isinstance(value, dict):
            return SyntaxResult(True, value, "", False)
        first_error = f"invalid type: {type(value).__name__}, expected an object"
    except (json.JSONDecodeError, ValueError) as err:
        first_error = str(err)
    start, end = unfenced.find("{"), unfenced.rfind("}")
    if 0 <= start < end:
        try:
            value = json.loads(unfenced[start : end + 1], parse_constant=_reject_constant)
            if isinstance(value, dict):
                return SyntaxResult(True, value, "", False)
        except (json.JSONDecodeError, ValueError):
            pass
    return SyntaxResult(False, None, f"not a JSON object: {first_error}", False)


# -- stage 2: canonical-schema conformance ------------------------------------


def schema_errors(value: dict) -> list[str]:
    errors = []
    for err in VALIDATOR.iter_errors(value):
        where = "/".join(str(part) for part in err.absolute_path) or "$"
        errors.append(f"{where}: {err.message}")
    return errors


# -- stage 3: the Rust CLI's lenient parse, mirrored by hand ------------------


def _lenient_str(value: object) -> str | None:
    # serde_json Value::as_str — only a genuine string survives.
    return value if isinstance(value, str) else None


def _lenient_f64(value: object) -> float | None:
    # Value::as_f64 — any JSON number; bools are not numbers in serde.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _lenient_u32(value: object) -> int | None:
    # Value::as_u64 + u32::try_from — integer-typed, 0..=u32::MAX.
    # A JSON 3.0 arrives as float and is refused, exactly like serde.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 0xFFFF_FFFF else None


def lenient_output(value: dict) -> x.ModelOutput:
    """src/extract.rs semantics: a wrong-shaped field reads as empty, a
    malformed array item costs the item, a wrong-typed scalar costs the
    field — never the chunk."""

    def items(key: str) -> list:
        raw = value.get(key)
        return raw if isinstance(raw, list) else []

    associations = [
        x.ModelAssociation(
            subject=_lenient_str(item.get("subject")),
            label=_lenient_str(item.get("label")),
            object=_lenient_str(item.get("object")),
            weight=_lenient_f64(item.get("weight")),
            paragraph=_lenient_u32(item.get("paragraph")),
        )
        for item in items("associations")
        if isinstance(item, dict)
    ]
    aliases = [
        x.ModelAlias(
            alias=_lenient_str(item.get("alias")),
            canonical=_lenient_str(item.get("canonical")),
            kind=_lenient_str(item.get("kind")),
        )
        for item in items("aliases")
        if isinstance(item, dict)
    ]
    questions = [
        x.ModelQuestion(
            paragraph=_lenient_u32(item.get("paragraph")),
            question=_lenient_str(item.get("question")),
        )
        for item in items("questions")
        if isinstance(item, dict)
    ]
    return x.ModelOutput(associations=associations, aliases=aliases, questions=questions)


# -- stage 4: the SDK producers' strict parse (imported, not mirrored) --------


def strict_parse(content: str) -> tuple[bool, str]:
    try:
        x.parse_model_output(content)
        return True, ""
    except Exception as err:  # the SDK raises ValueError; be safe
        return False, str(err)[:300]


# -- stage 5: golden scoring --------------------------------------------------


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if not ch.isspace())


def score_triples(facts: list, golden: dict) -> dict:
    gold = golden["triples"]
    triple_weights: dict[tuple[str, str, str], float] = {}
    pairs: set[tuple[str, str]] = set()
    for fact in facts:
        key = (normalize(fact.subject), normalize(fact.label), normalize(fact.object))
        triple_weights.setdefault(key, fact.weight)
        pairs.add((key[0], key[2]))
    matched = pair_matched = polarity_checked = polarity_ok = 0
    for item in gold:
        key = (normalize(item["subject"]), normalize(item["label"]), normalize(item["object"]))
        weight = triple_weights.get(key)
        if weight is not None:
            matched += 1
            polarity = item.get("polarity")
            if polarity:
                polarity_checked += 1
                if (weight < 0) == (polarity == "negative"):
                    polarity_ok += 1
        if (key[0], key[2]) in pairs:
            pair_matched += 1
    total = len(gold)
    return {
        "golden_total": total,
        "triple_recall": round(matched / total, 4),
        "pair_recall": round(pair_matched / total, 4),
        "polarity_checked": polarity_checked,
        "polarity_ok": polarity_ok,
    }


def score_alias_groups(aliases: list, golden: dict) -> dict:
    decoys = {normalize(name) for name in golden.get("decoy_names", [])}
    groups_found = surface_hits = surface_total = false_positives = 0
    for group in golden["groups"]:
        members = {normalize(name) for name in group["canonical_any"]}
        members |= {normalize(name) for name in group["surface_forms"]}
        surfaces = [normalize(name) for name in group["surface_forms"]]
        surface_total += len(surfaces)
        hit_any = False
        for surface in surfaces:
            for alias in aliases:
                spelling = normalize(alias.alias or "")
                canonical = normalize(alias.canonical or "")
                if alias.kind == "concept" and spelling == surface and canonical in members:
                    surface_hits += 1
                    hit_any = True
                    break
        groups_found += 1 if hit_any else 0
        for alias in aliases:
            spelling = normalize(alias.alias or "")
            canonical = normalize(alias.canonical or "")
            if (spelling in decoys and canonical in members) or (
                canonical in decoys and spelling in members
            ):
                false_positives += 1
    return {
        "alias_groups_total": len(golden["groups"]),
        "alias_groups_found": groups_found,
        "alias_surface_recall": round(surface_hits / surface_total, 4) if surface_total else None,
        "alias_false_positives": false_positives,
    }


# -- documents ----------------------------------------------------------------


@dataclass
class Doc:
    name: str  # matrix key, e.g. "long_dense" or "long_dense#p1"
    source: str  # what the prompt's Document '<source>' shows
    text: str  # the extraction unit's raw text
    paragraph_count: int
    golden: dict | None


def load_doc(name: str) -> Doc:
    if "#p" in name:
        # Option D: one paragraph presented as its own extraction unit.
        base, _, index_text = name.partition("#p")
        index = int(index_text)
        full = (CORPUS_DIR / f"{base}.md").read_text()
        paragraph = x.split_paragraphs(full)[index]
        golden = _load_golden(base)
        if golden and golden.get("match") == "exact_triple":
            golden = {
                "match": "exact_triple",
                "triples": [t for t in golden["triples"] if t.get("paragraph") == index],
            }
        return Doc(name, f"{base}.md#p{index}", paragraph, 1, golden)
    text = (CORPUS_DIR / f"{name}.md").read_text()
    return Doc(name, f"{name}.md", text, len(x.split_paragraphs(text)), _load_golden(name))


def _load_golden(name: str) -> dict | None:
    path = GOLDEN_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def build_prompt(doc: Doc) -> tuple[str, str]:
    """Exactly what the production producers send for this unit: empty
    vocabulary, no doc2query questions, no fact budget — today's defaults."""
    system = x.system_prompt([], 0, 0)
    labeled = x.labeled_document(doc.text, x.CHUNK_BYTES)
    user = x.user_message(doc.source, 0, 1, labeled)
    return system, user


# -- the full per-attempt verdict ---------------------------------------------


def validate_response(content: str, doc: Doc) -> dict:
    syntax = parse_syntax(content)
    record: dict = {
        "syntax_ok": syntax.ok,
        "syntax_error": syntax.error if not syntax.ok else "",
        "empty_answer": syntax.empty,
        "schema_valid": None,
        "schema_error_count": None,
        "schema_first_errors": [],
        "lenient": None,
        "strict_ok": None,
        "strict_error": "",
        "golden": None,
    }
    strict_ok, strict_error = strict_parse(content)
    record["strict_ok"] = strict_ok
    record["strict_error"] = strict_error
    if not syntax.ok:
        return record
    errors = schema_errors(syntax.value)
    record["schema_valid"] = not errors
    record["schema_error_count"] = len(errors)
    record["schema_first_errors"] = errors[:3]
    output = lenient_output(syntax.value)
    extraction = x.merge([output], 0, doc.paragraph_count)
    record["lenient"] = {
        "parsed_associations": len(output.associations),
        "parsed_aliases": len(output.aliases),
        "parsed_questions": len(output.questions),
        "merged_associations": len(extraction.associations),
        "merged_aliases": len(extraction.concepts) + len(extraction.labels),
        "duplicates": extraction.duplicates,
        "dropped": extraction.dropped,
    }
    if doc.golden:
        if doc.golden["match"] == "exact_triple":
            record["golden"] = score_triples(extraction.associations, doc.golden)
        elif doc.golden["match"] == "alias_group_recall":
            record["golden"] = score_alias_groups(output.aliases, doc.golden)
    return record
