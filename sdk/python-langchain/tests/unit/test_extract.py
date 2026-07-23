"""Ports of src/extract.rs's own tests — the cross-language regression floor.

Same inputs, same surviving/dropped counts as the Rust golden tests, so the
two producers can never silently diverge on the batch contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from taguru_langchain._extract import (
    MODEL_OUTPUT_JSON_SCHEMA,
    ModelAlias,
    ModelAssociation,
    ModelOutput,
    ModelQuestion,
    chunk,
    corrective_assistant_turn_content,
    corrective_message,
    indicates_length_limit,
    labeled_document,
    merge,
    parse_model_output,
    render_batch,
    split_paragraphs,
    system_prompt,
)

# sdk/python-langchain/tests/unit/test_extract.py -> repo root: same depth
# tests/integration/conftest.py's REPO_ROOT climbs (unit, tests,
# python-langchain, sdk).
FIXTURES_ROOT = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "model_output"


def association(subject: str, label: str, object_: str, weight: float) -> ModelAssociation:
    return ModelAssociation(subject=subject, label=label, object=object_, weight=weight)


def alias(spelling: str, canonical: str, kind: str) -> ModelAlias:
    return ModelAlias(alias=spelling, canonical=canonical, kind=kind)


def test_merge_folds_duplicates_and_drops_what_the_contract_refuses() -> None:
    """Port of extract.rs merge_folds_duplicates_and_drops_what_the_contract_refuses."""
    merged = merge(
        [
            ModelOutput(
                associations=[
                    ModelAssociation(
                        subject="青嶺酒造", label="杜氏", object="高瀬", weight=1.0, paragraph=0
                    ),
                    association("", "杜氏", "高瀬", 1.0),  # empty name
                    association("蔵", "重い", "石", 1e300),  # over the weight cap
                    association("蔵", "無", "石", 0.0),  # zero asserts nothing
                ],
                aliases=[alias("Aomine", "青嶺酒造", "concept")],
            ),
            ModelOutput(
                associations=[
                    # The exact triple again: folded, first weight kept.
                    association("青嶺酒造", "杜氏", "高瀬", 2.0),
                    ModelAssociation(
                        subject="青嶺酒造",
                        label="創業年",
                        object="1907年",
                        weight=1.0,
                        paragraph=99,  # out of range for a 2-paragraph document
                    ),
                ],
                aliases=[
                    alias("Aomine", "青嶺酒造", "concept"),  # same pair again
                    alias("蔵元", "存在しない", "concept"),  # canonical unknown
                    alias("高瀬", "青嶺酒造", "concept"),  # shadows a real name
                    alias("青嶺酒造", "青嶺酒造", "concept"),  # self
                    alias("x", "青嶺酒造", "banana"),  # unknown kind
                    alias("設立年", "創業年", "label"),  # canonical among labels
                ],
            ),
        ],
        0,
        2,
    )
    assert len(merged.associations) == 2
    assert merged.associations[0].weight == 1.0  # the surviving copy is chunk 0's
    assert merged.associations[0].paragraph == 0
    # Out-of-range self-reports cost only the tag: the fact survives.
    assert merged.associations[1].paragraph is None
    assert merged.concepts == {"Aomine": "青嶺酒造"}
    assert merged.labels == {"設立年": "創業年"}
    assert merged.duplicates == 2  # one triple, one alias pair
    assert merged.dropped == 7
    assert "杜氏" in merged.label_vocabulary()
    assert "創業年" in merged.label_vocabulary()


def test_merge_trims_names_so_whitespace_variants_fold() -> None:
    """Port of extract.rs merge_trims_names_so_whitespace_variants_fold."""
    merged = merge(
        [
            ModelOutput(
                associations=[
                    association("  青嶺酒造  ", "杜氏", "高瀬", 1.0),
                    association("青嶺酒造", "杜氏", "高瀬", 2.0),
                ],
                aliases=[alias("  Aomine  ", "  青嶺酒造  ", "concept")],
            )
        ],
        0,
        0,
    )
    assert len(merged.associations) == 1
    assert merged.associations[0].subject == "青嶺酒造"
    assert merged.associations[0].weight == 1.0
    assert merged.duplicates == 1
    assert merged.concepts == {"Aomine": "青嶺酒造"}


def test_merge_validates_questions_against_the_canonical_paragraph_count() -> None:
    merged = merge(
        [
            ModelOutput(
                questions=[
                    ModelQuestion(paragraph=0, question="最初の質問?"),
                    ModelQuestion(paragraph=0, question="最初の質問?"),  # duplicate
                    ModelQuestion(paragraph=0, question="二つ目の質問?"),  # over cap 1
                    ModelQuestion(paragraph=9, question="範囲外?"),
                    ModelQuestion(paragraph=1, question=""),  # empty
                ]
            )
        ],
        1,
        2,
    )
    assert merged.questions == [(0, "最初の質問?")]
    assert merged.duplicates == 1
    assert merged.dropped == 3


def test_cap_dropped_questions_are_not_mistaken_for_duplicates_on_repeat() -> None:
    """Port of extract.rs cap_dropped_questions_are_not_mistaken_for_duplicates_on_repeat.

    A question the per-paragraph cap drops must not register as seen —
    every document chunk sees the same paragraph list and independently
    proposes questions for it, so an identical question re-proposed by a
    later chunk is a realistic occurrence. Before the fix it read as a
    *duplicate* on the repeat, mislabeling the paragraph's overflow as
    deduplication instead of the cap that caused it.
    """
    first_chunk = ModelOutput(
        questions=[
            ModelQuestion(paragraph=0, question="質問A"),
            ModelQuestion(paragraph=0, question="質問B"),  # over this run's N=1
        ]
    )
    second_chunk = ModelOutput(
        questions=[
            ModelQuestion(paragraph=0, question="質問B"),  # re-proposed, still over the cap
        ]
    )
    merged = merge([first_chunk, second_chunk], 1, 1)
    assert merged.questions == [(0, "質問A")]
    assert merged.duplicates == 0, "the repeat is still a cap drop, not a duplicate"
    assert merged.dropped == 2


def test_chunks_split_at_paragraph_boundaries_and_survive_multibyte_walls() -> None:
    """Port of extract.rs chunks_split_at_paragraph_boundaries_and_survive_multibyte_walls."""
    text = "第一段落。\n\n第二段落。\n\n第三段落。"
    assert chunk(text, 1000) == [text]
    split = chunk(text, 20)
    assert len(split) == 3
    assert all(len(piece.encode("utf-8")) <= 20 for piece in split)

    # A single oversized paragraph hard-splits without slicing a multibyte
    # char, and loses nothing.
    wall = "あ" * 30
    pieces = chunk(wall, 32)
    assert len(pieces) > 1
    assert all(len(piece.encode("utf-8")) <= 32 for piece in pieces)
    assert "".join(pieces) == wall

    assert chunk("   \n\n  ", 100) == []


def test_split_paragraphs_mirrors_the_server_split() -> None:
    """Ports of src/paragraph.rs's own tests."""
    text = "\n最初の段落。\n二行目も同じ段落。\n\n \t \n次の段落。\n\n"
    assert split_paragraphs(text) == ["最初の段落。\n二行目も同じ段落。", "次の段落。"]
    assert split_paragraphs("a\r\nb\r\n\r\nc\r\n") == ["a\r\nb", "c"]
    assert split_paragraphs("") == []
    assert split_paragraphs("\n\n\n") == []
    assert split_paragraphs("  \n　\n") == []  # ideographic space is blank
    assert split_paragraphs("一行だけ。") == ["一行だけ。"]
    assert split_paragraphs("一行だけ。\n") == ["一行だけ。"]


def test_split_paragraphs_does_not_treat_information_separators_as_blank() -> None:
    """Port of paragraph.rs's split_paragraphs_does_not_treat_information_separators_as_blank.

    str.isspace() treats U+001C-001F (the Unicode "information separator"
    controls FS/GS/RS/US) as whitespace; Unicode's White_Space property,
    which src/paragraph.rs's char::is_whitespace() follows, does not. Before
    the fix, a line made of only one of these blanked here and split a
    paragraph the server keeps whole, drifting every paragraph index after
    it between the model's view of the document and the server's.
    """
    text = "最初の段落。\n\x1e\n続き。\n\n次の段落。"
    assert split_paragraphs(text) == ["最初の段落。\n\x1e\n続き。", "次の段落。"]
    # Every one of U+001C-U+001F, alone or amid real whitespace, is content.
    for control in "\x1c\x1d\x1e\x1f":
        assert split_paragraphs(f"a\n{control}\nb") == [f"a\n{control}\nb"]
        assert split_paragraphs(f"a\n  {control}  \nb") == [f"a\n  {control}  \nb"]


def test_labeled_document_numbers_the_canonical_paragraphs() -> None:
    text = "一段落目。\n\n二段落目。"
    # A cap that dwarfs the paragraphs leaves the numbering untouched.
    assert labeled_document(text, 10_000) == "[0] 一段落目。\n\n[1] 二段落目。"


def test_an_oversized_paragraph_repeats_its_number_on_every_continuation() -> None:
    """Port of extract.rs an_oversized_paragraph_repeats_its_number_on_every_continuation.

    One paragraph far larger than the cap: split at its interior line
    breaks, every piece must still name paragraph 0 so the model can
    attribute a question drawn from any of them. The old label-then-byte-
    split left every piece past the first unlabeled.
    """
    body = "あ\n" * 40
    cap = (len(b"[0] ") + len(body.encode())) // 3
    labeled = labeled_document(body, cap)
    blocks = labeled.split("\n\n")
    assert len(blocks) > 1, labeled
    assert all(block.startswith("[0] ") for block in blocks), labeled

    # chunk() packs the pre-sized blocks without re-splitting, so the label
    # survives to what the model sees: every \n\n-delimited block in every
    # chunk still opens with the paragraph number.
    chunks = chunk(labeled, cap)
    assert all(block.startswith("[0] ") for piece in chunks for block in piece.split("\n\n")), (
        chunks
    )


def test_parse_model_output_tolerates_fences_and_prose() -> None:
    payload = '{"associations": [{"subject": "a", "label": "b", "object": "c", "weight": 1.0}]}'
    assert len(parse_model_output(payload).associations) == 1
    assert len(parse_model_output(f"```json\n{payload}\n```").associations) == 1
    assert len(parse_model_output(f"Here you go:\n{payload}\nHope that helps!").associations) == 1
    try:
        parse_model_output("")
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("empty answers must raise")
    try:
        parse_model_output("no json here")
    except ValueError as error:
        assert "not a JSON object" in str(error)
    else:
        raise AssertionError("non-JSON answers must raise")


def test_rendered_batches_carry_the_import_line_shapes() -> None:
    """Port of extract.rs rendered_batches_pass_the_import_parser (shape level)."""
    extraction = merge(
        [
            ModelOutput(
                associations=[association("青嶺酒造", "杜氏", "高瀬", 2.0)],
                aliases=[alias("Aomine", "青嶺酒造", "concept")],
                questions=[ModelQuestion(paragraph=1, question="二行目には何が書いてある?")],
            )
        ],
        2,
        2,
    )
    body = render_batch(
        "sake", "docs/aomine.md", "酒蔵の記憶", extraction, "一段落目。\n\n二段落目。"
    )
    lines = [json.loads(line) for line in body.strip().split("\n")]
    # header, passage, question, fact, alias — one line each.
    assert len(lines) == 5
    assert lines[0] == {
        "taguru_batch": 1,
        "context": "sake",
        "source": "docs/aomine.md",
        "create": {"description": "酒蔵の記憶"},
    }
    assert lines[1] == {"passage": "一段落目。\n\n二段落目。"}
    assert lines[2] == {"paragraph": 1, "question": "二行目には何が書いてある?"}
    assert lines[3] == {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0}
    assert lines[4] == {"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}


def test_render_strips_paragraph_locators_without_a_passage() -> None:
    extraction = merge(
        [
            ModelOutput(
                associations=[
                    ModelAssociation(subject="a", label="b", object="c", weight=1.0, paragraph=0)
                ]
            )
        ],
        0,
        1,
    )
    body = render_batch("ctx", "src", None, extraction, None)
    lines = [json.loads(line) for line in body.strip().split("\n")]
    assert len(lines) == 2  # header + fact; no passage line
    assert "paragraph" not in lines[1]


def test_the_system_prompt_omits_the_fact_budget_clause_by_default() -> None:
    """Port of extract.rs the_system_prompt_omits_the_fact_budget_clause_by_default."""
    assert "association(s) total" not in system_prompt([], 0, 0)


def test_the_system_prompt_states_the_fact_budget_when_set() -> None:
    """Port of extract.rs the_system_prompt_states_the_fact_budget_when_set."""
    prompt = system_prompt([], 0, 5)
    assert "at most 5 association(s) total" in prompt


def test_corrective_assistant_turn_content_replays_in_full_by_default() -> None:
    """Port of extract.rs corrective_assistant_turn_replays_in_full_by_default."""
    assert corrective_assistant_turn_content("not json at all", None) == "not json at all"


def test_corrective_assistant_turn_content_omits_at_a_zero_cap() -> None:
    """Port of extract.rs corrective_assistant_turn_omits_at_a_zero_cap."""
    assert (
        corrective_assistant_turn_content("not json at all", 0)
        == "[omitted: not the requested JSON object]"
    )


def test_corrective_assistant_turn_content_truncates_at_a_char_boundary_under_a_cap() -> None:
    """Port of extract.rs corrective_assistant_turn_truncates_at_a_char_boundary_under_a_cap.

    The cap (3) lands one byte inside "…" (a 3-byte character starting at
    byte 2); truncation must back off to the char boundary instead of
    splitting it or raising.
    """
    assert corrective_assistant_turn_content("ab…cd", 3) == "ab… [truncated to 3 bytes]"


def test_corrective_assistant_turn_content_leaves_content_under_the_cap_untouched() -> None:
    """Port of extract.rs corrective_assistant_turn_leaves_content_under_the_cap_untouched."""
    assert corrective_assistant_turn_content("short", 1000) == "short"


def test_indicates_length_limit_is_true_only_for_output_cap_reasons() -> None:
    """Port of extract.rs indicates_length_limit_is_true_only_for_output_cap_reasons."""
    assert indicates_length_limit("length")
    assert indicates_length_limit("max_tokens")
    assert not indicates_length_limit("stop")
    assert not indicates_length_limit("content_filter")
    assert not indicates_length_limit(None)


def test_corrective_message_matches_todays_fixed_text_when_not_length_limited() -> None:
    """Port of extract.rs corrective_message_matches_todays_fixed_text_when_not_length_limited."""
    message = corrective_message("bad json", False, 0)
    assert message == (
        "That was not the single JSON object asked for (bad json). "
        "Answer again with only the JSON object."
    )
    # A fact budget is irrelevant to the ordinary ask — the model wasn't cut
    # off, so there's nothing to shorten.
    assert message == corrective_message("bad json", False, 5)


def test_corrective_message_asks_for_shorter_when_length_limited() -> None:
    """Port of extract.rs corrective_message_asks_for_shorter_when_length_limited."""
    message = corrective_message("bad json", True, 0)
    assert "SHORTER" in message
    assert "bad json" in message
    assert "association(s) total" not in message


def test_corrective_message_names_the_fact_budget_when_length_limited_and_set() -> None:
    """Port of extract.rs corrective_message_names_the_fact_budget_when_length_limited_and_set."""
    message = corrective_message("bad json", True, 5)
    assert "Keep it to at most 5 association(s) total." in message


def test_json_schema_accepts_and_rejects_the_shared_fixtures() -> None:
    """MODEL_OUTPUT_JSON_SCHEMA against tests/fixtures/model_output — the same
    corpus the Rust and TypeScript copies validate against, so the three
    mirrored schemas cannot silently drift apart."""
    validator = jsonschema.Draft202012Validator(MODEL_OUTPUT_JSON_SCHEMA)

    accepted_paths = sorted((FIXTURES_ROOT / "accepted").glob("*.json"))
    assert accepted_paths, "the accepted fixture directory must not be empty"
    for path in accepted_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(payload))
        assert not errors, f"{path.name} should validate against the schema: {errors}"
        # The schema's accepted set is meant to sit inside
        # parse_model_output's — every fixture the schema takes must also be
        # a real model answer.
        parse_model_output(json.dumps(payload))

    rejected_paths = sorted((FIXTURES_ROOT / "rejected").glob("*.json"))
    assert rejected_paths, "the rejected fixture directory must not be empty"
    for path in rejected_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert not validator.is_valid(payload), (
            f"{path.name} should NOT validate against the schema"
        )
