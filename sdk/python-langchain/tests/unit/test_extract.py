"""Ports of src/extract.rs's own tests — the cross-language regression floor.

Same inputs, same surviving/dropped counts as the Rust golden tests, so the
two producers can never silently diverge on the batch contract.
"""

from __future__ import annotations

import json

from taguru_langchain._extract import (
    ModelAlias,
    ModelAssociation,
    ModelOutput,
    ModelQuestion,
    chunk,
    labeled_document,
    merge,
    parse_model_output,
    render_batch,
    split_paragraphs,
)


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
