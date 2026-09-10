from promptxray.blocks import Block
from promptxray import blocks as b


PROMPT = """You are a classifier.

Answer with one word. [[keep]]

Text: {input}"""


def test_splits_on_blank_lines():
    parsed = b.parse_blocks(PROMPT)
    assert len(parsed) == 3
    assert parsed[0].text.startswith("You are")


def test_keep_marker_pins_block_and_is_stripped():
    parsed = b.parse_blocks(PROMPT)
    assert parsed[1].pinned is True
    assert "[[keep]]" not in parsed[1].text


def test_input_placeholder_pins_block():
    parsed = b.parse_blocks(PROMPT)
    assert parsed[2].pinned is True
    assert parsed[2].pin_reason == "contains {input}"


def test_render_can_skip_a_block():
    parsed = b.parse_blocks(PROMPT)
    rendered = b.render(parsed, skip=0)
    assert "You are a classifier" not in rendered
    assert "{input}" in rendered


def test_validate_rejects_prompt_without_input():
    try:
        b.validate(b.parse_blocks("Just a sentence."))
    except ValueError as exc:
        assert "{input}" in str(exc)
    else:
        raise AssertionError("validate should have raised")


def test_validate_rejects_empty_prompt():
    try:
        b.validate([])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("validate should have raised")


def test_preview_flattens_whitespace_and_truncates():
    long_block = Block(index=0, text="word " * 60, pinned=False)
    preview = long_block.preview
    assert len(preview) <= 90
    assert "\n" not in preview  # single line
    assert preview.endswith("...")
    assert Block(index=0, text="short", pinned=False).preview == "short"


def test_parse_blocks_strips_blank_lines_between_blocks():
    parsed = b.parse_blocks("one\n\n\n\n\n  \ntwo")
    assert [p.text for p in parsed] == ["one", "two"]
