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
