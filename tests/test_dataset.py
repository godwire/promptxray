"""Tests for the dataset loader and subset builder."""
import pytest

from promptxray import dataset as d
from promptxray.dataset import Example


def test_load_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label\nhello,greeting\ngoodbye,farewell\n", encoding="utf-8")
    examples = d.load(p)
    assert [e.id for e in examples] == [0, 1]
    assert examples[0].text == "hello"
    assert examples[1].label == "farewell"


def test_load_jsonl(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"text": "hi", "label": "greet"}\n{"text": "bye", "label": "leave"}\n',
                 encoding="utf-8")
    examples = d.load(p)
    assert len(examples) == 2
    assert examples[0].label == "greet"


def test_load_custom_columns(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("message,category\na,b\n", encoding="utf-8")
    examples = d.load(p, text_column="message", label_column="category")
    assert examples[0].text == "a"
    assert examples[0].label == "b"


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        d.load(tmp_path / "nope.csv")


def test_load_missing_columns_raises(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text\nonly-text\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing column"):
        d.load(p)


def test_load_blank_rows_are_skipped(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label\n,hi\nok,same\n,,\n", encoding="utf-8")
    examples = d.load(p)
    assert len(examples) == 1
    assert examples[0].text == "ok"


def test_load_raises_when_every_row_is_blank(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("text,label\n,\n,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable rows"):
        d.load(p)


def test_labels_of_is_sorted_distinct():
    ex = [Example(id=0, text="x", label="b"),
          Example(id=1, text="y", label="a"),
          Example(id=2, text="z", label="b")]
    assert d.labels_of(ex) == ["a", "b"]


def test_error_enriched_subset_includes_all_wrong_and_fills_the_rest():
    examples = [Example(id=i, text=str(i), label="x") for i in range(10)]
    wrong = {1, 2, 3}
    subset = d.error_enriched_subset(examples, wrong, size=6)
    subset_ids = {e.id for e in subset}
    assert wrong <= subset_ids  # every wrong example is included
    assert len(subset) == 6


def test_error_enriched_subset_budget_larger_than_dataset_returns_all():
    examples = [Example(id=i, text=str(i), label="x") for i in range(5)]
    subset = d.error_enriched_subset(examples, {1}, size=100)
    assert len(subset) == 5


def test_error_enriched_subset_zero_or_negative_size_returns_all():
    examples = [Example(id=i, text=str(i), label="x") for i in range(5)]
    assert len(d.error_enriched_subset(examples, {1}, size=0)) == 5
    assert len(d.error_enriched_subset(examples, {1}, size=-3)) == 5


def test_error_enriched_subset_is_deterministic():
    examples = [Example(id=i, text=str(i), label="x") for i in range(20)]
    a = d.error_enriched_subset(examples, {5}, size=8, seed=42)
    b = d.error_enriched_subset(examples, {5}, size=8, seed=42)
    assert [e.id for e in a] == [e.id for e in b]