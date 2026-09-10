"""Tests for the on-disk answer cache: reuse, management, and concurrency."""
import threading

from promptxray.cache import Cache
from promptxray.providers import Answer


def test_get_returns_none_on_miss(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    assert cache.get("m", "prompt") is None
    assert cache.stats["misses"] == 1
    cache.close()


def test_roundtrip_then_reuse(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    answer = Answer(text="spam", input_tokens=8, output_tokens=1)
    cache.put("model-a", "one prompt", answer)
    got = cache.get("model-a", "one prompt")
    assert got is not None
    assert got.text == "spam"
    assert got.input_tokens == 8
    # same model + prompt -> hit; hits counters both increment
    assert cache.get("model-a", "one prompt") is not None
    assert cache.stats["hits"] == 2
    cache.close()


def test_model_and_prompt_are_part_of_the_key(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    cache.put("model-a", "same text", Answer(text="x"))
    assert cache.get("model-b", "same text") is None  # different model
    assert cache.get("model-a", "different text") is None  # different prompt
    assert cache.get("model-a", "same text") is not None
    cache.close()


def test_disabled_cache_never_touches_disk(tmp_path):
    cache = Cache(tmp_path / "c.sqlite", enabled=False)
    assert cache.get("m", "p") is None
    cache.put("m", "p", Answer(text="x"))
    # nothing persisted; stats path is null
    assert cache.stats["entries"] == 0
    cache.close()


def test_clear_removes_entries_and_reports_count(tmp_path):
    path = tmp_path / "c.sqlite"
    cache = Cache(path)
    cache.put("m", "p1", Answer(text="a"))
    cache.put("m", "p2", Answer(text="b"))
    removed = cache.clear()
    assert removed == 2
    assert cache.get("m", "p1") is None
    assert cache.stats["entries"] == 0
    cache.close()


def test_thread_safe_counters_under_concurrency(tmp_path):
    """Many threads hitting get() must not lose counter increments."""
    cache = Cache(tmp_path / "c.sqlite")
    cache.put("m", "seed", Answer(text="x"))

    def worker(_):
        for _ in range(200):
            cache.get("m", "seed")  # all hits

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # The lock keeps hits == number of successful lookups, no torn reads.
    assert cache.stats["hits"] == 8 * 200
    assert cache.stats["misses"] == 0
    cache.close()


def test_path_attribute_exposed(tmp_path):
    p = tmp_path / "c.sqlite"
    cache = Cache(p)
    assert cache.path == str(p)
    cache.close()