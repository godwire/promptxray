"""Regression tests for the normalise() label matcher and score()."""

from promptxray.metrics import UNPARSED, ClassScore, normalise, score

LABELS = ["spam", "normal"]


def test_normalise_handles_noise_around_the_label():
    assert normalise("  Spam. ", LABELS) == "spam"
    assert normalise('"normal"', LABELS) == "normal"
    assert normalise("The label is spam", LABELS) == "spam"
    assert normalise("I cannot help with that", LABELS) == UNPARSED


def test_substring_match_is_word_boundary_aware():
    # "cat" must not match inside a longer word.
    assert normalise("concatenate", ["cat", "dog"]) == UNPARSED
    assert normalise("let the cat out", ["cat", "dog"]) == "cat"
    assert normalise("my dog is nice", ["cat", "dog"]) == "dog"


def test_longest_label_wins_over_a_contained_label():
    # "not urgent" is more specific than "urgent" and must win either ordering.
    labels = ["urgent", "not urgent"]
    assert normalise("not urgent", labels) == "not urgent"
    assert normalise("not urgent", ["not urgent", "urgent"]) == "not urgent"
    assert normalise("this is an urgent matter", ["urgent", "not urgent"]) == "urgent"
    # word boundary: "urgently" is not "urgent"
    assert normalise("please handle urgently", ["urgent", "not urgent"]) == UNPARSED


def test_case_insensitive_matching():
    assert normalise("SPAM", LABELS) == "spam"
    assert normalise("The Spam Folder", LABELS) == "spam"


def test_perfect_predictions():
    gold = {0: "spam", 1: "normal"}
    result = score(gold, {0: "spam", 1: "normal"}, LABELS)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.wrong_ids == set()


def test_counts_false_positives_on_the_predicted_class():
    gold = {0: "spam", 1: "normal"}
    result = score(gold, {0: "normal", 1: "normal"}, LABELS)
    assert result.per_class["normal"].false_positives == 1
    assert result.per_class["spam"].false_negatives == 1
    assert result.wrong_ids == {0}


def test_unparsed_answers_are_counted_not_dropped():
    gold = {0: "spam"}
    result = score(gold, {0: UNPARSED}, LABELS)
    assert result.unparsed == 1
    assert result.accuracy == 0.0


def test_missing_prediction_counts_as_unparsed():
    gold = {0: "spam", 1: "normal"}
    result = score(gold, {0: "spam"}, LABELS)  # id 1 has no prediction
    assert result.unparsed == 1
    assert result.per_class["normal"].support == 1
    assert result.per_class["normal"].false_negatives == 1


def test_empty_gold_scores_zero_without_dividing_by_zero():
    result = score({}, {}, LABELS)
    assert result.accuracy == 0.0
    assert result.macro_f1 == 0.0
    assert result.correct == 0


def test_class_score_properties_guard_divide_by_zero():
    cls = ClassScore(label="x")
    assert cls.precision == 0.0
    assert cls.recall == 0.0
    assert cls.f1 == 0.0


def test_confusion_matrix_rows_and_columns():
    gold = {0: "spam", 1: "spam", 2: "normal"}
    predicted = {0: "spam", 1: "normal", 2: "spam"}
    result = score(gold, predicted, LABELS)
    # row spam -> {spam:1, normal:1}
    assert result.confusion["spam"]["spam"] == 1
    assert result.confusion["spam"]["normal"] == 1
    # row normal -> {spam:1}
    assert result.confusion["normal"]["spam"] == 1
    assert result.confusion["normal"]["normal"] == 0