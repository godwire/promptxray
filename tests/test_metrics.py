from promptxray.metrics import UNPARSED, normalise, score

LABELS = ["spam", "normal"]


def test_normalise_handles_noise_around_the_label():
    assert normalise("  Spam. ", LABELS) == "spam"
    assert normalise('"normal"', LABELS) == "normal"
    assert normalise("The label is spam", LABELS) == "spam"
    assert normalise("I cannot help with that", LABELS) == UNPARSED


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
