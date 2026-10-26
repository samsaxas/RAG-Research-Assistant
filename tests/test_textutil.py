import pytest

from textutil import contains_keywords, contains_phrase, normalize


def test_normalize_ignores_case_punctuation_quotes_and_hyphens():
    assert normalize("Human-cognitive “biases”,\n  3 kinds") == "humancognitive biases 3 kinds"
    assert normalize("human-\ncognitive") == normalize("humancognitive") == "humancognitive"


def test_contains_phrase_matches_whole_words_only():
    hay = normalize("The risk tolerance is not prescribed; riskless is not a word here.")
    assert contains_phrase(hay, "Risk  Tolerance")
    assert not contains_phrase(hay, "risk toler")          # partial word
    assert not contains_phrase(hay, "")                     # empty phrase never matches


@pytest.mark.parametrize("answer, keywords, expected", [
    ("Options: mitigating, transferring, avoiding or accepting.", ["mitigat", "transfer", "avoid", "accept"], True),
    ("It is GOVERN 2.3.", ["2.3"], True),
    ("Stop development and deployment.", ["cease|stop|halt"], True),
    ("Nothing relevant.", ["cease|stop|halt"], False),
    ("The go/no-go decision.", "no-go", True),
    ("Only systemic bias.", ["systemic", "statistical"], False),   # every group must match
])
def test_contains_keywords(answer, keywords, expected):
    assert contains_keywords(answer, keywords) is expected
