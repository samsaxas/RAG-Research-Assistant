"""
Small text helpers shared by the parser, the evaluation harness and the label validator.

`normalize` reduces text to lowercase alphanumeric words so that two strings can be compared even if one
came out of a PDF (line breaks, curly quotes, hyphenation, odd spacing) and the other was typed by hand.
"""
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HYPHEN = re.compile(r"-\s*")  # a hyphen and any whitespace/newline right after it


def normalize(text: str) -> str:
    """Lowercase, drop hyphens/soft hyphens, turn every other non-alphanumeric run into one space.

    Hyphens are *removed* (not replaced by a space) on purpose: PDF extraction sometimes keeps "human-cognitive"
    and sometimes yields "humancognitive" after a line break, and removing hyphens on both sides makes the two
    spellings compare equal.
    """
    text = _HYPHEN.sub("", text.lower().replace("\u00ad", ""))
    return _NON_ALNUM.sub(" ", text).strip()


def contains_phrase(haystack_norm: str, phrase: str) -> bool:
    """True if `phrase` occurs in an already-normalised haystack, matching whole words only."""
    needle = normalize(phrase)
    return bool(needle) and f" {needle} " in f" {haystack_norm} "


def contains_keywords(answer: str, keywords) -> bool:
    """True if every keyword group is found in `answer` (case/punctuation-insensitive substring match).

    A group may offer alternatives separated by '|', e.g. "cease|stop|halt" is satisfied by any of the three.
    Substring (not whole-word) matching is deliberate so that "transfer" matches "transferring".
    """
    if isinstance(keywords, str):
        keywords = [keywords]
    haystack = normalize(answer)
    for group in keywords:
        alternatives = [normalize(alt) for alt in group.split("|")]
        if not any(alt and alt in haystack for alt in alternatives):
            return False
    return True
