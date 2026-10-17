"""Excerpts for the model, and a check that an answer is backed by the excerpts it cites.

The model sees the excerpts as a numbered list and cites them as ``[1]`` or ``[1][2]``. The check is plain code, with
no model: every number must be an excerpt that exists, and every sentence that cites must share its content words with
the excerpts it cites. It catches an invented or misplaced citation. It cannot judge meaning, and it cannot compare
text in two different scripts, so a sentence in another script than its excerpt is only checked for its numbers.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document

# How many characters of an excerpt the model is shown, which keeps a request within the token limit of a key.
EXCERPT_CHARS = 1500

# The share of the content words of a cited sentence that must appear in the excerpts it cites.
MIN_SUPPORT = 0.5

# A sentence with fewer content words than this says too little to be checked.
MIN_WORDS = 3

# Words shorter than this are left out of the comparison (articles, prepositions), except numbers.
MIN_WORD_LENGTH = 4

# Words are compared by their first letters, so that "duplicate" and "duplicates" match.
STEM_LENGTH = 5

NO_CITATION_PROBLEM = "The answer cites no excerpt."

# Characters are given by their code points, because some of them are invisible or look like others.
ELLIPSIS = chr(0x2026)
_ARABIC_QUESTION_MARK = chr(0x061F)
_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F))  # Arabic and Arabic Supplement
_LATIN_RANGE = (0x00C0, 0x024F)  # Latin letters with accents, besides the ASCII ones

_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A sentence ends at a full stop, a question or exclamation mark or at a line break, but not before a citation, so
# that "... nothing. [1]" stays one sentence.
_SENTENCE_BREAK = re.compile(rf"(?<=[.!?{_ARABIC_QUESTION_MARK}])\s+(?!\[)|\n+")
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Citation:
    """An excerpt that an answer cites.

    Attributes:
        number: The number the answer cites it by, counting from 1.
        url: The address of the source it came from.
        title: The title of the source.
        heading: The heading of the section, or an empty string.
        text: The text of the excerpt.
    """

    number: int
    url: str
    title: str
    heading: str
    text: str


@dataclass(frozen=True)
class Verification:
    """The result of checking an answer against the excerpts.

    Attributes:
        valid: ``True`` when the answer cites at least one excerpt and every citation holds.
        citations: The excerpts the answer cites, by number. Empty when the answer is not valid.
        problem: What is wrong with the answer. ``None`` when it is valid.
    """

    valid: bool
    citations: tuple[Citation, ...] = ()
    problem: str | None = None


def format_excerpts(documents: Sequence[Document], max_chars: int = EXCERPT_CHARS) -> str:
    """Number the excerpts and label each with the title and section it came from.

    Args:
        documents: The excerpts, best first. Their metadata has the ``url``, ``title`` and ``heading``.
        max_chars: The most characters of each excerpt to show; a longer one is cut and ends with an ellipsis.

    Returns:
        The excerpts as one text, blocks separated by a blank line. An empty text when there are none.
    """
    blocks = []
    for number, document in enumerate(documents, start=1):
        text = document.page_content
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + ELLIPSIS
        blocks.append(f"[{number}] {_label(document)}\n{text}")
    return "\n\n".join(blocks)


def verify_answer(answer: str, documents: Sequence[Document], *, min_support: float = MIN_SUPPORT) -> Verification:
    """Check that an answer cites the excerpts it was given, and that each cited sentence is backed by its excerpts.

    A sentence without a citation is not checked, and a sentence that is too short to compare is accepted.

    Args:
        answer: The answer of the model, with citations like ``[1]`` or ``[1][2]``.
        documents: The excerpts the model was given, in the order they were numbered.
        min_support: The share of the content words of a cited sentence that must appear in its excerpts.

    Returns:
        Whether the answer holds. When it does, the excerpts it cites in order of their numbers.
    """
    numbers = _numbers(answer)
    if not numbers:
        return Verification(valid=False, problem=NO_CITATION_PROBLEM)
    missing = sorted({number for number in numbers if not 1 <= number <= len(documents)})
    if missing:
        return Verification(valid=False, problem=f"The answer cites excerpt {missing[0]}, which does not exist.")
    for sentence in _SENTENCE_BREAK.split(answer):
        cited = sorted(set(_numbers(sentence)))
        if cited and not _is_supported(sentence, [documents[number - 1] for number in cited], min_support):
            problem = f"A sentence is not supported by the excerpts it cites: {sentence.strip()}"
            return Verification(valid=False, problem=problem)
    citations = tuple(_citation(number, documents[number - 1]) for number in sorted(set(numbers)))
    return Verification(valid=True, citations=citations)


def _label(document: Document) -> str:
    title = str(document.metadata.get("title") or document.metadata.get("url") or "")
    heading = str(document.metadata.get("heading") or "")
    return f"{title} > {heading}" if heading and heading != title else title


def _citation(number: int, document: Document) -> Citation:
    return Citation(
        number=number,
        url=str(document.metadata.get("url") or ""),
        title=str(document.metadata.get("title") or ""),
        heading=str(document.metadata.get("heading") or ""),
        text=document.page_content,
    )


def _numbers(text: str) -> list[int]:
    return [int(part) for marker in _MARKER.findall(text) for part in marker.split(",")]


def _words(text: str) -> list[str]:
    words = _WORD.findall(text.casefold())
    return [word for word in words if len(word) >= MIN_WORD_LENGTH or any(character.isdigit() for character in word)]


def _is_supported(sentence: str, documents: Sequence[Document], min_support: float) -> bool:
    words = _words(_MARKER.sub(" ", sentence))
    if len(words) < MIN_WORDS:
        return True
    evidence_text = " ".join(f"{_label(document)} {document.page_content}" for document in documents)
    if _script(sentence) != _script(evidence_text):
        return True  # the words of two scripts cannot be compared
    evidence = {word[:STEM_LENGTH] for word in _words(evidence_text)}
    found = sum(word[:STEM_LENGTH] in evidence for word in words)
    return found / len(words) >= min_support


def _script(text: str) -> str:
    """The script most of the letters of a text are written in: ``arabic``, ``latin`` or ``other``."""
    counts = {"arabic": 0, "latin": 0, "other": 0}
    for character in text:
        if not character.isalpha():
            continue
        code = ord(character)
        if any(first <= code <= last for first, last in _ARABIC_RANGES):
            counts["arabic"] += 1
        elif character.isascii() or _LATIN_RANGE[0] <= code <= _LATIN_RANGE[1]:
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return max(counts, key=lambda script: counts[script])
