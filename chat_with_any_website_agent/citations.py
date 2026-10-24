"""Excerpts for the model, and a check that an answer is backed by the excerpts it cites.

The model sees the excerpts as a numbered list and cites them as ``[1]`` or ``[1][2]``. Some models write the brackets
in other styles (lenticular or full-width brackets, with or without a dagger and a label after the number), so
those are read as citations too and are rewritten as ``[1]``.

The check is plain code, with no model: every number must be an excerpt that exists, every sentence that cites must
share its content words with the excerpts it cites, and every number written with digits in such a sentence must be
written in those excerpts (or in the question). It catches an invented or misplaced citation, and a figure that the
model made up or worked out itself, such as a count of the lines of a list. It cannot judge meaning, and it cannot
compare text in two different scripts, so a sentence in another script than its excerpt is only checked for its
numbers.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document

# How many characters of an excerpt the model is shown, which keeps a request within the token limit of a key. A chunk
# has at most 2,000 characters (``ingest.DEFAULT_CHUNK_SIZE``), a little more when it keeps a heading with the text that
# follows, so the limit must not be lower: the end of a chunk would be cut off and a sentence there could not be read.
EXCERPT_CHARS = 2200

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

# The square brackets, the black lenticular brackets and the full-width square brackets, and the dagger that some
# models put between the number and a label.
_OPENING = re.escape("[" + chr(0x3010) + chr(0xFF3B))
_CLOSING = re.escape("]" + chr(0x3011) + chr(0xFF3D))
_DAGGER = chr(0x2020)
_MARKER = re.compile(rf"[{_OPENING}](\d+(?:\s*,\s*\d+)*)(?:{_DAGGER}[^{_CLOSING}]*)?[{_CLOSING}]")
# A sentence ends at a full stop, a question or exclamation mark or at a line break, but not before a citation, so
# that "... nothing. [1]" stays one sentence.
_SENTENCE_BREAK = re.compile(rf"(?<=[.!?{_ARABIC_QUESTION_MARK}])\s+(?!\[)|\n+")
_WORD = re.compile(r"\w+")
# What may separate the groups of thousands of a number: a comma ("1,000") or one of the spaces that typography uses
# (a model wrote "37 000" with a narrow no-break space): the no-break, thin and narrow no-break spaces.
_GROUP_SEPARATORS = ",\N{NO-BREAK SPACE}\N{THIN SPACE}\N{NARROW NO-BREAK SPACE}"
# A number written with digits, with groups of thousands and decimals ("3.11"). In an answer it is not part of a word
# ("x86", "2nd") and not the tail of a version ("3.11.4" is read as 3.11).
_NUMBER = re.compile(rf"(?<![\w.])\d+(?:[{_GROUP_SEPARATORS}]\d{{3}})*(?:\.\d+)?(?!\w)")
# A number written in an excerpt. The text of a PDF file often loses the space after a number ("dmodel = 1024on"), so a
# number there may touch a word, and only the digits around it are looked at.
_WRITTEN_NUMBER = re.compile(rf"(?<!\d)\d+(?:[{_GROUP_SEPARATORS}]\d{{3}})*(?:\.\d+)?")
# The number that starts a line of a numbered list ("1. First step").
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+")
# The digits 0 to 9 are often written as words in a text, so a small number is also accepted as its English word.
_SMALL_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


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
        answer: The answer with every citation written as ``[1]``, whatever style the model used.
    """

    valid: bool
    citations: tuple[Citation, ...] = ()
    problem: str | None = None
    answer: str = ""


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


def verify_answer(
    answer: str, documents: Sequence[Document], *, min_support: float = MIN_SUPPORT, question: str = ""
) -> Verification:
    """Check that an answer cites the excerpts it was given, and that each cited sentence is backed by its excerpts.

    A sentence without a citation is not checked, and a sentence that is too short to compare is accepted. A number
    that a cited sentence writes with digits must be written in the excerpts it cites, or in the question: the answer
    may not state a figure that the source does not state, such as a count that the model made by itself.

    Args:
        answer: The answer of the model, with citations like ``[1]`` or ``[1][2]``.
        documents: The excerpts the model was given, in the order they were numbered.
        min_support: The share of the content words of a cited sentence that must appear in its excerpts.
        question: The question, whose own numbers an answer may repeat.

    Returns:
        Whether the answer holds, and the answer with its citations written as ``[1]``. When it holds, also the
        excerpts it cites in order of their numbers.
    """
    answer = normalize_citations(answer)
    numbers = _numbers(answer)
    if not numbers:
        return Verification(valid=False, problem=NO_CITATION_PROBLEM, answer=answer)
    missing = sorted({number for number in numbers if not 1 <= number <= len(documents)})
    if missing:
        problem = f"The answer cites excerpt {missing[0]}, which does not exist."
        return Verification(valid=False, problem=problem, answer=answer)
    for sentence in _SENTENCE_BREAK.split(answer):
        cited = sorted(set(_numbers(sentence)))
        if not cited:
            continue
        evidence = [documents[number - 1] for number in cited]
        if not _is_supported(sentence, evidence, min_support):
            problem = f"A sentence is not supported by the excerpts it cites: {sentence.strip()}"
            return Verification(valid=False, problem=problem, answer=answer)
        invented = _unwritten_number(sentence, _evidence_text(evidence), question)
        if invented is not None:
            problem = f"The answer states {invented}, which the excerpts it cites do not contain."
            return Verification(valid=False, problem=problem, answer=answer)
    citations = tuple(_citation(number, documents[number - 1]) for number in sorted(set(numbers)))
    return Verification(valid=True, citations=citations, answer=answer)


def normalize_citations(answer: str) -> str:
    """Write every citation of an answer as ``[1]``, whatever brackets the model used.

    Args:
        answer: The answer of the model.

    Returns:
        The answer with citations in lenticular or full-width brackets, or with a label after the number,
        rewritten as plain ``[1]``. Plain ``[1]`` and ``[1, 2]`` are kept as they are.
    """
    return _MARKER.sub(lambda match: f"[{match.group(1)}]", answer)


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


def _evidence_text(documents: Sequence[Document]) -> str:
    return " ".join(f"{_label(document)} {document.page_content}" for document in documents)


def _canonical(number: str) -> str:
    """A number written with digits of any script as plain digits without the separators of groups of thousands."""
    digits = "".join(str(unicodedata.digit(character)) if character.isdigit() else character for character in number)
    return re.sub(f"[{_GROUP_SEPARATORS}]", "", digits)


def _unwritten_number(sentence: str, evidence_text: str, question: str) -> str | None:
    """The first number of a sentence that neither its excerpts nor the question write, as the sentence writes it."""
    body = _LIST_MARKER.sub("", _MARKER.sub(" ", sentence))
    found = [match.group() for match in _NUMBER.finditer(body)]
    if not found:
        return None
    known = {_canonical(match.group()) for match in _WRITTEN_NUMBER.finditer(f"{evidence_text} {question}")}
    lowered = evidence_text.casefold()
    for shown in found:
        value = _canonical(shown)
        if value in known:
            continue
        if len(value) == 1 and re.search(rf"\b{_SMALL_NUMBER_WORDS[int(value)]}\b", lowered):
            continue
        return shown
    return None


def _is_supported(sentence: str, documents: Sequence[Document], min_support: float) -> bool:
    words = _words(_MARKER.sub(" ", sentence))
    if len(words) < MIN_WORDS:
        return True
    evidence_text = _evidence_text(documents)
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
