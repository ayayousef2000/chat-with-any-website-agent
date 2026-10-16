"""Normalizing extracted text before it is split into chunks."""

import re
import unicodedata

_FOOTNOTES = re.compile(r"<sup>.*?</sup>", re.DOTALL)

# The words that Wikipedia and similar sites use for their "edit" links, in many languages. The links show up as
# "[edit]", "[Bearbeiten | Quelltext bearbeiten]" or as a line such as "تعديل مصدري - تعديل", and are noise in the text.
_EDIT_LABELS = (
    "edit",
    "edit source",
    "edit section",  # English
    "تعديل",
    "تعديل مصدري",
    "تعديل الشفرة",
    "عدل",
    "عدل المصدر",
    "تحرير",  # Arabic
    "modifier",
    "modifier le code",
    "modifier le wikicode",  # French
    "bearbeiten",
    "quelltext bearbeiten",  # German
    "editar",
    "editar código",
    "editar fuente",
    "editar código-fonte",
    "editar código fonte",  # Spanish, Portuguese
    "modifica",
    "modifica wikitesto",  # Italian
    "править",
    "править код",
    "редактировать",  # Russian
    "ред.",
    "ред. код",
    "редагувати",  # Ukrainian
    "edytuj",
    "edytuj kod",  # Polish
    "bewerken",
    "brontekst bewerken",  # Dutch
    "değiştir",
    "kaynağı değiştir",  # Turkish
    "ویرایش",
    "ویرایش ویکی‌متن",  # Persian
    "עריכה",
    "עריכת קוד מקור",  # Hebrew
    "编辑",
    "编辑源代码",
    "編輯",
    "編輯原始碼",  # Chinese
    "編集",
    "ソースを編集",  # Japanese
    "편집",
    "원본 편집",  # Korean
    "sunting",
    "sunting sumber",  # Indonesian
    "redigera",
    "redigera wikitext",  # Swedish
    "editovat",
    "editovat zdroj",  # Czech
    "sửa",
    "sửa mã nguồn",  # Vietnamese
    "संपादित करें",
    "स्रोत संपादित करें",  # Hindi
)
_LABEL = "(?:" + "|".join(re.escape(label) for label in sorted(_EDIT_LABELS, key=len, reverse=True)) + ")"
_LABELS = rf"{_LABEL}(?:[ \t]*[|·•–—-][ \t]*{_LABEL})*"
_EDIT_LINKS = re.compile(rf"\n?[ \t]*\\?\[[ \t]*{_LABELS}[ \t]*\\?\]", re.IGNORECASE)
_EDIT_ONLY_LINES = re.compile(rf"^[ \t]*{_LABELS}[ \t]*$\n?", re.IGNORECASE | re.MULTILINE)
# The same links can sit alone in a row of a table, such as the foot of an info box: "| تعديل مصدري - تعديل | |".
_EDIT_ONLY_ROWS = re.compile(rf"^[ \t]*\|[ \t]*{_LABELS}[ \t]*(?:\|[ \t]*)*$\n?", re.IGNORECASE | re.MULTILINE)
_HEADING_MARKS = re.compile(r"^#{1,6}[ \t]+")
_INLINE_TAGS = re.compile(r"</?(?:sub|sup|span|br)\b[^>]*>")
# Control characters, and the zero-width characters U+200B to U+200F and U+FEFF.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏﻿]")
_INLINE_WHITESPACE = re.compile(r"[ \t]+")  # NFKC normalization already maps non-breaking spaces to spaces
_EXTRA_NEWLINES = re.compile(r"\n{3,}")


def _line_key(line: str) -> str:
    """Reduce a line to what it says, ignoring heading marks, emphasis and capitalization."""
    return _HEADING_MARKS.sub("", line).strip("*_` \t").casefold()


def collapse_repeated_lines(lines: list[str]) -> list[str]:
    """Drop a line that repeats the line right before it, as when a page shows its title twice in a row.

    Only lines that say something are compared (at least three characters, with a letter), so repeated
    separators and numbers are kept, and a blank line between two lines means they are not repeats. If the
    second of two repeats is a heading, it replaces the first so the structure is kept.

    Args:
        lines: The lines of a text.

    Returns:
        The lines without immediate repeats.
    """
    result: list[str] = []
    previous = ""
    for line in lines:
        key = _line_key(line)
        meaningful = len(key) >= 3 and any(character.isalpha() for character in key)
        if meaningful and key == previous:
            if _HEADING_MARKS.match(line) and not _HEADING_MARKS.match(result[-1]):
                result[-1] = line
            continue
        result.append(line)
        previous = key if meaningful else ""
    return result


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace and remove page noise.

    Removes the "edit" links of Wikipedia-style sites in many languages, a line that repeats the line before it,
    and repeated paragraphs (menus, cookie banners, etc.).

    Args:
        text: The text extracted from a page.

    Returns:
        The cleaned text, with paragraphs separated by one blank line.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _FOOTNOTES.sub("", text)
    text = _EDIT_LINKS.sub("", text)
    text = _EDIT_ONLY_LINES.sub("", text)
    text = _EDIT_ONLY_ROWS.sub("", text)
    text = _INLINE_TAGS.sub("", text)

    lines = collapse_repeated_lines([_INLINE_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")])
    text = _EXTRA_NEWLINES.sub("\n\n", "\n".join(lines))

    seen: set[str] = set()
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        key = paragraph.casefold()
        if not paragraph or key in seen:
            continue
        seen.add(key)
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)
