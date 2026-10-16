"""Tests of the text cleaning, including the removal of "edit" links in many languages."""

import time

import pytest

from chat_with_any_website_agent.cleaning import clean_text, collapse_repeated_lines

NBSP = chr(0xA0)


def test_collapses_whitespace_and_blank_lines() -> None:
    assert clean_text("a \t b" + NBSP * 2 + "c\n\n\n\n\nd") == "a b c\n\nd"


def test_normalizes_line_endings() -> None:
    assert clean_text("one\r\n\r\ntwo\rthree") == "one\n\ntwo\nthree"


def test_removes_zero_width_and_control_characters() -> None:
    assert clean_text("he" + chr(0x200B) + "llo\x00 wor" + chr(0xFEFF) + "ld") == "hello world"


def test_removes_footnote_markers_and_inline_tags() -> None:
    assert clean_text("Fact.<sup>\\[1\\]</sup> More<br>text <span class='x'>here</span>.") == "Fact. Moretext here."


def test_removes_edit_links_from_headings() -> None:
    assert clean_text("## History\n\\[edit\\]\n\nBody.") == "## History\n\nBody."


def test_drops_repeated_paragraphs_case_insensitively() -> None:
    text = "Accept cookies\n\nReal content.\n\naccept COOKIES\n\nMore content."
    assert clean_text(text) == "Accept cookies\n\nReal content.\n\nMore content."


def test_empty_input_stays_empty() -> None:
    assert clean_text("  \n\n  ") == ""


# --- edit links ---


@pytest.mark.parametrize(
    ("heading", "edit_link"),
    [
        ("History", "\\[edit\\]"),
        ("History", "[edit source]"),
        ("Geschichte", "[Bearbeiten | Quelltext bearbeiten]"),
        ("Histoire", "[modifier | modifier le code]"),
        ("Historia", "[editar]"),
        ("Storia", "[modifica | modifica wikitesto]"),
        ("История", "[править | править код]"),
        ("التاريخ", "[تعديل مصدري]"),
        ("التاريخ", "\\[تعديل\\]"),
        ("历史", "[编辑]"),
        ("歴史", "[編集]"),
        ("역사", "[편집]"),
        ("Sejarah", "[sunting | sunting sumber]"),
        ("Geschiedenis", "[bewerken | brontekst bewerken]"),
        ("Tarih", "[değiştir | kaynağı değiştir]"),
        ("Historia", "[redigera | redigera wikitext]"),
    ],
)
def test_removes_edit_links_of_wikipedia_in_many_languages(heading: str, edit_link: str) -> None:
    after_heading_text = clean_text(f"## {heading} {edit_link}\n\nBody text.")
    own_line_text = clean_text(f"## {heading}\n{edit_link}\n\nBody text.")
    assert after_heading_text == f"## {heading}\n\nBody text."
    assert own_line_text == f"## {heading}\n\nBody text."


@pytest.mark.parametrize("line", ["تعديل مصدري - تعديل", "Edit", "modifier | modifier le code", "编辑 · 编辑源代码"])
def test_removes_lines_that_only_hold_edit_links(line: str) -> None:
    assert clean_text(f"First line\n{line}\nSecond line") == "First line\nSecond line"


def test_removes_table_rows_that_only_hold_edit_links() -> None:
    table = "| Name | Born |\n|---|---|\n| Ibn Khaldun | 1332 |\n| تعديل مصدري - تعديل | |\n\nText"
    assert clean_text(table) == "| Name | Born |\n|---|---|\n| Ibn Khaldun | 1332 |\n\nText"
    assert clean_text("| Edit | |\nNext") == "Next"


def test_keeps_table_rows_with_other_content_next_to_an_edit_word() -> None:
    assert clean_text("| Edit | Delete |") == "| Edit | Delete |"


@pytest.mark.parametrize(
    "text",
    [
        "You can edit the file in any editor.",
        "Edit settings",
        "Press [Enter] to continue and [Esc] to leave.",
        "Use list[0] or items[edit_mode].",
        "تعديل الصور في البرنامج سهل",
    ],
)
def test_keeps_ordinary_text_that_mentions_editing(text: str) -> None:
    assert clean_text(text) == text


# --- repeated lines ---


def test_collapses_a_line_that_repeats_the_line_before() -> None:
    assert clean_text("ابن خلدون\nابن خلدون\nالإمام") == "ابن خلدون\nالإمام"
    assert clean_text("Title\nTITLE\nText") == "Title\nText"


def test_keeps_the_heading_when_a_title_is_repeated_as_plain_text() -> None:
    assert clean_text("# Title\nTitle\nText") == "# Title\nText"
    assert clean_text("Title\n# Title\nText") == "# Title\nText"
    assert clean_text("## **Title**\nTitle\nText") == "## **Title**\nText"


def test_keeps_lines_that_repeat_without_being_next_to_each_other() -> None:
    assert clean_text("ابن خلدون\nالإمام\nابن خلدون") == "ابن خلدون\nالإمام\nابن خلدون"


@pytest.mark.parametrize("text", ["100\n100", "***\n***", "ok\nok", "...\n..."])
def test_keeps_repeats_that_say_nothing_or_are_too_short_to_judge(text: str) -> None:
    assert clean_text(text) == text


def test_collapse_repeated_lines_works_on_plain_lists() -> None:
    assert collapse_repeated_lines(["a long line", "a long line", "", "a long line"]) == [
        "a long line",
        "",
        "a long line",
    ]


def test_cleaning_a_large_page_is_fast() -> None:
    page = "\n".join(f"Paragraph {i} has words in it [edit] and more text to read." for i in range(20_000))
    started = time.perf_counter()
    cleaned = clean_text(page)
    assert time.perf_counter() - started < 3
    assert "[edit]" not in cleaned
