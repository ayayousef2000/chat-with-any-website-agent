"""Tests of the excerpts shown to the model and of the check of its citations."""

import pytest
from langchain_core.documents import Document

from chat_with_any_website_agent.citations import (
    ELLIPSIS,
    EXCERPT_CHARS,
    NO_CITATION_PROBLEM,
    Citation,
    format_excerpts,
    normalize_citations,
    verify_answer,
)
from chat_with_any_website_agent.ingest import DEFAULT_CHUNK_SIZE

SENTENCE = " Their file names all end with `.sample`."
PASS_TEXT = (
    "The pass statement does nothing. "
    "It is used when a statement is required syntactically but the program requires no action."
)
PASS = Document(
    page_content=PASS_TEXT,
    metadata={"url": "https://docs.example/control", "title": "Control Flow", "heading": "pass Statements"},
)
SET = Document(
    page_content="A set is an unordered collection with no duplicate elements.",
    metadata={"url": "https://docs.example/data", "title": "Data Structures", "heading": "Sets"},
)
ARABIC = Document(
    page_content="صمم لغة بايثون خيدو فان روسم في معهد الرياضيات",
    metadata={"url": "https://ar.example/python", "title": "بايثون", "heading": ""},
)


# --- the excerpts shown to the model ---


def test_the_excerpts_are_numbered_and_labelled_with_their_title_and_section() -> None:
    text = format_excerpts([PASS, SET])

    assert text == f"[1] Control Flow > pass Statements\n{PASS_TEXT}\n\n[2] Data Structures > Sets\n{SET.page_content}"


def test_a_section_with_the_title_of_the_page_is_not_repeated() -> None:
    metadata = {"url": "https://x.example", "title": "Guide", "heading": "Guide"}
    document = Document(page_content="text", metadata=metadata)

    assert format_excerpts([document]) == "[1] Guide\ntext"


def test_an_excerpt_without_a_title_is_labelled_with_its_address() -> None:
    document = Document(page_content="text", metadata={"url": "https://x.example"})

    assert format_excerpts([document]) == "[1] https://x.example\ntext"


def test_a_long_excerpt_is_cut_and_ends_with_an_ellipsis() -> None:
    document = Document(page_content="a" * 20, metadata={"title": "t"})

    assert format_excerpts([document], max_chars=10) == f"[1] t\n{'a' * 10}{ELLIPSIS}"


def test_a_whole_chunk_is_shown_to_the_model() -> None:
    # The last sentence of a chunk of 2,000 characters was once cut off, and the model never read it.
    text = "x" * (DEFAULT_CHUNK_SIZE - len(SENTENCE)) + SENTENCE
    document = Document(page_content=text, metadata={"title": "t"})

    shown = format_excerpts([document])

    assert shown.endswith(SENTENCE)
    assert ELLIPSIS not in shown
    assert EXCERPT_CHARS >= DEFAULT_CHUNK_SIZE


def test_no_excerpts_make_an_empty_text() -> None:
    assert format_excerpts([]) == ""


# --- an answer that holds ---


def test_an_answer_that_cites_an_excerpt_it_is_backed_by_is_valid() -> None:
    result = verify_answer("The pass statement does nothing [1].", [PASS, SET])

    assert result.valid
    assert result.problem is None
    assert result.citations == (
        Citation(
            number=1,
            url="https://docs.example/control",
            title="Control Flow",
            heading="pass Statements",
            text=PASS_TEXT,
        ),
    )


def test_a_citation_after_the_full_stop_belongs_to_the_sentence_before_it() -> None:
    assert verify_answer("A set is an unordered collection without duplicates. [2]", [PASS, SET]).valid


def test_a_sentence_may_cite_several_excerpts_in_either_notation() -> None:
    sentence = "The pass statement does nothing and a set is an unordered collection"

    assert verify_answer(f"{sentence} [1][2].", [PASS, SET]).valid
    assert verify_answer(f"{sentence} [1, 2].", [PASS, SET]).valid


def test_the_cited_excerpts_are_listed_once_each_in_the_order_of_their_numbers() -> None:
    answer = (
        "A set is an unordered collection [2]. The pass statement does nothing [1]. "
        "Again, the pass statement does nothing [1]."
    )

    result = verify_answer(answer, [PASS, SET])

    assert [citation.number for citation in result.citations] == [1, 2]


def test_a_sentence_without_a_citation_is_not_checked() -> None:
    answer = "Python is great and very popular worldwide. The pass statement does nothing [1]."

    assert verify_answer(answer, [PASS, SET]).valid


def test_a_sentence_too_short_to_compare_is_accepted() -> None:
    assert verify_answer("It does nothing [1].", [SET]).valid


# --- an answer that does not hold ---


def test_an_answer_that_cites_nothing_is_not_valid() -> None:
    result = verify_answer("The pass statement does nothing.", [PASS])

    assert not result.valid
    assert result.problem == NO_CITATION_PROBLEM
    assert result.citations == ()


def test_a_citation_of_an_excerpt_that_does_not_exist_is_not_valid() -> None:
    result = verify_answer("The pass statement does nothing [3].", [PASS, SET])

    assert not result.valid
    assert result.problem == "The answer cites excerpt 3, which does not exist."


def test_the_number_zero_is_not_an_excerpt() -> None:
    assert not verify_answer("The pass statement does nothing [0].", [PASS]).valid


def test_a_sentence_that_the_cited_excerpt_does_not_back_is_not_valid() -> None:
    result = verify_answer("Python was released in 1991 by a Dutch programmer [1].", [PASS, SET])

    assert not result.valid
    assert result.problem is not None
    assert result.problem.startswith("A sentence is not supported by the excerpts it cites: Python was released")
    assert result.citations == ()


def test_one_unsupported_sentence_makes_the_whole_answer_invalid() -> None:
    answer = "The pass statement does nothing [1]. Python was released in 1991 by a Dutch programmer [1]."

    assert not verify_answer(answer, [PASS]).valid


def test_the_share_of_words_that_must_be_backed_can_be_changed() -> None:
    answer = "The pass statement does nothing special [1]."  # four of its five words are in the excerpt

    assert verify_answer(answer, [PASS]).valid
    assert not verify_answer(answer, [PASS], min_support=0.9).valid


# --- numbers ---

TREE = Document(
    page_content=(
        "## Exception hierarchy\nThe class hierarchy for built-in exceptions is:\n"
        "BaseException\n ├── BaseExceptionGroup\n ├── GeneratorExit\n ├── KeyboardInterrupt\n"
        " ├── SystemExit\n └── Exception\n      ├── ArithmeticError\n      └── LookupError"
    ),
    metadata={
        "url": "https://docs.example/exceptions",
        "title": "Built-in Exceptions",
        "heading": "Exception hierarchy",
    },
)
RELEASE = Document(
    page_content="Python 3.11 was released on 2022-10-24. It added exception groups. The download is 25,000 KB large.",
    metadata={"url": "https://docs.example/news", "title": "What's new", "heading": ""},
)


def test_a_count_that_the_excerpt_does_not_state_is_not_valid() -> None:
    # The model counted the lines of the tree. The words of the sentence match the excerpt, but the figure is not there.
    answer = "Python's built-in exception hierarchy defines 68 distinct exception classes [1]."

    result = verify_answer(answer, [TREE])

    assert not result.valid
    assert result.problem == "The answer states 68, which the excerpts it cites do not contain."
    assert result.citations == ()


def test_a_number_that_the_cited_excerpt_writes_is_valid() -> None:
    assert verify_answer("Python 3.11 added exception groups [1].", [RELEASE]).valid


def test_the_first_number_that_is_not_written_is_the_one_named() -> None:
    result = verify_answer("Python 3.11 was released in 2023 with 40 features [1].", [RELEASE])

    assert result.problem == "The answer states 2023, which the excerpts it cites do not contain."


def test_a_number_may_be_written_with_or_without_groups_of_thousands() -> None:
    assert verify_answer("The download is 25000 KB large [1].", [RELEASE]).valid
    assert verify_answer("The download is 25,000 KB large [1].", [RELEASE]).valid


@pytest.mark.parametrize("separator", ["\N{NO-BREAK SPACE}", "\N{THIN SPACE}", "\N{NARROW NO-BREAK SPACE}"])
def test_a_number_may_be_grouped_with_the_spaces_of_typography(separator: str) -> None:
    # A model wrote "37 000 tokens" with a narrow no-break space for an excerpt that says "37000 tokens".
    plain = Document(
        page_content="The text has a shared vocabulary of about 37000 tokens.", metadata={"url": "u", "title": "T"}
    )
    grouped = Document(
        page_content=f"The text has a shared vocabulary of about 37{separator}000 tokens.",
        metadata={"url": "u", "title": "T"},
    )

    assert verify_answer(f"The text has a shared vocabulary of about 37{separator}000 tokens [1].", [plain]).valid
    assert verify_answer("The text has a shared vocabulary of about 37000 tokens [1].", [grouped]).valid
    assert not verify_answer(f"The text has a shared vocabulary of about 38{separator}000 tokens [1].", [plain]).valid


def test_a_plain_space_does_not_group_digits() -> None:
    excerpt = Document(page_content="The text has about 37000 tokens in 2014.", metadata={"url": "u", "title": "T"})

    result = verify_answer("The text has about 37 000 tokens in 2014 [1].", [excerpt])

    assert result.problem == "The answer states 37, which the excerpts it cites do not contain."


def test_a_number_that_the_text_of_a_pdf_file_writes_against_a_word_is_written() -> None:
    # The text of the paper has "dmodel = 1024on the Wall Street Journal": the space after the number was lost.
    excerpt = Document(
        page_content="We trained a 4-layer transformer with dmodel = 1024on the Wall Street Journal portion.",
        metadata={"url": "u", "title": "T", "heading": "Page 9"},
    )
    answer = "A 4-layer transformer with dmodel = {} was trained on the Wall Street Journal portion [1]."

    assert verify_answer(answer.format("1024"), [excerpt]).valid
    for other in ("102", "10", "4096"):  # only the whole run of digits is written
        assert verify_answer(answer.format(other), [excerpt]).problem == (
            f"The answer states {other}, which the excerpts it cites do not contain."
        )


def test_digits_of_another_script_are_the_same_number_as_western_digits() -> None:
    excerpt = Document(
        page_content="ظهرت النسخة الأولى سنة ١٩٨٩ في هولندا", metadata={"url": "u", "title": "بايثون", "heading": ""}
    )

    assert verify_answer("ظهرت النسخة الأولى سنة 1989 [1].", [excerpt]).valid
    assert not verify_answer("ظهرت النسخة الأولى سنة 1991 [1].", [excerpt]).valid


def test_a_small_number_is_accepted_when_the_excerpt_writes_it_as_a_word() -> None:
    excerpt = Document(
        page_content="The function takes two arguments and returns the sum.", metadata={"url": "u", "title": "T"}
    )

    assert verify_answer("The function takes 2 arguments [1].", [excerpt]).valid
    assert not verify_answer("The function takes 3 arguments [1].", [excerpt]).valid
    for larger in ("12", "25", "34", "89", "10"):  # only the digits 0 to 9 are read as words
        result = verify_answer(f"The function takes {larger} arguments [1].", [excerpt])
        assert result.problem == f"The answer states {larger}, which the excerpts it cites do not contain."


def test_a_number_that_the_question_asks_about_may_be_repeated_in_the_answer() -> None:
    answer = "In Python 3.12 the pass statement does nothing [1]."

    assert not verify_answer(answer, [PASS]).valid
    assert verify_answer(answer, [PASS], question="What does pass do in Python 3.12?").valid


def test_the_number_that_starts_a_line_of_a_list_is_not_a_figure_of_the_answer() -> None:
    answer = "1. The pass statement does nothing [1].\n2. The pass statement is a placeholder in code [1]."

    assert verify_answer(answer, [PASS]).valid


def test_ordinals_and_the_digits_in_names_are_not_numbers_of_the_answer() -> None:
    assert verify_answer("The pass statement does nothing in the 2nd chapter on x86 machines [1].", [PASS]).valid


def test_a_version_is_read_by_its_first_two_parts_and_its_tail_is_not_a_number_of_its_own() -> None:
    assert verify_answer("Python 3.11.4 added exception groups [1].", [RELEASE]).valid

    result = verify_answer("Python 3.12.1 added exception groups [1].", [RELEASE])

    assert result.problem == "The answer states 3.12, which the excerpts it cites do not contain."


def test_a_number_in_a_sentence_without_a_citation_is_not_checked() -> None:
    answer = "About 90 percent of people like it. The pass statement does nothing [1]."

    assert verify_answer(answer, [PASS]).valid


def test_a_number_is_checked_against_the_excerpts_that_its_own_sentence_cites() -> None:
    answer = "A set has no duplicate elements and 25000 KB is the size [2]."

    assert not verify_answer(
        answer, [RELEASE, SET]
    ).valid  # the 25,000 is in excerpt 1, which this sentence does not cite


def test_a_number_in_the_title_of_the_cited_page_counts_as_written() -> None:
    excerpt = Document(
        page_content="The statement does nothing at all.", metadata={"url": "u", "title": "Python 3.14 tutorial"}
    )

    assert verify_answer("In Python 3.14 the statement does nothing at all [1].", [excerpt]).valid


def test_an_answer_without_numbers_is_not_affected() -> None:
    assert verify_answer("The pass statement does nothing [1].", [PASS]).valid


# --- other languages ---


def test_an_arabic_sentence_is_checked_against_an_arabic_excerpt() -> None:
    assert verify_answer("صمم لغة بايثون خيدو فان روسم [1].", [ARABIC]).valid
    assert not verify_answer("ظهرت النسخة الأولى سنة 1989 [1].", [ARABIC]).valid


def test_a_sentence_in_another_script_than_its_excerpt_is_only_checked_for_its_numbers() -> None:
    assert verify_answer("عبارة pass لا تفعل شيئا [1].", [PASS]).valid
    assert not verify_answer("عبارة pass لا تفعل شيئا [2].", [PASS]).valid


def test_the_arabic_question_mark_ends_a_sentence() -> None:
    answer = "هل بايثون لغة برمجة؟ ظهرت النسخة الأولى سنة 1989 [1]."

    result = verify_answer(answer, [ARABIC])

    assert result.problem == "A sentence is not supported by the excerpts it cites: ظهرت النسخة الأولى سنة 1989 [1]."


# --- other styles of brackets ---

LENTICULAR = (chr(0x3010), chr(0x3011))  # the black lenticular brackets that some models cite with
FULL_WIDTH = (chr(0xFF3B), chr(0xFF3D))
DAGGER = chr(0x2020)


def _cite(number: str, brackets: tuple[str, str] = LENTICULAR) -> str:
    return f"{brackets[0]}{number}{brackets[1]}"


def test_a_citation_in_lenticular_brackets_counts_and_is_written_with_plain_brackets() -> None:
    result = verify_answer(f"The pass statement does nothing {_cite('1')}.", [PASS, SET])

    assert result.valid
    assert result.answer == "The pass statement does nothing [1]."
    assert [citation.number for citation in result.citations] == [1]


def test_a_label_after_the_number_is_dropped() -> None:
    for label in ("source", "L5-L9"):
        result = verify_answer(f"The pass statement does nothing {_cite('1' + DAGGER + label)}.", [PASS])

        assert result.valid
        assert result.answer == "The pass statement does nothing [1]."


def test_full_width_brackets_and_lists_of_numbers_are_read() -> None:
    answer = f"The pass statement does nothing and a set is an unordered collection {_cite('1, 2', FULL_WIDTH)}."

    result = verify_answer(answer, [PASS, SET])

    assert result.valid
    assert result.answer.endswith("collection [1, 2].")
    assert [citation.number for citation in result.citations] == [1, 2]


def test_a_citation_after_the_full_stop_in_other_brackets_still_belongs_to_the_sentence_before_it() -> None:
    result = verify_answer(f"A set is an unordered collection without duplicates. {_cite('2')}", [PASS, SET])

    assert result.valid
    assert result.answer == "A set is an unordered collection without duplicates. [2]"


def test_the_other_styles_are_checked_like_plain_brackets() -> None:
    assert not verify_answer(f"The pass statement does nothing {_cite('3')}.", [PASS, SET]).valid
    unsupported = verify_answer(f"Python was released in 1991 by a Dutch programmer {_cite('1')}.", [PASS])
    assert not unsupported.valid
    assert unsupported.answer == "Python was released in 1991 by a Dutch programmer [1]."


def test_plain_citations_are_left_as_they_are() -> None:
    text = "One [1]. Two [1, 2][3]. No citation here."

    assert normalize_citations(text) == text
    assert normalize_citations("") == ""
