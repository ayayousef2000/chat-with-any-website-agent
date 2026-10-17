"""Tests of the excerpts shown to the model and of the check of its citations."""

from langchain_core.documents import Document

from chat_with_any_website_agent.citations import (
    ELLIPSIS,
    NO_CITATION_PROBLEM,
    Citation,
    format_excerpts,
    verify_answer,
)

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
