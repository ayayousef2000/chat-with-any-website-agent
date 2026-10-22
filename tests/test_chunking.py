from itertools import pairwise

import pytest

from chat_with_any_website_agent.chunking import chunk_pages, chunk_text, contextualize


def test_short_text_is_a_single_chunk() -> None:
    chunks = chunk_text("Hello world.", 100, 10)
    assert [c.text for c in chunks] == ["Hello world."]
    assert chunks[0].heading == ""


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("", 100, 10) == []


def test_chunks_respect_the_size_limit() -> None:
    text = "\n\n".join(f"Paragraph number {i} has some words in it." for i in range(60))
    chunks = chunk_text(text, 200, 40)
    assert len(chunks) > 5
    assert all(len(c.text) <= 200 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    paragraphs = [f"Paragraph {i:02d} " + "x" * 30 for i in range(30)]
    chunks = chunk_text("\n\n".join(paragraphs), 200, 80)
    assert len(chunks) > 2
    for previous, current in pairwise(chunks):
        last_paragraph = previous.text.split("\n")[-1]
        assert last_paragraph in current.text


def test_no_text_is_lost() -> None:
    paragraphs = [f"Unique paragraph {i} " + "word " * 20 for i in range(25)]
    chunks = chunk_text("\n\n".join(paragraphs), 300, 50)
    joined = "\n".join(c.text for c in chunks)
    assert all(p.strip() in joined for p in paragraphs)


def test_an_oversized_paragraph_is_split_by_sentence() -> None:
    sentences = [f"This is sentence number {i}." for i in range(40)]
    chunks = chunk_text(" ".join(sentences), 150, 0)
    assert len(chunks) > 1
    assert all(len(c.text) <= 150 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_an_oversized_sentence_is_split_by_word() -> None:
    chunks = chunk_text("word " * 200, 100, 0)
    assert all(len(c.text) <= 100 for c in chunks)
    assert sum(len(c.text.split()) for c in chunks) == 200


def test_chunks_record_their_section_heading() -> None:
    text = (
        "# Title\n\nIntro text.\n\n## Setup\n\n" + "Setup details. " * 40 + "\n\n## Usage\n\n" + "Usage details. " * 40
    )
    chunks = chunk_text(text, 200, 0)
    assert chunks[0].heading == "Title"
    assert chunks[-1].heading == "Usage"
    assert "Setup" in {c.heading for c in chunks}


def test_a_heading_is_not_left_at_the_end_of_a_chunk() -> None:
    body = "Body sentence. " * 12
    text = f"{body}\n\n## Next section\n\n{body}"
    chunks = chunk_text(text, len(body) + 20, 0)
    assert not any(c.text.rstrip().splitlines()[-1].startswith("#") for c in chunks)


def test_a_lone_heading_stays_with_the_following_text() -> None:
    chunks = chunk_text("# T\n\n" + "word " * 1000, 2000, 300)
    assert chunks[0].text.startswith("# T\n")
    assert len(chunks[0].text) > 100


@pytest.mark.parametrize(
    ("title", "heading", "expected"),
    [
        ("Page", "Section", "Page > Section\n\nbody"),
        ("Page", "", "Page\n\nbody"),
        ("Page", "Page", "Page\n\nbody"),
    ],
)
def test_contextualize(title: str, heading: str, expected: str) -> None:
    assert contextualize(title, heading, "body") == expected


# --- the text of a PDF file, a page at a time ---

PAGES = "## Page 1\n\nFirst page text.\n\n## Page 2\n\nSecond page text.\n\n## Page 3\n\nThird page text."


def test_a_chunk_of_a_pdf_file_never_crosses_a_page() -> None:
    chunks = chunk_pages(PAGES, 2000, 300)

    assert [(chunk.heading, "## Page" in chunk.text.split("\n", 1)[-1]) for chunk in chunks] == [
        ("Page 1", False),
        ("Page 2", False),
        ("Page 3", False),
    ]
    assert [chunk.text.split("\n")[-1] for chunk in chunks] == [
        "First page text.",
        "Second page text.",
        "Third page text.",
    ]


def test_the_same_text_chunked_as_one_text_merges_short_pages_under_the_heading_of_the_first() -> None:
    # This is why a PDF is chunked a page at a time: here the text of pages 2 and 3 is labelled "Page 1".
    chunks = chunk_text(PAGES, 2000, 300)

    assert [chunk.heading for chunk in chunks] == ["Page 1"]
    assert "Third page text." in chunks[0].text


def test_the_overlap_of_chunks_stays_inside_a_page() -> None:
    page_one = "\n\n".join(f"Sentence {n} of the first page is here." for n in range(40))
    text = f"## Page 1\n\n{page_one}\n\n## Page 2\n\nThe second page starts here."

    chunks = chunk_pages(text, 300, 80)

    page_two = [chunk for chunk in chunks if chunk.heading == "Page 2"]
    assert len(page_two) == 1
    assert page_two[0].text == "## Page 2\nThe second page starts here."  # nothing of page 1 is repeated in it
    assert all("Page 2" not in chunk.text for chunk in chunks if chunk.heading == "Page 1")


def test_every_chunk_of_a_long_page_has_the_heading_of_that_page() -> None:
    body = "\n\n".join(f"Sentence {n} of the long page says something useful." for n in range(60))
    chunks = chunk_pages(f"## Page 1\n\nShort.\n\n## Page 2\n\n{body}\n\n## Page 3\n\nEnd.", 300, 60)

    page_two = [chunk for chunk in chunks if chunk.heading == "Page 2"]
    assert len(page_two) > 3
    assert next(chunk.heading for chunk in chunks) == "Page 1"
    assert [chunk.heading for chunk in chunks][-1] == "Page 3"
    assert all(len(chunk.text) <= 300 for chunk in chunks)


def test_the_note_at_the_end_is_a_section_of_its_own() -> None:
    text = f"{PAGES}\n\n## Note\n\nOnly the first 3 of the 5 pages of this document were read."

    chunks = chunk_pages(text, 2000, 300)

    assert [chunk.heading for chunk in chunks] == ["Page 1", "Page 2", "Page 3", "Note"]
    assert "Third page text." not in chunks[-1].text


def test_text_before_the_first_page_heading_is_kept() -> None:
    chunks = chunk_pages("A line before any page.\n\n## Page 1\n\nFirst page text.", 2000, 300)

    assert [chunk.heading for chunk in chunks] == ["", "Page 1"]


def test_a_pdf_text_without_text_has_no_chunks() -> None:
    assert chunk_pages("", 2000, 300) == []
    assert chunk_pages("\n\n  \n", 2000, 300) == []


def test_a_heading_that_only_looks_like_a_page_heading_does_not_split_the_text() -> None:
    text = "## Page 1\n\nIntro. See also ## Page 7 for more.\n\n### Page 9\n\nA deeper heading."

    chunks = chunk_pages(text, 2000, 300)

    assert [chunk.heading for chunk in chunks] == ["Page 1"]
