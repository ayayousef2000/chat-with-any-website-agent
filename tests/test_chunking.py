from itertools import pairwise

import pytest

from chat_with_any_website_agent.chunking import chunk_text, contextualize


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
