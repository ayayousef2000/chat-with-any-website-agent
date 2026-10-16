"""Small PDF files for the tests, made in memory so that no file has to be downloaded or kept."""

from io import BytesIO
from typing import TYPE_CHECKING

from pypdf import PdfReader, PdfWriter

if TYPE_CHECKING:
    from collections.abc import Sequence


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: Sequence[Sequence[str | tuple[str, int]]], title: str | None = None) -> bytes:
    """Build a PDF with one page per list of lines. A page with no lines has no text, like a scanned page.

    A line is a text, or a text with its font size in points (12 when it is only a text).
    """
    body = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []

    def add(content: bytes) -> int:
        offsets.append(len(body))
        number = len(offsets)
        body.extend(f"{number} 0 obj\n".encode() + content + b"\nendobj\n")
        return number

    count = len(pages)
    first_page = 4  # 1 catalog, 2 page tree, 3 font, then a page object and a content stream for every page
    kids = " ".join(f"{first_page + 2 * index} 0 R" for index in range(count))
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode())
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index, lines in enumerate(pages):
        content_number = first_page + 2 * index + 1
        add(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_number} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>".encode()
        )
        sized = [line if isinstance(line, tuple) else (line, 12) for line in lines]
        operations = [
            f"BT /F1 {size} Tf 72 {720 - 28 * row} Td ({_escape(text)}) Tj ET" for row, (text, size) in enumerate(sized)
        ]
        stream = "\n".join(operations).encode("latin-1")
        add(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    info = add(f"<< /Title ({_escape(title)}) >>".encode()) if title else None

    xref = len(body)
    body.extend(f"xref\n0 {len(offsets) + 1}\n".encode() + b"0000000000 65535 f \n")
    for offset in offsets:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    trailer = f"<< /Size {len(offsets) + 1} /Root 1 0 R" + (f" /Info {info} 0 R" if info else "") + " >>"
    body.extend(f"trailer\n{trailer}\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(body)


def protect(pdf: bytes, password: str) -> bytes:
    """The same PDF, locked with a password."""
    writer = PdfWriter(clone_from=PdfReader(BytesIO(pdf)))
    writer.encrypt(password, algorithm="AES-256")
    out = BytesIO()
    writer.write(out)
    return out.getvalue()
