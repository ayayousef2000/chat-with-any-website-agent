"""Extracting the readable content of a web page.

The main text is kept as Markdown, and the navigation, advertisements and footers of the page are dropped. The
contact details that such an extraction throws away (phone and WhatsApp numbers and email addresses in icon buttons,
cards and footers) are collected separately and added to the end of the text.
"""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import trafilatura

from chat_with_any_website_agent.errors import ExtractionError

NO_TEXT_MESSAGE = (
    "No readable text was found on this page. It may load its content with JavaScript or block automated tools. "
    "Try a different page."
)
UNREADABLE_MESSAGE = "This page could not be read. Try a different one."

# A page with less text than this has nothing to answer from. It is typically an empty shell that fills in its
# content with JavaScript: such pages measure a few dozen characters, while the smallest real pages (like
# example.com) still have well over this.
MIN_READABLE_CHARS = 100

# Contact details that the text extraction throws away: they sit in icon buttons, cards and footers, and the number
# or address of a link is only in its ``href``. Every pattern has bounded lengths so a page cannot make it slow.
CONTACT_LIMIT = 10  # per kind
_CONTACT_SCAN_CHARS = 2_000_000
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,24}")
_NOT_EMAIL_ENDINGS = frozenset({"png", "jpg", "jpeg", "gif", "svg", "webp", "avif", "ico", "css", "js"})  # logo@2x.png
_LABELLED_PHONE = re.compile(
    r"\b(?P<label>phone|tel|telephone|call|mobile|cell|whatsapp)\b[^\d+(]{0,12}(?P<number>\+?\(?\d[\d\s().-]{5,17}\d)",
    re.IGNORECASE,
)
_PHONE_CHARS = re.compile(r"[+\d\s().-]+")
_DATE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}")
_WHATSAPP_HOSTS = frozenset({"wa.me", "api.whatsapp.com", "web.whatsapp.com"})


@dataclass(frozen=True)
class ExtractedPage:
    """The main content of a web page.

    Attributes:
        url: The address of the page.
        title: The title of the page, or its address when it has none.
        text: The readable content as Markdown, followed by a "Contact details" section when the page shows any.
    """

    url: str
    title: str
    text: str


def extract_page(html: bytes, url: str) -> ExtractedPage:
    """Keep the main readable content of a page as Markdown and drop navigation, advertisements and footers.

    Args:
        html: The HTML of the page.
        url: The address of the page, used as a hint for the extraction and as the fallback title.

    Returns:
        The extracted page.

    Raises:
        ExtractionError: If the HTML cannot be read, or the page has too little text to answer from.
    """
    tree = trafilatura.load_html(html)
    if tree is None:
        raise ExtractionError(UNREADABLE_MESSAGE)

    contacts = _contact_details(tree)  # before the extraction, which changes the tree
    metadata = trafilatura.extract_metadata(tree, default_url=url)
    text = trafilatura.extract(
        tree,
        url=url,
        output_format="markdown",
        include_tables=True,
        include_comments=False,
        include_links=False,
        favor_recall=True,
    )
    # The contact details are added only now: a page of nothing but a contact link has no readable text.
    if not text or len(text.strip()) < MIN_READABLE_CHARS:
        raise ExtractionError(NO_TEXT_MESSAGE)

    title = metadata.title if metadata and metadata.title else url
    return ExtractedPage(url=url, title=title, text=f"{text}\n\n{contacts}" if contacts else text)


def _phone_digits(value: str) -> str | None:
    """The digits of a phone number, or ``None`` when there are too few or too many for a real number."""
    digits = re.sub(r"\D", "", value)
    return digits if 7 <= len(digits) <= 15 else None


def _link_contact(href: str, shown: str) -> tuple[str, str, str] | None:
    """Read a ``tel:``, ``mailto:`` or WhatsApp link as ``(kind, key, text)``, or ``None`` for any other link."""
    href = href.strip()
    scheme = href.split(":", 1)[0].lower()
    if scheme == "tel":
        number = unquote(href[4:]).split(";")[0].strip()
        digits = _phone_digits(number) if _PHONE_CHARS.fullmatch(number) else None
        return ("Phone", digits, number) if digits else None
    if scheme == "mailto":
        address = unquote(href[7:]).split("?")[0].split(",")[0].strip()
        return ("Email", address.lower(), address) if _EMAIL.fullmatch(address) else None
    parts = urlsplit(href)
    host = (parts.hostname or "").lower()
    if parts.scheme in {"http", "https"} and host in _WHATSAPP_HOSTS:
        candidate = parts.path.strip("/") if host == "wa.me" else (parse_qs(parts.query).get("phone") or [""])[0]
    elif scheme == "whatsapp":
        candidate = (parse_qs(parts.query).get("phone") or [""])[0]
    else:
        return None
    digits = _phone_digits(candidate) if re.fullmatch(r"\+?\d+", candidate) else None
    if not digits:
        return None
    # The number as the page writes it, when the link text is that number; otherwise plain digits.
    written = _PHONE_CHARS.fullmatch(shown) and _phone_digits(shown) == digits
    return "WhatsApp", digits, shown if written else f"+{digits}"


def _contact_details(tree: Any) -> str:
    """Collect phone numbers, WhatsApp numbers and email addresses from links and from the visible text.

    Returns:
        A short Markdown section, or an empty string when the page shows none of them.
    """
    found: dict[str, dict[str, str]] = {"Phone": {}, "WhatsApp": {}, "Email": {}}

    def add(kind: str, key: str, text: str) -> None:
        entries = found[kind]
        if key in entries:
            if entries[key] == f"+{key}":  # only digits so far; prefer the number as the page writes it
                entries[key] = text
        elif len(entries) < CONTACT_LIMIT:
            entries[key] = text

    for anchor in tree.iter("a"):
        contact = _link_contact(anchor.get("href") or "", " ".join(anchor.text_content().split()))
        if contact:
            add(*contact)

    words = tree.xpath("//text()[not(ancestor::script or ancestor::style or ancestor::noscript)]")
    visible = " ".join(word.strip() for word in words if word.strip())[:_CONTACT_SCAN_CHARS]
    for match in _EMAIL.finditer(visible):
        if match.group().rsplit(".", 1)[-1].lower() not in _NOT_EMAIL_ENDINGS:
            add("Email", match.group().lower(), match.group())
    for match in _LABELLED_PHONE.finditer(visible):
        digits = _phone_digits(match.group("number"))
        if digits and not _DATE.fullmatch(match.group("number").strip()):
            add(
                "WhatsApp" if match.group("label").lower() == "whatsapp" else "Phone",
                digits,
                match.group("number").strip(),
            )

    lines = [f"- {kind}: {text}" for kind, entries in found.items() for text in entries.values()]
    return "## Contact details\n\n" + "\n".join(lines) if lines else ""
