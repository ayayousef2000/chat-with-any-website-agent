"use strict";

// Chat With Any Website Agent: front-end behaviour.
// The helpers up to "Page wiring" only use `document` for creating elements. Everything is built from text nodes;
// no HTML string is ever inserted.

// The same address as the tab icon, including its version, so both come from one cached file.
const AVATAR_URL = "/static/favicon.svg?v=2";
const STORAGE_KEY = "chat-with-any-website-agent:page";

function createElement(tag, className) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  return element;
}

// An error response from the server, with its HTTP status.
class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "HttpError";
    this.status = status;
  }
}

// "just now", "4 minutes ago", "2 hours ago"
function formatAge(seconds) {
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.round(seconds / 3600);
  return `${hours} hour${hours === 1 ? "" : "s"} ago`;
}

// "Opened just now", "Opened 4 minutes ago": how long ago this visitor opened the page, given that time (ms).
// It is the visitor's own time. The server never says how old a stored copy is, because that would tell one visitor
// that someone else loaded the same address a moment ago.
function openedText(openedAt, now = Date.now()) {
  return `Opened ${formatAge(Math.max(0, (now - openedAt) / 1000))}`;
}

// Asks a question. The server loads the page again by itself when its saved copy is gone or has changed, so one
// request is enough. `post(path, payload)` sends a request and rejects with an HttpError on failure.
function askQuestion(post, url, question, history = []) {
  // The earlier turns are only sent when there are some, so a first question has no history.
  const payload = history.length ? { url, question, history } : { url, question };
  return post("/api/ask", payload);
}

// The last few questions and answers of this conversation, which are sent with the next question so that a follow-up
// like "When did he die?" can be understood. They live only in this tab; the server keeps no conversation.
const HISTORY_TURNS = 3;

function rememberTurn(history, question, answer, limit = HISTORY_TURNS) {
  return [...history, { question, answer }].slice(-limit);
}

// Puts the assistant's icon beside one of its messages. Returns the row that holds both.
function withAvatar(bubble) {
  const row = createElement("div", "row");
  const avatar = createElement("img", "avatar");
  avatar.src = AVATAR_URL;
  avatar.alt = "Assistant";
  avatar.width = 32;
  avatar.height = 32;
  row.append(avatar, bubble);
  return row;
}

// Asks the user to confirm an action in a modal <dialog> and resolves to true only if they confirm.
// `parts` holds the dialog and its title, text and confirm button elements. The dialog closes by itself on
// Escape, on its buttons (a form with method="dialog") and, handled here, on a click on the dimmed backdrop.
// Browsers without <dialog> support fall back to the built-in confirmation.
function confirmDialog(parts, { title, message, confirmLabel }) {
  const { dialog, titleElement, textElement, confirmButton } = parts;
  if (typeof dialog.showModal !== "function") return Promise.resolve(window.confirm(`${title}\n\n${message}`));
  titleElement.textContent = title;
  textElement.textContent = message;
  confirmButton.textContent = confirmLabel;
  return new Promise((resolve) => {
    const onBackdropClick = (event) => {
      if (event.target === dialog) dialog.close();
    };
    dialog.addEventListener("click", onBackdropClick);
    dialog.addEventListener(
      "close",
      () => {
        dialog.removeEventListener("click", onBackdropClick);
        resolve(dialog.returnValue === "confirm");
      },
      { once: true },
    );
    dialog.returnValue = "";
    dialog.showModal();
  });
}

// If a slow step takes longer than `delay` milliseconds, shows `message`. Returns a function that cancels it.
// An answer can take long when the free plans of the AI services ask the app to wait.
function startSlowNotice(setStatus, message, delay = 8000) {
  const timer = setTimeout(() => setStatus(message), delay);
  return () => clearTimeout(timer);
}

const SLOW_ANSWER_NOTICE = "Still working. The free plans of the AI services limit how fast questions are answered, so this can take a minute or more.";

// Only absolute http(s) addresses may become links.
function safeUrl(value) {
  return /^https?:\/\/[^\s]+$/i.test(value) ? value : null;
}

// Strips Markdown symbols so source excerpts read as plain text.
function plainText(markdown) {
  return markdown
    .replace(/^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$\n?/gm, "")
    .replace(/^[ \t]*\|(.*)$/gm, (_, row) => row.replace(/\|\s*$/, "").split("|").map((cell) => cell.trim()).filter(Boolean).join("  \u00b7  "))
    .replace(/```\w*\n?/g, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\[([^\]]+)\]\(https?:\/\/[^)]*\)/g, "$1")
    .replace(/(\*\*|__)(.+?)\1/g, "$2")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1$2")
    .replace(/`([^`\n]+)`/g, "$1")
    .replace(/\\([\\`*_{}[\]()#+\-.!|>~<])/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function friendlyError(error) {
  if (error instanceof TypeError) {
    return "We couldn't connect. Check your internet connection and try again.";
  }
  return error && error.message ? error.message : "Something went wrong. Please try again.";
}

// The note shown under an error when a failed load left the previous page in place.
function stillLoadedNote(title) {
  return `Your current page, "${title}", is still loaded, so you can keep asking questions about it.`;
}

// Code is written with one backtick or, as documentation written in reStructuredText does, with two: ``write()``.
const INLINE = /(``[^`\n]+``|`[^`\n]+`)|(\*\*[^*\n]+?\*\*)|(\*[^\s*][^*\n]*?\*)|(\[\d+\])|(\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;

// Renders inline Markdown (code, bold, italic, links) and [n] citation markers into `parent`.
// `cite(number)` returns the node to use for a citation, or null to leave the marker as plain text.
function renderInline(parent, text, cite) {
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const token = match[0];
    if (match.index > last) parent.append(text.slice(last, match.index));
    if (match[1]) {
      const code = createElement("code");
      const fence = token.startsWith("``") ? 2 : 1; // the backticks around the code are not part of it
      code.textContent = token.slice(fence, -fence);
      parent.append(code);
    } else if (match[2]) {
      const strong = createElement("strong");
      renderInline(strong, token.slice(2, -2), cite);
      parent.append(strong);
    } else if (match[3]) {
      const emphasis = createElement("em");
      renderInline(emphasis, token.slice(1, -1), cite);
      parent.append(emphasis);
    } else if (match[4]) {
      parent.append(cite(Number(token.slice(1, -1))) || token);
    } else {
      const split = token.indexOf("](");
      const href = safeUrl(token.slice(split + 2, -1));
      if (href) {
        const link = createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = token.slice(1, split);
        parent.append(link);
      } else {
        parent.append(token);
      }
    }
    last = match.index + token.length;
  }
  if (last < text.length) parent.append(text.slice(last));
}

const FENCE = /^\s*```/;
const HEADING = /^#{1,6}\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+/;
const ORDERED = /^\s*\d+[.)]\s+/;

function startsBlock(line) {
  return FENCE.test(line) || HEADING.test(line) || BULLET.test(line) || ORDERED.test(line);
}

// Renders a safe subset of Markdown (paragraphs, headings, lists, code blocks) into `container`.
function renderMarkdown(container, text, cite) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
    } else if (FENCE.test(line)) {
      const code = [];
      i += 1;
      while (i < lines.length && !FENCE.test(lines[i])) code.push(lines[i++]);
      i += 1; // the closing fence, if there is one
      const pre = createElement("pre");
      const inner = createElement("code");
      inner.textContent = code.join("\n");
      pre.append(inner);
      container.append(pre);
    } else if (HEADING.test(line)) {
      const heading = createElement("p", "md-heading");
      heading.dir = "auto";
      renderInline(heading, line.match(HEADING)[1], cite);
      container.append(heading);
      i += 1;
    } else if (BULLET.test(line) || ORDERED.test(line)) {
      const marker = ORDERED.test(line) ? ORDERED : BULLET;
      const list = createElement(marker === ORDERED ? "ol" : "ul");
      list.dir = "auto";
      while (i < lines.length && marker.test(lines[i])) {
        const item = createElement("li");
        renderInline(item, lines[i].replace(marker, ""), cite);
        list.append(item);
        i += 1;
      }
      container.append(list);
    } else {
      const paragraph = createElement("p");
      paragraph.dir = "auto";
      let first = true;
      while (i < lines.length && lines[i].trim() && (first || !startsBlock(lines[i]))) {
        if (!first) paragraph.append(createElement("br"));
        renderInline(paragraph, lines[i], cite);
        first = false;
        i += 1;
      }
      container.append(paragraph);
    }
  }
}

// ---------------------------------------------------------------------------------------------------
// Finding the part of a source that answers the question
//
// A source is a whole section of a page, but a question is usually about one fact in it. The part that matches
// the question and the answer best is shown first, highlighted, with the full section one click away. Matching
// compares three-letter groups instead of whole words, so it works across word forms ("ديانته" and "الديانة"),
// across languages written without spaces, and needs no outside service.
// ---------------------------------------------------------------------------------------------------

const SHORT_SOURCE = 320; // a source this short is shown whole
const MAX_UNIT = 240; // a longer line is split into sentences
const HARD_LIMIT = 320; // and a sentence longer than this is split at a space
const MAX_CONTEXT = 420; // a passage grows with neighbouring sentences up to about this length
let sourceCounter = 0;

// Lowercases and removes accents, Arabic vowel marks and variant letter forms, so word forms can be compared.
function normalizeForMatch(text) {
  return text
    .normalize("NFKD")
    .replace(/\p{M}/gu, "")
    .toLowerCase()
    .replace(/\u0671/g, "\u0627")
    .replace(/\u0649/g, "\u064a")
    .replace(/\u0629/g, "\u0647")
    .replace(/\u0640/g, "");
}

function gramsOfToken(token) {
  if (token.length < 3) return /\d/.test(token) ? [token] : [];
  const grams = [];
  for (let i = 0; i <= token.length - 3; i += 1) grams.push(token.slice(i, i + 3));
  return grams;
}

// The set of three-letter groups of every word in a text (short numbers count as they are).
function gramSet(text) {
  const grams = new Set();
  for (const token of normalizeForMatch(text).match(/[\p{L}\p{N}]+/gu) || []) {
    for (const gram of gramsOfToken(token)) grams.add(gram);
  }
  return grams;
}

function splitAtSpaces(text) {
  const parts = [];
  let rest = text;
  while (rest.length > HARD_LIMIT) {
    const cut = rest.lastIndexOf(" ", HARD_LIMIT);
    const at = cut > HARD_LIMIT / 2 ? cut : HARD_LIMIT;
    parts.push(rest.slice(0, at).trim());
    rest = rest.slice(at).trim();
  }
  if (rest) parts.push(rest);
  return parts;
}

// Cuts a source into lines, then long lines into sentences (including Arabic, Chinese and Japanese stops).
// Each unit remembers the line it came from.
function splitUnitObjects(plain) {
  const units = [];
  plain.split("\n").forEach((line, lineIndex) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const pieces = trimmed.length <= MAX_UNIT ? [trimmed] : trimmed.split(/(?<=[.!?\u061f\u06d4\u061b])\s+|(?<=[\u3002\uff01\uff1f])\s*/u);
    for (const piece of pieces) {
      const sentence = piece.trim();
      if (!sentence) continue;
      for (const part of sentence.length > HARD_LIMIT ? splitAtSpaces(sentence) : [sentence]) {
        units.push({ text: part, line: lineIndex });
      }
    }
  });
  return units;
}

function splitUnits(plain) {
  return splitUnitObjects(plain).map((unit) => unit.text);
}

// Names of code in backticks in the answer, such as `Path.exists()` -> "exists".
function codeNames(answer) {
  const names = new Set();
  for (const match of answer.matchAll(/`([^`\n]+)`/g)) {
    const name = match[1].replace(/\(.*$/, "").split(".").pop().trim();
    if (/^[\p{L}_][\p{L}\p{N}_]{2,}$/u.test(name)) names.add(name);
  }
  return names;
}

function escapeForRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Finds the part of `plain` that best matches the question and the answer, or null if nothing does.
// The part is a line or sentence plus, in running text, the sentences around it for context. Groups that appear
// in many lines count for little (the topic of the page is in most of them), and groups that appear in few count for
// a lot. A match needs at least three shared groups covering a tenth of the question and answer, so that a chance
// overlap such as "ics" in two unrelated words is not taken for an answer. If the answer names code in backticks
// (`Path.exists()`), the line that defines that name wins, together with the line after it.
function findPassage(plain, question, answer) {
  const parts = splitUnitObjects(plain);
  if (!parts.length) return null;
  const units = parts.map((part) => part.text);
  const query = gramSet(`${question} ${answer.replace(/\[\d+\]/g, " ")}`);
  const unitGrams = units.map(gramSet);
  const lines = new Map();
  for (const grams of unitGrams) for (const gram of grams) lines.set(gram, (lines.get(gram) || 0) + 1);

  const scoreOf = (i) => {
    let sum = 0;
    let count = 0;
    for (const gram of unitGrams[i]) {
      if (!query.has(gram)) continue;
      sum += Math.log(1 + units.length / lines.get(gram)) ** 2;
      count += 1;
    }
    return { score: sum / unitGrams[i].size ** 0.15, count };
  };

  // Lines that define a name the answer mentions as code are preferred over lines that merely resemble the question.
  // Names are tried in the order the answer gives them, because the first one is what the answer is about.
  const all = units.map((_, i) => i);
  let named = [];
  let byDefinition = [];
  for (const name of codeNames(answer)) {
    const escaped = escapeForRegExp(name);
    const starts = new RegExp(`^[\\p{L}\\p{N}_.]*${escaped}\\s*\\(`, "iu");
    const calls = new RegExp(`(^|[^\\p{L}\\p{N}_])${escaped}\\s*\\(`, "iu");
    const mentions = new RegExp(`(^|[^\\p{L}\\p{N}_])${escaped}($|[^\\p{L}\\p{N}_])`, "iu");
    byDefinition = all.filter((i) => starts.test(units[i]));
    if (!byDefinition.length) byDefinition = all.filter((i) => calls.test(units[i]));
    named = byDefinition.length ? byDefinition : all.filter((i) => mentions.test(units[i]));
    if (named.length) break;
  }

  let index = -1;
  let best = -1;
  let shared = 0;
  for (const i of named.length ? named : all) {
    const { score, count } = scoreOf(i);
    if (score > best) {
      best = score;
      index = i;
      shared = count;
    }
  }
  if (index < 0) return null;
  if (!named.length && (best <= 0 || shared < 3 || shared / query.size < 0.1)) return null;

  // Context: neighbouring sentences of the same line, and after a definition the line that explains it.
  let first = index;
  let last = index;
  let length = units[index].length;
  for (let grew = true; grew; ) {
    grew = false;
    if (first > 0 && parts[first - 1].line === parts[index].line && length + units[first - 1].length < MAX_CONTEXT) {
      first -= 1;
      length += units[first].length;
      grew = true;
    }
    if (last < units.length - 1 && parts[last + 1].line === parts[index].line && length + units[last + 1].length < MAX_CONTEXT) {
      last += 1;
      length += units[last].length;
      grew = true;
    }
  }
  if (byDefinition.includes(index)) {
    while (last < units.length - 1 && length + units[last + 1].length < MAX_CONTEXT) {
      last += 1;
      length += units[last].length;
    }
  }
  const unit = units.slice(first, last + 1).reduce((text, piece, i) => {
    if (i === 0) return piece;
    return text + (parts[first + i].line === parts[first + i - 1].line ? " " : "\n") + piece;
  }, "");

  // A word is highlighted if most of it matches and it is not a word that fills the whole section.
  const specific = Math.max(1, units.length * 0.15);
  const matches = (word) => {
    const grams = gramsOfToken(normalizeForMatch(word));
    const hits = grams.filter((gram) => query.has(gram));
    return grams.length > 0 && hits.length / grams.length >= 0.6 && hits.some((gram) => (lines.get(gram) || 0) <= specific);
  };
  return { units, index, first, last, unit, matches };
}

// The first lines of a source, up to about 280 characters, for when nothing in it matches.
function leadingPassage(plain) {
  let text = "";
  for (const unit of splitUnits(plain)) {
    if (text && text.length + unit.length > 280) break;
    text += (text ? "\n" : "") + unit;
  }
  return text;
}

// Appends `text` to `container` with the words that match the question and the answer wrapped in <mark>.
function appendHighlighted(container, text, matches) {
  let last = 0;
  for (const word of text.matchAll(/[\p{L}\p{M}\p{N}]+/gu)) {
    if (!matches(word[0])) continue;
    if (word.index > last) container.append(text.slice(last, word.index));
    const mark = createElement("mark");
    mark.textContent = word[0];
    container.append(mark);
    last = word.index + word[0].length;
  }
  if (last < text.length) container.append(text.slice(last));
}

// The whole section, with the passage that answers the question marked.
function appendFullSection(container, plain, passage) {
  const at = passage ? plain.indexOf(passage) : -1;
  if (at < 0) {
    container.textContent = plain;
    return;
  }
  const mark = createElement("mark", "passage");
  mark.textContent = passage;
  container.append(plain.slice(0, at), mark, plain.slice(at + passage.length));
}

// One source: its label and either its whole (short) text, or the matching passage with a toggle for the rest.
function buildSourceItem(source, options) {
  const item = createElement("div", "source");
  item.dir = "auto";
  const label = createElement("span", "source-label");
  // An excerpt of another page, which the agent opened by following a link, says which page it is from.
  const otherPage = options.pageUrl && source.url && source.url !== options.pageUrl && source.title;
  const where = [otherPage ? plainText(source.title) : "", source.heading ? plainText(source.heading) : ""];
  const name = where.filter(Boolean).join(" › ");
  label.textContent = name ? `[${source.number}] ${name}` : `[${source.number}]`;
  item.append(label);

  const plain = plainText(source.text);
  const full = createElement("div", "source-text");
  if (plain.length <= SHORT_SOURCE) {
    full.textContent = plain;
    item.append(full);
    return item;
  }

  const found = findPassage(plain, options.question || "", options.answer || "");
  const passage = createElement("div", "source-passage");
  if (found) {
    if (found.first > 0) passage.append("\u2026 ");
    appendHighlighted(passage, found.unit, found.matches);
    if (found.last < found.units.length - 1) passage.append(" \u2026");
  } else {
    passage.append(leadingPassage(plain), " \u2026");
  }
  appendFullSection(full, plain, found ? found.unit : null);
  full.hidden = true;
  sourceCounter += 1;
  full.id = `source-text-${sourceCounter}`;

  const toggle = createElement("button", "link toggle");
  toggle.type = "button";
  toggle.textContent = "Show full section";
  toggle.setAttribute("aria-expanded", "false");
  toggle.setAttribute("aria-controls", full.id);
  toggle.addEventListener("click", () => {
    const expand = full.hidden;
    full.hidden = !expand;
    passage.hidden = expand;
    toggle.textContent = expand ? "Show less" : "Show full section";
    toggle.setAttribute("aria-expanded", String(expand));
  });
  item.append(passage, toggle, full);
  return item;
}

// Builds the collapsible source list. Clicking a citation shows only that source; clicking the
// "Sources" line shows all of them.
function buildSources(sources, options = {}) {
  const details = createElement("details");
  const summary = createElement("summary");
  details.append(summary);

  const items = new Map();
  let only = null; // the source number currently shown alone, or null for all

  function render() {
    summary.textContent =
      only === null ? `Sources (${sources.length})` : `Sources (${sources.length}) · showing [${only}] only`;
    items.forEach((item, number) => {
      item.hidden = only !== null && number !== only;
    });
  }

  sources.forEach((source) => {
    const item = buildSourceItem(source, options);
    details.append(item);
    items.set(source.number, item);
  });

  summary.addEventListener("click", (event) => {
    event.preventDefault();
    if (details.open && only === null) {
      details.open = false;
    } else {
      only = null;
      details.open = true;
      render();
    }
  });

  function showOnly(number) {
    if (details.open && only === number) {
      only = null;
      details.open = false;
      render();
      return;
    }
    only = number;
    details.open = true;
    render();
    const item = items.get(number);
    // Start at the top of the excerpt, which may be taller than the visible part of the chat.
    item.scrollIntoView({ block: "start", behavior: options.reduceMotion ? "auto" : "smooth" });
    item.classList.add("highlight");
    setTimeout(() => item.classList.remove("highlight"), 1600);
  }

  render();
  return { details, showOnly, has: (number) => items.has(number) };
}

// Builds an assistant answer: Markdown text whose [n] markers are buttons that reveal the matching source.
function buildAnswer(text, sources, options = {}) {
  const body = createElement("div", "md");
  const sourceList = sources.length ? buildSources(sources, { ...options, answer: text }) : null;

  function cite(number) {
    if (!sourceList || !sourceList.has(number)) return null;
    const button = createElement("button", "cite");
    button.type = "button";
    button.textContent = `[${number}]`;
    button.setAttribute("aria-label", `Show source ${number}`);
    button.addEventListener("click", () => sourceList.showOnly(number));
    return button;
  }

  renderMarkdown(body, text, cite);
  return { body, details: sourceList ? sourceList.details : null };
}

// ---------------------------------------------------------------------------------------------------
// Page wiring
// ---------------------------------------------------------------------------------------------------

function init() {
  const ingestForm = document.getElementById("ingest-form");
  const askForm = document.getElementById("ask-form");
  const urlInput = document.getElementById("url");
  const questionInput = document.getElementById("question");
  const ingestButton = document.getElementById("ingest-button");
  const askButton = document.getElementById("ask-button");
  const newChatButton = document.getElementById("new-chat");
  const refreshButton = document.getElementById("refresh-page");
  const forgetButton = document.getElementById("forget-page");
  const statusEl = document.getElementById("status");
  const messagesEl = document.getElementById("messages");
  const pageBar = document.getElementById("page-bar");
  const pageLink = document.getElementById("page-link");
  const pageMeta = document.getElementById("page-meta");

  let page = null; // the page the questions are about: { url, title, openedAt }
  let history = []; // the last turns of this conversation: [{ question, answer }]
  let busy = false;

  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Shows one line of status. An optional note goes on a second, quieter line, so that information
  // (such as which page is still loaded) is not mixed into the error itself.
  function setStatus(text, isError = false, note = "") {
    statusEl.replaceChildren();
    if (text) {
      const main = createElement("span", "status-main");
      main.textContent = text;
      statusEl.append(main);
    }
    if (text && note) {
      const extra = createElement("span", "status-note");
      extra.textContent = note;
      statusEl.append(extra);
    }
    statusEl.classList.toggle("error", isError);
  }

  function updateControls() {
    urlInput.disabled = busy;
    ingestButton.disabled = busy;
    newChatButton.disabled = busy;
    refreshButton.disabled = busy;
    forgetButton.disabled = busy;
    questionInput.disabled = busy || page === null;
    askButton.disabled = busy || page === null;
  }

  function setBusy(value) {
    busy = value;
    updateControls();
  }

  function savePage() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(page));
    } catch {
      // Storage can be unavailable (private windows); the page just will not be remembered.
    }
  }

  // Shows how long ago the page was loaded; pages remembered by an older version of the script have no time.
  function renderPageMeta() {
    pageMeta.textContent = page && page.openedAt ? openedText(page.openedAt) : "";
  }

  function showPage(newPage) {
    page = newPage;
    const href = safeUrl(page.url);
    if (href) pageLink.href = href;
    pageLink.textContent = page.title;
    renderPageMeta();
    pageBar.hidden = false;
    savePage();
    updateControls();
  }

  function showEmptyState() {
    messagesEl.replaceChildren();
    history = []; // a new chat, a new page or deleted data starts a new conversation
    const empty = createElement("div", "empty");
    if (page === null) {
      const first = createElement("p");
      first.textContent = "No page loaded yet.";
      const hint = createElement("p");
      hint.textContent = "Paste an address above.";
      empty.append(first, hint);
    } else {
      const only = createElement("p");
      only.textContent = "Page loaded. Ask a question about it below.";
      empty.append(only);
    }
    messagesEl.append(empty);
  }

  function clearEmptyState() {
    messagesEl.querySelector(".empty")?.remove();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // Long answers should be read from the top, so scroll to the start of the new message.
  function scrollToMessage(element) {
    messagesEl.scrollTo({ top: Math.max(element.offsetTop - 12, 0), behavior: reduceMotion() ? "auto" : "smooth" });
  }

  function addMessage(role, text) {
    clearEmptyState();
    const element = createElement("div", `message ${role}`);
    element.dir = "auto";
    element.textContent = text;
    messagesEl.append(role.includes("assistant") ? withAvatar(element) : element);
    scrollToBottom();
    return element;
  }

  function addAnswer(text, sources, question) {
    clearEmptyState();
    const element = createElement("div", "message assistant");
    const answer = buildAnswer(text, sources, { reduceMotion: reduceMotion(), question, pageUrl: page && page.url });
    element.append(answer.body);
    if (answer.details) element.append(answer.details);
    const row = withAvatar(element);
    messagesEl.append(row);
    scrollToMessage(row);
  }

  function addTyping() {
    clearEmptyState();
    const element = createElement("div", "message assistant");
    element.setAttribute("aria-label", "Thinking");
    const dots = createElement("span", "typing");
    dots.append(createElement("span"), createElement("span"), createElement("span"));
    element.append(dots);
    const row = withAvatar(element);
    messagesEl.append(row);
    scrollToBottom();
    return row;
  }

  async function request(path, options) {
    const response = await fetch(path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
      throw new HttpError(detail || `Request failed with status ${response.status}`, response.status);
    }
    return data;
  }

  const postJson = (path, payload) =>
    request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

  ingestForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setBusy(true);
    setStatus("Reading the page… Long pages can take up to a minute.");
    try {
      const result = await postJson("/api/ingest", { url: urlInput.value });
      urlInput.value = result.url;
      showPage({ url: result.url, title: result.title, openedAt: Date.now() });
      showEmptyState();
      setStatus("");
    } catch (error) {
      // The previous page, if any, stays loaded and visible in the bar above the conversation.
      setStatus(friendlyError(error), true, page ? stillLoadedNote(page.title) : "");
    } finally {
      setBusy(false);
      if (page) questionInput.focus();
    }
  });

  askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || !page) return;

    addMessage("user", question);
    questionInput.value = "";
    const typing = addTyping();
    setBusy(true);
    setStatus("Thinking…");
    const cancelSlowNotice = startSlowNotice(setStatus, SLOW_ANSWER_NOTICE);
    try {
      const result = await askQuestion(postJson, page.url, question, history);
      typing.remove();
      addAnswer(result.answer, result.sources, question);
      history = rememberTurn(history, question, result.answer);
    } catch (error) {
      typing.remove();
      addMessage("assistant error", friendlyError(error));
      questionInput.value = question; // keep the question so it can be sent again
    } finally {
      cancelSlowNotice();
      setStatus("");
      setBusy(false);
      questionInput.focus();
    }
  });

  refreshButton.addEventListener("click", async () => {
    if (!page) return;
    setBusy(true);
    setStatus("Fetching the page again…");
    try {
      const result = await postJson("/api/ingest", { url: page.url, refresh: true });
      showPage({ url: result.url, title: result.title, openedAt: Date.now() });
      setStatus(`Refreshed "${result.title}".`);
    } catch (error) {
      setStatus(friendlyError(error), true);
    } finally {
      setBusy(false);
    }
  });

  forgetButton.addEventListener("click", async () => {
    if (!page) return;
    const confirmed = await confirmDialog(
      {
        dialog: document.getElementById("confirm-dialog"),
        titleElement: document.getElementById("confirm-title"),
        textElement: document.getElementById("confirm-text"),
        confirmButton: document.getElementById("confirm-ok"),
      },
      {
        title: "Delete the saved copy?",
        message: `This removes the stored text of "${page.title}". You can load the page again at any time.`,
        confirmLabel: "Delete",
      },
    );
    if (!confirmed) return;
    setBusy(true);
    try {
      await request(`/api/page?url=${encodeURIComponent(page.url)}`, { method: "DELETE" });
      page = null;
      pageBar.hidden = true;
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch {
        // Ignore unavailable storage.
      }
      // The empty state that follows already says what to do next, so no extra message is shown.
      showEmptyState();
      setStatus("");
    } catch (error) {
      setStatus(friendlyError(error), true);
    } finally {
      setBusy(false);
    }
  });

  newChatButton.addEventListener("click", () => {
    showEmptyState();
    setStatus("");
    questionInput.focus();
  });

  // Restore the page from earlier in this browser tab; its text is still stored on the server.
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (saved && saved.url && saved.title) {
      urlInput.value = saved.url;
      showPage(saved);
    }
  } catch {
    // Ignore unreadable or unavailable storage.
  }
  showEmptyState();
  updateControls();
  setInterval(renderPageMeta, 30_000); // keeps "Opened 4 minutes ago" true while the tab stays open
}

init();
