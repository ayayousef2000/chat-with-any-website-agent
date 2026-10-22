# Chat With Any Website Agent

An agentic RAG workflow built with [LangGraph](https://langchain-ai.github.io/langgraph/) and LangChain. It answers questions about a web page or a PDF file with cited sources. It grades what it finds, searches again with a better query when the evidence is not enough, checks its own citations, and says that the source does not cover the question instead of inventing an answer.

This is the agent version of [Chat With Any Website](https://github.com/ayayousef2000/chat-with-any-website), which answers questions about one page or PDF with a fixed pipeline. It is a separate project with its own repository, keys and vector database.

> **Status.** Working, and evaluated on a small question set (see [Results](#results)). The agent reads the page or PDF it is given and, when that is not enough, follows links to other pages of the same site.

## How it works

```
question → analyze → retrieve → grade ─ enough ─→ generate → verify ─ citations hold ─→ answer
                        ↑          │                  │          │
                        │          │                  │          └─ citations do not hold ─→ repair (once) ─→ verify
                        └ rewrite ─┘ not enough       └─ no answer in the excerpts ─┐
                                     and a retry fits                               ↓
                                     otherwise ───────────────────────────────→ abstain
```

When the excerpts are not enough, the graph first tries `choose` → `follow` → `retrieve` (pages of the same site, see below) and only then `rewrite`.

- **Retrieve.** A hybrid search (keywords and vectors) in Weaviate, then a Cohere rerank keeps the best few excerpts.
- **Grade.** A small model decides whether the excerpts are enough to answer, and says what is missing when they are not.
- **Follow links.** When the pages read so far are not enough, a small model is shown a numbered list of the links on them and picks at most two by number, so it can only choose links the page really has. Only links to the same site are offered, and only pages that the site's `robots.txt` allows are opened. The new pages are loaded like any source and searched together with the first one, within `AGENT_MAX_LINK_HOPS` rounds and `AGENT_MAX_PAGES` pages for a question.
- **Rewrite and retry.** When the evidence is weak, the query is rewritten and searched again, within a limit. A new search that finds nothing the agent has not already seen ends the question, because the same excerpts would get the same grade.
- **Generate and verify.** A larger model writes the answer and cites the excerpts as `[1]`, `[2]`. Plain code then checks that every cited excerpt exists and that each cited sentence shares its content words with it. A rejected answer is corrected once.
- **Abstain.** When the source does not cover the question, the answer says so and gives the reason.

Every question runs within hard limits on steps, model calls and rewrites, and the graph looks ahead so that it stops before a limit is hit.

Also included:

- **PDF reading** page by page, with the page number in the excerpt heading. A question can name a page ("What does page 5 say about the heads?") and is then answered from that page only. A PDF with no selectable text, or mostly scans, is refused.
- **Safe fetching.** Only public addresses on the standard ports are fetched, every redirect is checked again, and size and time limits apply.
- **Several languages.** The questions and the sources can be in different languages (the evaluation includes Arabic).
- **Free-plan friendly.** Calls and tokens are counted, and requests are paced to stay within the limits of a plan.

## Results

The agent was evaluated on 24 questions over public sources: pages of the Python tutorial, the paper "Attention Is All You Need" (PDF) and an Arabic Wikipedia article. A separate model judges each answer against a reference answer and the excerpts the agent used. The set is small, so treat the numbers as a first measurement and not as a benchmark.

| Category | Questions | Correct |
|---|---|---|
| Single page | 4 | 3 |
| Two sources given | 4 | 4 |
| PDF, facts from a given page | 3 | 3 |
| Arabic | 2 | 2 |
| Unanswerable (the right answer is that the source does not cover it) | 5 | 5 |
| **Multi-page** (the answer is on a page linked from the start page) | 6 | 6 |
| **All** | 24 | **23 (96%)** |

- All 24 questions were answered in one run with the current version. Every answer but one was judged faithful to its excerpts. That one ("When is the else clause of a loop not executed?") adds that a `return` or an exception also skips the clause, which the excerpt says too, so the small judge model was too strict and the answer is right.
- The multi-page questions start from the Python tutorial index, which is only a list of chapters. The agent finds that this is not enough and opens the pages that look relevant. Without link following, the same agent answers none of them.
- An answer may not state a number that the excerpts it cites do not write. Asked how many exceptions Python has, which no page states, the agent refuses; an earlier version counted the lines of a list and answered 68.
- A question costs about 6,000 Groq tokens on average, including the judge. An answer from one page takes 2 model calls and about 5,000 tokens; a question that needs links takes about 5 calls and 6,000 to 12,000 tokens.
- Known limits: the check of the citations compares words and numbers, so an answer can still add a detail that the source does not state. The agent does not always open the best page first (it sometimes picks the language reference over the tutorial chapter), but it still found the answer in every multi-page question.

## Requirements

Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). You need your own API keys for [Cohere](https://dashboard.cohere.com/api-keys), [Weaviate Cloud](https://console.weaviate.cloud) and [Groq](https://console.groq.com/keys). The free plans are enough.

## Setup

```bash
uv sync                      # create the virtual environment and install the locked dependencies
cp .env.example .env         # then fill in your own keys; the file is never committed
uv run agent-smoke --dry-run # show what a check of the three services would do, without calling anything
uv run agent-smoke --prepare # check the services and create or update the Weaviate collection and tenants
```

## Usage

Ask a question about a page or a PDF file. The source is stored the first time it is loaded, and a stored page is loaded again when it changes:

```bash
uv run agent-ask "https://docs.python.org/3/tutorial/index.html" "What does the pass statement do?"
```

It prints the answer, the sources it cites, the pages it opened by following links, every step the agent took and what the question cost. Useful options:

- `--also URL` searches a further source as well; it can be repeated.
- `--no-follow` stays on the given sources and does not follow links.
- `--refresh` downloads the source again even if a copy is stored and the page did not change.
- `--no-check` uses a stored copy without asking the website whether the page changed.
- `--tenant NAME` uses another Weaviate tenant, for example `demo`.
- `--no-steps` hides the steps.

Or try it in a browser:

```bash
uv run agent-web
```

Then open <http://127.0.0.1:8000>, paste the address of a page or PDF file, click **Load page** and ask a question. Answers cite their sources as `[1]`, `[2]`: click one to see the excerpt it comes from, with the sentence that answers the question highlighted. Follow-up questions such as "When did he start it?" are understood from the last few turns of the chat. **Refresh** loads the page again, **Delete data** removes the stored copy and **New chat** starts a new conversation.

The page is for one person on one computer. The server listens on `127.0.0.1` only, refuses requests that do not come from its own page, and does one thing at a time because of the limits of the free plans. It stores pages in the Weaviate tenant `demo` (change it with `--tenant`) and follows links as the settings allow.

Evaluate the agent on a question set:

```bash
uv run agent-eval validate eval/questions.json     # check the file; no service is used
WEAVIATE_TENANT=eval uv run agent-eval run eval/questions.json --system agent --out eval/results.json
uv run agent-eval summary eval/results.json        # read saved results; no service is used
```

A run stops cleanly at the limits of calls and tokens in the settings, and `--resume` continues it. Results are saved after every question and include how the agent got to each answer.

`--system` also accepts `agent-single` (the agent without links), `plain` (one search and one answer) and `crawl` (the plain pipeline over a crawled site), to compare the agent with simpler systems. `--category NAME` answers only one category of the set, and `--fresh` empties the `eval` tenant first, so that loading pages is part of the cost.

## Docker

```bash
docker build --tag chat-with-any-website-agent .
docker run --rm --publish 8000:8000 --env-file .env chat-with-any-website-agent
```

Then open <http://localhost:8000>.

## Deploy on Render

[`render.yaml`](render.yaml) creates two free web services: `production` (the `main` branch, Weaviate tenant `production`) and `staging` (the `develop` branch, tenant `staging`).

1. In Render choose **New**, then **Blueprint**, and select this repository and the `main` branch.
2. For each service, enter the secrets Render asks for: `COHERE_API_KEY`, `WEAVIATE_URL`, `WEAVIATE_API_KEY` and `GROQ_API_KEY`. Use one Groq key and leave the optional settings empty.
3. Wait for the build. The page is at the address of each service.

Good to know:

- A free service sleeps after 15 minutes without traffic and takes about a minute to wake.
- A free Weaviate plan allows three tenants, so give the two services a cluster of their own, not the one you use on your computer.
- If the page answers "This server answers only to pages that it serves itself", set `WEB_ALLOWED_HOSTS` to the address of the service.
- Each visitor is limited in questions and page loads (the `RATE_LIMIT_*` settings below).

## Configuration

Settings are read from the environment or from `.env`; `.env.example` lists all of them with their defaults. The ones that shape the agent:

| Setting | Default | Meaning |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | The model that writes the answer |
| `GROQ_GRADE_MODEL` | `openai/gpt-oss-20b` | The model that grades the excerpts and rewrites queries |
| `GROQ_BACKUP_API_KEYS` | empty | More Groq keys, separated by commas, used in turn when one reaches its limit |
| `SOURCE_CHECK_SECONDS` | 0 | A stored page checked less than this many seconds ago is not checked again; 0 checks at every question |
| `SOURCE_MAX_AGE_HOURS` | 0 | A stored copy of a source older than this many hours is loaded again; 0 never does that |
| `AGENT_MAX_STEPS` | 24 | The most graph steps for one question |
| `AGENT_MAX_LLM_CALLS` | 12 | The most calls to the language model for one question |
| `AGENT_MAX_REWRITES` | 2 | The most times the query is rewritten |
| `AGENT_MAX_LINK_HOPS` | 3 | The most times the agent chooses links of the site to open; 0 keeps it on the pages it was given |
| `AGENT_MAX_PAGES` | 5 | The most pages the agent loads by following links; 0 keeps it on the pages it was given |
| `WEAVIATE_TENANT` | `development` | Keeps the data of different uses apart |
| `WEB_ALLOWED_HOSTS` | empty | Names the web server answers to besides this computer, separated by commas (a custom domain) |
| `WEB_MAX_STORED_CHUNKS` | 0 | When more chunks than this are stored, the stored pages are deleted before the next page is loaded; 0 keeps them all |
| `RATE_LIMIT_ENABLED` | `True` | Limit how often one visitor may ask questions and load pages |
| `RATE_LIMIT_ASKS_PER_MINUTE` | 6 | Questions one visitor may ask in a minute |
| `RATE_LIMIT_ASKS_PER_DAY` | 60 | Questions one visitor may ask in a day (a sliding 24 hours) |
| `RATE_LIMIT_LOADS_PER_HOUR` | 20 | Page loads and deletes one visitor may make in an hour |

The agent warns at startup when a limit is too small for the link hops and retries it allows.

## Development

```bash
uv run pre-commit install        # install the Git hooks (they run the checks before every commit)
uv run poe check                 # type-check (mypy), lint and format-check (ruff), and run the tests (pytest)
```

The tests use fakes, so they call no external service. Commits follow [Conventional Commits](https://www.conventionalcommits.org/). See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Planned

- A check of the answer by a model, to catch claims that the cited excerpt does not support.
- A better choice of the first link to open.

## License

[MIT](LICENSE)
