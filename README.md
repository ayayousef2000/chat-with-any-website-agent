# Chat With Any Website Agent

An agentic RAG workflow built with [LangGraph](https://langchain-ai.github.io/langgraph/) and LangChain. It answers questions about a web page or a PDF file with cited sources. It grades what it finds, searches again with a better query when the evidence is not enough, checks its own citations, and says that the source does not cover the question instead of inventing an answer.

This is the agent version of [Chat With Any Website](https://github.com/ayayousef2000/chat-with-any-website), which answers questions about one page or PDF with a fixed pipeline. This project is separate: it has its own repository, API keys and vector database, and it copies modules from the original instead of importing them.

> **Status.** The single-source agent works and is evaluated (see [Results](#results)). Following links to other pages of a site is the next step and is not built yet.

## How it works

```
question → analyze → retrieve → grade ─ enough ─→ generate → verify ─ citations hold ─→ answer
                        ↑          │                  │          │
                        │          │                  │          └─ citations do not hold ─→ repair (once) ─→ verify
                        └ rewrite ─┘ not enough       └─ no answer in the excerpts ─┐
                                     and a retry fits                               ↓
                                     otherwise ───────────────────────────────→ abstain
```

- **Retrieve.** A hybrid search (keywords and vectors) in Weaviate, then a Cohere rerank keeps the best few excerpts.
- **Grade.** A small model decides whether the excerpts are enough to answer, and says what is missing when they are not.
- **Rewrite and retry.** When the evidence is weak, the query is rewritten and searched again, within a limit. A new search that finds nothing the agent has not already seen ends the question, because the same excerpts would get the same grade.
- **Generate and verify.** A larger model writes the answer and cites the excerpts as `[1]`, `[2]`. Plain code then checks that every cited excerpt exists and that each cited sentence shares its content words with it. A rejected answer is corrected once.
- **Abstain.** When the source does not cover the question, the answer says so and gives the reason.

Every question runs within hard limits on steps, model calls and rewrites, and the graph looks ahead so that it stops before a limit is hit.

Also included:

- **PDF reading** page by page, with the page number in the excerpt heading.
- **Safe fetching.** Only public addresses on the standard ports are fetched, every redirect is checked again, and size and time limits apply.
- **Several languages.** The questions and the sources can be in different languages (the evaluation includes Arabic).
- **Free-plan friendly.** Calls and tokens are counted, a per-key pace keeps requests within the limits of a plan, and a request that reaches a limit moves to the next Groq key when more than one is listed.

## Results

The agent was evaluated on 24 questions over public sources: pages of the Python tutorial, the paper "Attention Is All You Need" (PDF) and an Arabic Wikipedia article. A separate model judges each answer against a reference answer and the excerpts the agent used. The set is small, so treat the numbers as a first measurement and not as a benchmark.

| Category | Questions | Correct |
|---|---|---|
| Single page | 4 | 4 |
| Two sources given | 4 | 4 (one answer could not be judged) |
| PDF, facts from a given page | 3 | 3 |
| Arabic | 2 | 2 |
| Unanswerable (the right answer is that the source does not cover it) | 5 | 5 |
| **Multi-page** (the answer is on a page linked from the start page) | 6 | **0** |
| **All** | 24 | **17 of 23 judged (74%)** |

- Every answer was judged faithful to its excerpts.
- The multi-page questions fail because the agent reads one source and does not follow links yet. It searches, finds only the list of chapters, retries and then says that the source does not cover the question. This is the gap the next step is meant to close.
- A question costs about 4,800 Groq tokens on average. An answer needs 2 model calls. A question the source cannot answer needs 5, because of the retries.
- Known limits: the check of the citations compares words, so an answer can still add a detail that the source does not state. A check by a model is planned.

## Requirements

Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). You need your own API keys for [Cohere](https://dashboard.cohere.com/api-keys), [Weaviate Cloud](https://console.weaviate.cloud) and [Groq](https://console.groq.com/keys). The free plans are enough.

## Setup

```bash
uv sync                      # create the virtual environment and install the locked dependencies
cp .env.example .env         # then fill in your own keys; the file is never committed
uv run agent-smoke --dry-run # show what a check of the three services would do, without calling anything
uv run agent-smoke --prepare # check the services, create the Weaviate collection and tenants if missing, and add the field that keeps when a source was loaded
```

## Usage

Ask a question about a page or a PDF file. The source is loaded once and stored. At the next questions the website is asked whether the page changed (a very small request), and the stored copy is used if it did not; if the page changed, the copy is made again, so the answer is based on the page as it is when you ask:

```bash
uv run agent-ask "https://docs.python.org/3/tutorial/controlflow.html" "What does the pass statement do?"
```

It prints the answer, the sources it cites, every step the agent took and what the question cost. Useful options:

- `--also URL` searches a further source as well; it can be repeated.
- `--refresh` downloads the source again even if a copy is stored and the page did not change.
- `--no-check` uses a stored copy without asking the website whether the page changed.
- `--tenant NAME` uses another Weaviate tenant, for example `demo`.
- `--no-steps` hides the steps.

Evaluate the agent on a question set:

```bash
uv run agent-eval validate eval/questions.json     # check the file; no service is used
WEAVIATE_TENANT=eval uv run agent-eval run eval/questions.json --system agent --out eval/results.json
uv run agent-eval summary eval/results.json        # read saved results; no service is used
```

A run stops cleanly at the limits of calls and tokens in the settings, and `--resume` continues it. Results are saved after every question and include how the agent got to each answer.

## Configuration

Settings are read from the environment or from `.env`; `.env.example` lists all of them with their defaults. The ones that shape the agent:

| Setting | Default | Meaning |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | The model that writes the answer |
| `GROQ_GRADE_MODEL` | `openai/gpt-oss-20b` | The model that grades the excerpts and rewrites queries |
| `GROQ_BACKUP_API_KEYS` | empty | More Groq keys, separated by commas, used in turn when one reaches its limit |
| `SOURCE_CHECK_SECONDS` | 0 | A stored page checked less than this many seconds ago is not checked again; 0 checks at every question |
| `SOURCE_MAX_AGE_HOURS` | 0 | A stored copy of a source older than this many hours is loaded again; 0 never does that |
| `AGENT_MAX_STEPS` | 12 | The most graph steps for one question |
| `AGENT_MAX_LLM_CALLS` | 8 | The most calls to the language model for one question |
| `AGENT_MAX_REWRITES` | 2 | The most times the query is rewritten |
| `WEAVIATE_TENANT` | `development` | Keeps the data of different uses apart |

The agent warns at startup when a limit is too small for the retries it allows.

## Development

```bash
uv run pre-commit install        # install the Git hooks (they run the checks before every commit)
uv run poe check                 # type-check (mypy), lint and format-check (ruff), and run the tests (pytest)
```

The tests use fakes, so they call no external service. Commits follow [Conventional Commits](https://www.conventionalcommits.org/). See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and [SECURITY.md](SECURITY.md) to report a vulnerability.

## Planned

- Following same-site links, respecting `robots.txt`, within a limit of hops, to answer the multi-page questions.
- A check of the answer by a model, to catch claims that the cited excerpt does not support.
- A local web page to try the agent.
- A comparison with the plain pipeline and with a version that indexes the whole site up front, on the same questions.

## License

[MIT](LICENSE)
