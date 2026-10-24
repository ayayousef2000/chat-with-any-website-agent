# Chat With Any Website Agent

An agentic RAG workflow built with [LangGraph](https://langchain-ai.github.io/langgraph/) and LangChain. It answers questions about a web page or a PDF file with cited sources. It grades what it finds, follows links to other pages of the same site when the evidence is not enough, checks its own citations, and says that the source does not cover the question instead of inventing an answer.

This is the agent version of [Chat With Any Website](https://github.com/ayayousef2000/chat-with-any-website), which answers questions about one page or PDF with a fixed pipeline. It is a separate project with its own repository, keys and vector database.

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
- **Grade.** A small model decides whether the excerpts are enough to answer, and names the ones the answer needs, so that the larger model reads only those.
- **Follow links.** When they are not, a small model picks at most two links from a numbered list of the links on the pages read so far. Only links to the same site are offered, and only pages that its `robots.txt` allows are opened. The new pages are searched together with the first one.
- **Rewrite and retry.** If that is not enough either, the query is rewritten and searched again, within a limit.
- **Generate and verify.** A larger model writes the answer and cites the excerpts as `[1]`, `[2]`. Plain code then checks that every cited excerpt exists, that each cited sentence shares its content words with it and that every number in the answer is written in the excerpts. A rejected answer is corrected once.
- **Abstain.** When the source does not cover the question, the answer says so and gives the reason.

Every question runs within hard limits on steps, model calls, rewrites and pages. PDF files are read page by page, and a question can name a page ("What does page 5 say about the heads?").

## Results

The agent was evaluated on 55 questions over public sources: pages of the Python tutorial, the paper "Attention Is All You Need" (PDF), the Pro Git book and an Arabic Wikipedia article. A separate model judges each answer against a reference answer and the excerpts the agent used. The set is small, so treat the numbers as a first measurement and not as a benchmark.

| Category | Questions | Correct |
|---|---|---|
| Single page | 4 | 3 |
| Two sources given | 4 | 4 |
| PDF, facts from a given page | 6 | 6 |
| Arabic | 2 | 2 |
| Text that the chunker cuts apart (an example or a list across two chunks) | 16 | 16 |
| Unanswerable (the right answer is that the source does not cover it) | 9 | 8 |
| Multi-page (the answer is on a page linked from the start page; 6 on the Python tutorial, 8 on the Pro Git book with 148 links on its index page) | 14 | 14 |
| **All** | 55 | **53 (96%)** |

- Both misses were marked wrong by the judge model although the answers are right: the judge was too strict. Read by hand, all 55 are right.
- Without link following, the same agent answers none of the multi-page questions.
- A question costs about 4,600 Groq tokens on average, not counting the judge.
- Known limit: the citation check compares words and numbers, so an answer can still add a detail that the source does not state.

## Requirements

Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). You need your own API keys for [Cohere](https://dashboard.cohere.com/api-keys), [Weaviate Cloud](https://console.weaviate.cloud) and [Groq](https://console.groq.com/keys). The free plans are enough.

## Setup

```bash
uv sync                      # create the virtual environment and install the locked dependencies
cp .env.example .env         # then fill in your own keys; the file is never committed
uv run agent-smoke --prepare # check the services and create or update the Weaviate collection and tenants
```

## Usage

Ask a question about a page or a PDF file. The source is stored the first time it is loaded, and a stored page is loaded again when it changes:

```bash
uv run agent-ask "https://docs.python.org/3/tutorial/index.html" "What does the pass statement do?"
```

It prints the answer, the sources it cites, the pages it opened by following links and what the question cost. Useful options: `--also URL` searches a further source as well, `--no-follow` does not follow links, `--refresh` downloads the source again, and `--tenant NAME` uses another Weaviate tenant.

Or try it in a browser:

```bash
uv run agent-web
```

Then open <http://127.0.0.1:8000>, paste the address of a page or PDF file and ask a question. Click a citation to see the excerpt it comes from. The server is for one person on one computer: it listens on `127.0.0.1` only.

Evaluate the agent on a question set:

```bash
uv run agent-eval validate eval/questions.json     # check the file; no service is used
WEAVIATE_TENANT=eval uv run agent-eval run eval/questions.json --system agent --out eval/results.json
uv run agent-eval summary eval/results.json        # read saved results; no service is used
```

A run stops cleanly at the limits of calls and tokens in the settings, and `--resume` continues it. `--system` also accepts `agent-single` (no links), `plain` and `crawl`, to compare the agent with simpler systems.

## Docker

```bash
docker build --tag chat-with-any-website-agent .
docker run --rm --publish 8000:8000 --env-file .env chat-with-any-website-agent
```

Then open <http://localhost:8000>.

## Deploy on Render

[`render.yaml`](render.yaml) creates two free web services: `production` (the `main` branch, Weaviate tenant `production`) and `staging` (the `develop` branch, tenant `staging`).

1. In Render choose **New**, then **Blueprint**, and select this repository and the `main` branch.
2. For each service, enter the secrets Render asks for: `COHERE_API_KEY`, `WEAVIATE_URL`, `WEAVIATE_API_KEY` and `GROQ_API_KEY`.
3. Wait for the build. The page is at the address of each service.

A free service sleeps after 15 minutes without traffic and takes about a minute to wake. If the page answers "This server answers only to pages that it serves itself", set `WEB_ALLOWED_HOSTS` to the address of the service.

## Configuration

Settings are read from the environment or from `.env`; [`.env.example`](.env.example) lists all of them with their defaults and a comment for each. The ones that shape the agent:

| Setting | Default | Meaning |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-120b` | The model that writes the answer |
| `GROQ_GRADE_MODEL` | `openai/gpt-oss-20b` | The model that grades the excerpts and rewrites queries |
| `AGENT_MAX_STEPS` | 24 | The most graph steps for one question |
| `AGENT_MAX_LLM_CALLS` | 12 | The most calls to the language model for one question |
| `AGENT_MAX_REWRITES` | 2 | The most times the query is rewritten |
| `AGENT_MAX_LINK_HOPS` | 3 | The most times the agent chooses links to open; 0 keeps it on the pages it was given |
| `AGENT_MAX_PAGES` | 5 | The most pages the agent loads by following links |
| `AGENT_MAX_EXPANSIONS` | 0 | The most times the excerpts are widened with the text next to the best ones (0 is off; with 1, use `AGENT_MAX_STEPS=26` and `AGENT_MAX_LLM_CALLS=13`) |
| `WEAVIATE_TENANT` | `development` | Keeps the data of different uses apart |

## Development

```bash
uv run pre-commit install        # install the Git hooks (they run the checks before every commit)
uv run poe check                 # type-check (mypy), lint and format-check (ruff), and run the tests (pytest)
```

The tests use fakes, so they call no external service. Commits follow [Conventional Commits](https://www.conventionalcommits.org/). See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

[MIT](LICENSE)
