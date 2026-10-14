# Chat With Any Website Agent

Agentic RAG built with LangGraph: answers questions about web pages and PDFs with cited sources, follows same-site links when the first page is not enough, and abstains when the source does not cover the answer.

> **Status: planning. There is no working code yet.** Everything under "Planned" describes the goal, not what exists today.

This is the agent version of [Chat With Any Website](https://github.com/ayayousef2000/chat-with-any-website), which answers questions about one page or PDF with a fixed pipeline. This project is separate: it has its own repository, API keys and vector database, and it copies modules from the original instead of importing them.

## Planned

- **Self-correcting retrieval.** When the retrieved passages are not enough, the question is rewritten and searched again instead of answered weakly.
- **Evidence grading.** The agent checks that the evidence is sufficient before it answers, and says so when the source does not cover the question.
- **Bounded navigation.** It follows same-site links, respecting `robots.txt`, within a limit of hops and pages.
- **PDF-aware reading.** It reads specific pages and neighboring pages, and keeps page numbers in its citations.
- **Question decomposition and multi-source answers.** A question with several parts is split, answered, and merged, across a web page and a PDF if needed.
- **Citation verification.** Every claim in an answer is checked against the passage it cites.
- **Hard budgets.** Steps, link hops and model calls per question are capped, so the free plans of the services are not used up.
- **Measured results.** The agent is compared with the plain pipeline, and with a version that indexes the whole site up front, on the same questions. Only measured numbers are published.

## Requirements

Python 3.14 or newer and [uv](https://docs.astral.sh/uv/). API keys for [Cohere](https://dashboard.cohere.com/api-keys), [Weaviate Cloud](https://console.weaviate.cloud) and [Groq](https://console.groq.com/keys) will be needed once the code exists; see `.env.example`.

## Development

```bash
uv sync                          # create the virtual environment and install the locked dependencies
uv run pre-commit install        # install the Git hooks (they run the checks before every commit)
uv run poe check                 # type-check (mypy), lint and format-check (ruff), and run the tests (pytest)
```

Copy `.env.example` to `.env` and fill in your own keys before running anything that calls an external service. Commits follow [Conventional Commits](https://www.conventionalcommits.org/). See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch workflow and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

[MIT](LICENSE)
