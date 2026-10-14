# Contributing

## Branches

- `main` is the released state. Nothing is pushed to it directly, and pull requests into it come only from `develop`.
- `develop` is the integration branch. Nothing is pushed to it directly either.
- All work happens on short-lived branches created from `develop`, named `<type>/<short-description>`, for example `feat/grade-evidence` or `fix/url-validation`.

## Workflow

1. Create a branch from `develop` and make your change.
2. Run `uv run poe check` (the Git hooks run it on every commit; see [Git hooks](#git-hooks) to install them).
3. Push the branch and open a pull request **into `develop`**. The title must follow
   [Conventional Commits](https://www.conventionalcommits.org/), for example `fix(fetching): reject invalid ports`.
4. When the required checks pass and the review is done, the pull request is merged into `develop` with a **merge commit**.
   The individual commits of the branch are kept, so each one must follow Conventional Commits (the Git hook checks this).
5. To release, open a pull request from `develop` into `main`, and merge it with a **merge commit**.

## Development commands

One command runs everything that must pass before a commit: type-check (mypy, strict), lint (Ruff), format check (Ruff) and tests (pytest with coverage):

```bash
uv run poe check
```

| Command | What it does |
|---|---|
| `uv run poe check` | Types, lint, format check and tests |
| `uv run poe fix` | Apply Ruff's automatic lint fixes and formatting |
| `uv run poe types` / `lint` / `format` / `test` | Run a single step |

## Git hooks

Install the hooks once after cloning:

```bash
uv run pre-commit install
```

Before every commit the hooks then run:

- file hygiene checks (merge conflict markers, large files, case conflicts, private keys, valid YAML/TOML/JSON, line endings)
- a secret scan of the staged changes (gitleaks), and a block on committing `.env` files
- validation of the workflows, Dependabot configuration and issue forms against their schemas, and a security audit of the workflows (zizmor)
- a check that `uv.lock` matches `pyproject.toml`, and a vulnerability audit of the locked dependencies (when either file changes)
- a compile check and a package build (the same as CI)
- `uv run poe check`

## Commit messages

Conventional Commits, for example `feat: add evidence grading` or `fix(fetching): reject invalid ports`. Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

## Conventions

- Public modules, classes and functions have docstrings (Google style), enforced by Ruff.
- Tests and CI never call the external services (Cohere, Weaviate, Groq) and need no keys: logic that does not need the network lives in plain functions that are unit-tested in `tests/`, and the services are replaced by fakes.
- Anything that does call an external service, such as an evaluation run, is started by hand, never by CI, and must stop at a call limit.
- Secrets live only in `.env`, which is git-ignored.

## Security

Never commit secrets. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).
