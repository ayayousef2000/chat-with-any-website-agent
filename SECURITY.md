# Security policy

## Supported versions

Security fixes are applied to the latest release and to the `main` branch.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report them privately through GitHub: open the **Security** tab of this repository and choose
**Report a vulnerability**. Include a description, steps to reproduce, and the affected version or commit.

You can expect an acknowledgement within a few days. Fixes are released as soon as they are ready, and you
will be credited if you wish.

## Handling secrets

This project needs API keys for Cohere, Weaviate and Groq. They belong in a local `.env` file, which is
git-ignored. If you believe a key was exposed, revoke it with the provider immediately.
