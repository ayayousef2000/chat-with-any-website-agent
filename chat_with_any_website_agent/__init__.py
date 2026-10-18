"""Chat With Any Website Agent: a LangGraph workflow for grounded, cited answers about web pages and PDFs."""

import warnings

__version__ = "0.1.0"

# Two libraries that this project uses call a Python function that will be removed in Python 3.16, and print a notice
# about it every time they are imported. The notice is about their code and there is nothing to change here, so
# only that notice from those two libraries is hidden. Every other warning is left as it is.
warnings.filterwarnings(
    "ignore",
    message=r"'asyncio\.iscoroutinefunction' is deprecated",
    category=DeprecationWarning,
    module=r"(cohere|langsmith)(\.|$)",
)
