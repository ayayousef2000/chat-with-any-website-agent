"""The systems that the evaluation can score, by name.

The plain pipeline, the version that crawls the whole site first, the original app and the agent are added here as
they are built.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.evaluation import System
    from chat_with_any_website_agent.usage import UsageCounter

# A factory builds a system from the settings and the counter that limits the run.
type SystemFactory = Callable[[Settings, UsageCounter], System]

SYSTEMS: dict[str, SystemFactory] = {}
