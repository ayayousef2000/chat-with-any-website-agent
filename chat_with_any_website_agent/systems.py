"""The systems that the evaluation can score, by name.

The agent is registered here, with the systems it is compared to: the same agent without links, a plain pipeline of one
search and one answer, and that pipeline over a site that was crawled first.
"""

from typing import TYPE_CHECKING

from chat_with_any_website_agent.agent_system import create_agent_system, create_single_page_agent_system
from chat_with_any_website_agent.baselines import create_crawl_system, create_plain_system

if TYPE_CHECKING:
    from collections.abc import Callable

    from chat_with_any_website_agent.config import Settings
    from chat_with_any_website_agent.evaluation import System
    from chat_with_any_website_agent.usage import UsageCounter

# A factory builds a system from the settings and the counter that limits the run.
type SystemFactory = Callable[[Settings, UsageCounter], System]

SYSTEMS: dict[str, SystemFactory] = {
    "agent": create_agent_system,
    "agent-single": create_single_page_agent_system,
    "plain": create_plain_system,
    "crawl": create_crawl_system,
}
