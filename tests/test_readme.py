"""The README describes the code as it is."""

import re
from pathlib import Path

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.graph import SourceAgent
from chat_with_any_website_agent.retriever import SourceRetriever
from tests.test_graph import FIRST_SEARCH, LIMITS, FakeEmbedder, FakeReranker, FakeStore, ScriptedClient

README = Path(__file__).resolve().parent.parent / "README.md"
FLOW_ENDS = {"question", "answer"}  # the two words of the flow that are not steps of the graph


def _agent() -> SourceAgent:
    retriever = SourceRetriever(FakeStore([FIRST_SEARCH]), FakeEmbedder(), FakeReranker())
    return SourceAgent(retriever, GroqChatModel(ScriptedClient()), GroqChatModel(ScriptedClient()), LIMITS)


def test_every_step_in_the_flow_of_the_readme_is_a_step_of_the_graph() -> None:
    flow = re.search(r"^question → .*$", README.read_text(encoding="utf-8"), flags=re.MULTILINE)

    assert flow, "The README needs the flow of the graph."
    steps = {step.strip() for step in flow.group().split("→")} - FLOW_ENDS
    assert steps <= set(_agent()._graph.get_graph().nodes)


def test_every_setting_of_the_agent_that_the_readme_names_exists() -> None:
    named = set(re.findall(r"`(AGENT_[A-Z_]+)`", README.read_text(encoding="utf-8")))

    assert named <= {name.upper() for name in Settings.model_fields}
