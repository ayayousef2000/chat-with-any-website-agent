"""The README describes the code as it is."""

import re
from pathlib import Path

from chat_with_any_website_agent.chat_model import GroqChatModel
from chat_with_any_website_agent.config import Settings
from chat_with_any_website_agent.graph import AgentLimits, SourceAgent
from chat_with_any_website_agent.retriever import SourceRetriever
from tests.test_graph import FIRST_SEARCH, LIMITS, FakeEmbedder, FakeReranker, FakeStore, ScriptedClient

README = Path(__file__).resolve().parent.parent / "README.md"


def _agent(limits: AgentLimits = LIMITS) -> SourceAgent:
    retriever = SourceRetriever(FakeStore([FIRST_SEARCH]), FakeEmbedder(), FakeReranker())
    return SourceAgent(retriever, GroqChatModel(ScriptedClient()), GroqChatModel(ScriptedClient()), limits)


def test_the_diagram_in_the_readme_is_the_graph_of_the_agent() -> None:
    blocks = re.findall(r"```mermaid\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.DOTALL)

    assert blocks, "The README needs the diagram of the graph."
    assert blocks[0].strip() == _agent().diagram().strip(), (
        "The graph changed: paste the new `SourceAgent.diagram()` into the README."
    )


def test_the_diagram_does_not_depend_on_the_limits() -> None:
    assert _agent(AgentLimits(max_steps=24, max_llm_calls=12, max_rewrites=2, max_expansions=1)).diagram() == (
        _agent().diagram()
    )


def test_every_setting_of_the_agent_that_the_readme_names_exists() -> None:
    named = set(re.findall(r"`(AGENT_[A-Z_]+)`", README.read_text(encoding="utf-8")))

    assert named <= {name.upper() for name in Settings.model_fields}
