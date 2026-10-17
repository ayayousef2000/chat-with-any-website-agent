"""Tests of the state of the agent graph."""

from langgraph.graph import END, START, StateGraph

from chat_with_any_website_agent.state import AgentState, new_state


def test_a_new_state_starts_with_the_question_as_the_query_and_the_counters_at_zero() -> None:
    state = new_state("What does pass do?", ["https://docs.example/control"])

    assert state == {
        "question": "What does pass do?",
        "urls": ["https://docs.example/control"],
        "query": "What does pass do?",
        "rewrites": 0,
        "documents": [],
        "abstained": False,
        "steps": [],
        "llm_calls": 0,
    }


def test_a_new_state_searches_every_source_when_none_are_given() -> None:
    assert new_state("Q?")["urls"] is None


def test_the_steps_of_the_nodes_are_added_up_while_the_other_keys_are_replaced() -> None:
    def search(state: AgentState) -> AgentState:
        return {"steps": ["searched"], "rewrites": 1, "query": "first"}

    def answer(state: AgentState) -> AgentState:
        return {"steps": ["answered"], "rewrites": 2}

    graph = StateGraph(AgentState)
    graph.add_node("search", search)
    graph.add_node("answer", answer)
    graph.add_edge(START, "search")
    graph.add_edge("search", "answer")
    graph.add_edge("answer", END)

    result = graph.compile().invoke(new_state("Q?"))

    assert result["steps"] == ["searched", "answered"]
    assert result["rewrites"] == 2
    assert result["query"] == "first"
    assert result["question"] == "Q?"
