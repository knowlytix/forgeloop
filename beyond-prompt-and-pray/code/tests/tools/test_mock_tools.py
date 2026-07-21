from agentlab.core import ToolCall
from agentlab.tools import GovernedToolExecutor, ToolRegistry
from agentlab.tools.mock_tools import calculator, register_all, search_documents, send_email


def _with_tool(t):
    r = ToolRegistry()
    r.register(t)
    return r


def test_register_all():
    r = ToolRegistry()
    register_all(r)
    assert len(r.all()) == 3


def test_search_documents_finds_keyword():
    ex = GovernedToolExecutor(_with_tool(search_documents))
    result = ex.execute(ToolCall(tool_name="search_documents", arguments={"query": "overdraft fees"}))
    assert result.success
    assert len(result.output["results"]) >= 1


def test_search_documents_empty():
    ex = GovernedToolExecutor(_with_tool(search_documents))
    result = ex.execute(ToolCall(tool_name="search_documents", arguments={"query": "unrelated"}))
    assert result.success
    assert result.output["results"] == []


def test_calculator_basic():
    ex = GovernedToolExecutor(_with_tool(calculator))
    result = ex.execute(ToolCall(tool_name="calculator", arguments={"expression": "2+2"}))
    assert result.success
    assert result.output["result"] == 4.0


def test_calculator_rejects_unsafe():
    ex = GovernedToolExecutor(_with_tool(calculator))
    result = ex.execute(
        ToolCall(tool_name="calculator", arguments={"expression": "__import__('os')"})
    )
    assert not result.success


def test_send_email():
    ex = GovernedToolExecutor(_with_tool(send_email))
    result = ex.execute(
        ToolCall(tool_name="send_email", arguments={"to": "a@b", "subject": "s", "body": "b"})
    )
    assert result.success
    assert result.output["success"] is True
