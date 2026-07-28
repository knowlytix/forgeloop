import hashlib

from pydantic import BaseModel

from agentlab.tools import (
    EmbeddingRouter,
    LMRouter,
    RiskLevel,
    Router,
    RuleRouter,
    Tool,
    ToolRegistry,
)


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


class MockEmbedder:
    dim = 8

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            v = [b / 255.0 for b in h[: self.dim]]
            out.append(v)
        return out


class MockLM:
    def __init__(self, response: str = "search") -> None:
        self._response = response

    def complete(self, prompt: str, **kwargs) -> str:
        return self._response

    def token_count(self, text: str) -> int:
        return len(text.split())


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.register(Tool(name="search", description="search documents",
                    input_schema=_In, output_schema=_Out, risk=RiskLevel.LOW))
    r.register(Tool(name="calc", description="evaluate arithmetic",
                    input_schema=_In, output_schema=_Out, risk=RiskLevel.LOW))
    r.register(Tool(name="send_email", description="send an email",
                    input_schema=_In, output_schema=_Out, risk=RiskLevel.HIGH))
    return r


def test_rule_router_routes():
    r = _registry()
    router = RuleRouter({"search": "search", "calculate": "calc", "email": "send_email"})
    assert router.route("please search the docs", r).name == "search"
    assert router.route("calculate 2+2", r).name == "calc"
    assert router.route("send an email to bob", r).name == "send_email"
    assert router.route("hello", r) is None


def test_embedding_router_returns_a_tool_at_zero_threshold():
    r = _registry()
    router = EmbeddingRouter(MockEmbedder(), threshold=0.0)
    out = router.route("search documents", r)
    assert out is not None


def test_embedding_router_none_when_above_threshold():
    r = _registry()
    router = EmbeddingRouter(MockEmbedder(), threshold=1.1)
    assert router.route("anything", r) is None


def test_lm_router_picks_named_tool():
    r = _registry()
    router = LMRouter(MockLM(response="search"))
    assert router.route("anything", r).name == "search"


def test_lm_router_returns_none_on_unknown():
    r = _registry()
    router = LMRouter(MockLM(response="not-a-tool"))
    assert router.route("anything", r) is None


def test_lm_router_returns_none_on_NONE():
    r = _registry()
    router = LMRouter(MockLM(response="NONE"))
    assert router.route("anything", r) is None


def test_router_protocol_runtime_check():
    assert isinstance(RuleRouter({}), Router)
    assert isinstance(EmbeddingRouter(MockEmbedder()), Router)
    assert isinstance(LMRouter(MockLM()), Router)
