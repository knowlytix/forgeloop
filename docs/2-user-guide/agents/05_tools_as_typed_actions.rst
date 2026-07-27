Tools as Typed Actions
======================

This page shows how to define a tool with explicit input and output schemas,
register it so calls are validated against those schemas, and select a tool for a
query with the three router families in :mod:`forgeloop.agents.tools`. A tool here
is a registered capability that carries a pydantic input schema, a pydantic output
schema, a :class:`~forgeloop.agents.tools.RiskLevel` and a stable name, so a
malformed call is rejected at the boundary rather than passed through to the
implementation.

The pieces
----------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Type
     - Role
   * - :class:`~forgeloop.agents.tools.Tool`
     - A registered capability: schemas, risk level, optional ``fn``.
   * - :class:`~forgeloop.agents.tools.RiskLevel`
     - Enforced metadata read by the governance layer.
   * - :class:`~forgeloop.agents.tools.ToolInput`, :class:`~forgeloop.agents.tools.ToolOutput`
     - Marker base classes for input and output schemas.
   * - :class:`~forgeloop.agents.tools.ToolRegistry`
     - Holds tools by name and validates a call against its schema.
   * - :func:`~forgeloop.agents.tools.validate_arguments`
     - Parses an argument mapping against a tool's input schema.
   * - :class:`~forgeloop.agents.tools.Router`
     - Protocol selecting a tool for a query, or ``None``.

Declare a tool with its schemas
-------------------------------

Each :class:`~forgeloop.agents.tools.Tool` names a pydantic model for its
arguments and another for its result. The models are the contract between the
agent and the tool. A :class:`~forgeloop.agents.tools.RiskLevel` is recorded on
the tool so the governance harness can scrutinize a high-risk call more strictly
than a read-only one; a risk not recorded here cannot be enforced later.

.. code-block:: python

   from pydantic import BaseModel
   from forgeloop.agents.tools import RiskLevel, Tool, ToolRegistry

   class SearchIn(BaseModel):
       query: str
   class SearchOut(BaseModel):
       results: list[dict]

   class CalcIn(BaseModel):
       expression: str
   class CalcOut(BaseModel):
       result: float

   class EmailIn(BaseModel):
       to: str
       subject: str
       body: str
   class EmailOut(BaseModel):
       sent: bool

   search_tool = Tool(name='search', description='search documents',
                      input_schema=SearchIn, output_schema=SearchOut, risk=RiskLevel.LOW)
   calc_tool = Tool(name='calculator', description='evaluate basic arithmetic',
                    input_schema=CalcIn, output_schema=CalcOut, risk=RiskLevel.LOW)
   email_tool = Tool(name='send_email', description='send an email to a recipient',
                     input_schema=EmailIn, output_schema=EmailOut, risk=RiskLevel.HIGH)

:class:`~forgeloop.agents.tools.ToolInput` and
:class:`~forgeloop.agents.tools.ToolOutput` are marker base classes for the same
purpose; a schema may subclass them or a plain ``BaseModel`` as above.

Register and validate calls
---------------------------

:meth:`~forgeloop.agents.tools.ToolRegistry.register` refuses a name that is not a
valid identifier or already registered.
:meth:`~forgeloop.agents.tools.ToolRegistry.validate` runs the named tool's input
schema through :func:`~forgeloop.agents.tools.validate_arguments` and returns the
parsed model, raising on a missing or ill-typed field. Extra keys are ignored by
the default pydantic model, so a call carrying a spurious argument still validates
against the declared fields.

.. code-block:: python

   registry = ToolRegistry()
   registry.register(search_tool)
   registry.register(calc_tool)
   registry.register(email_tool)
   print('registered:', [t.name for t in registry.all()])

   try:
       registry.validate('calculator', {'wrong': 'no expression field'})
   except Exception:
       print('missing field rejected')

Route a query to a tool
-----------------------

Routing selects a tool for a natural-language query. Three strategies conform to
the :class:`~forgeloop.agents.tools.Router` protocol, which is
``runtime_checkable``, so a router is any object with a
``route(query, registry) -> Tool | None`` method.
:class:`~forgeloop.agents.tools.RuleRouter` maps keywords to tool names and is
transparent but brittle to phrasing.
:class:`~forgeloop.agents.tools.EmbeddingRouter` ranks tool descriptions by cosine
similarity to the query and returns ``None`` below its ``threshold``.
:class:`~forgeloop.agents.tools.LMRouter` asks a language model to name the
matching tool and returns ``None`` when the reply is ``NONE`` or names a tool the
registry does not hold.

.. code-block:: python

   from forgeloop.agents.tools import EmbeddingRouter, LMRouter, RuleRouter

   rule = RuleRouter({'search': 'search', 'calculate': 'calculator',
                      'compute': 'calculator', 'email': 'send_email'})
   emb = EmbeddingRouter(embedder, threshold=0.0)   # embedder: embed(list[str]) -> list[list[float]]
   lm = LMRouter(language_model)                    # language_model: complete(prompt) -> str

   for name, router in [('RuleRouter', rule), ('EmbeddingRouter', emb), ('LMRouter', lm)]:
       picked = router.route('please search the policy docs', registry)
       print(name, '->', picked.name if picked else None)

A benchmark drives every router over the same labelled queries and compares the
selected tool against the expected one, which is how a router is chosen for a
domain rather than assumed. Because the language model can name a tool that does
not exist, the registry catches the error by returning ``None`` rather than
dispatching a call to a missing tool.

How the layers combine
----------------------

The three routers trade transparency against generalization. Keyword rules are
auditable and cheap but fail on unseen phrasing; embedding similarity generalizes
across phrasing but needs a calibrated cutoff; language-model routing reads intent
flexibly but adds latency and can hallucinate a name. A production stack layers
them and treats which layer fires as a parameter tuned against a real benchmark.
Whichever router selects a tool, the registry still validates the arguments before
the call proceeds, and the governed executor in :doc:`06_safe_tool_execution`
screens it before it runs.

See also
--------

- :doc:`/3-api-reference/modules/tools/index` — full signatures for :class:`~forgeloop.agents.tools.Tool`, :class:`~forgeloop.agents.tools.ToolRegistry` and the routers.
- :doc:`06_safe_tool_execution` — the gate stack that screens a validated call before execution.
- :doc:`/4-notebook-examples/agents/index` — the routers compared on a labelled benchmark.
