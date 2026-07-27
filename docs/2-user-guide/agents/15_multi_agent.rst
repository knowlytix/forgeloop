Multi-Agent: When, How and When Not To
======================================

This page shows how to build a supervisor-worker system from
:mod:`forgeloop.agents.multiagent`, where each worker carries its own governance
harness and workers communicate only through typed messages. The pattern is
worth reaching for when the workers genuinely differ in capability, policy or
audit boundary. When a single well-engineered agent could do the job it is the
cheaper and more reliable choice, because splitting adds latency, a larger
failure surface and more trajectories to evaluate.

The pieces
----------

Four types make up the pattern. A worker wraps a
:class:`~forgeloop.agents.governance.harness.GovernanceHarness`, so each worker
brings its own audit chain, policy set and tool registry.

=====================================================================  ==================================================
Type                                                                   Role
=====================================================================  ==================================================
:class:`~forgeloop.agents.multiagent.message.AgentMessage`             A frozen message: sender, receiver, ``message_type`` and a structured payload.
:class:`~forgeloop.agents.multiagent.message.MessageBus`               An append-only log of every message, queryable by sender, receiver or thread.
:class:`~forgeloop.agents.multiagent.worker.Worker`                    A named agent that handles a ``delegate`` message by running its harness.
:class:`~forgeloop.agents.multiagent.supervisor.Supervisor`            Routes a subtask to a named worker and records both messages on the bus.
=====================================================================  ==================================================

Build two workers, each with its own harness
---------------------------------------------

A :class:`~forgeloop.agents.multiagent.worker.Worker` is constructed with a name,
a capability label and a
:class:`~forgeloop.agents.governance.harness.GovernanceHarness`. Building the
harness per worker is what gives the governance-per-worker guarantee: the two
workers below share no audit chain, so a question about one is answered entirely
from its own chain.

.. code-block:: python

   from forgeloop.agents.core import BaseAgent, Finish
   from forgeloop.agents.governance import GovernanceHarness
   from forgeloop.agents.multiagent import AgentMessage, MessageBus, Supervisor, Worker
   from forgeloop.agents.tools import GovernedToolExecutor, ToolRegistry

   class FixedFinish(BaseAgent):
       def __init__(self, answer):
           self.answer = answer
       def propose_action(self, state):
           return Finish(output=self.answer)

   def make_worker(name, capability, answer):
       executor = GovernedToolExecutor(ToolRegistry(), gates=[])
       harness = GovernanceHarness(FixedFinish(answer), executor)
       return Worker(name=name, capability=capability, harness=harness)

   classifier = make_worker("classifier", "classify_complaint", "complaint")
   drafter    = make_worker("drafter", "draft_response", {"text": "thank you for reaching out"})

Route by name over the bus
--------------------------

A :class:`~forgeloop.agents.multiagent.supervisor.Supervisor` holds the workers by
name and a shared :class:`~forgeloop.agents.multiagent.message.MessageBus`.
:meth:`~forgeloop.agents.multiagent.supervisor.Supervisor.delegate` builds a
``delegate`` message, records it, runs the named worker's
:meth:`~forgeloop.agents.multiagent.worker.Worker.handle` and records the response.
The worker turns the payload into a ``TaskSpec``, runs its harness and returns the
status, final output and step count. Every exchange is a typed
:class:`~forgeloop.agents.multiagent.message.AgentMessage`, so the interaction is
replayable rather than free-form prose that drifts.

.. code-block:: python

   bus = MessageBus()
   sup = Supervisor(name="sup", workers=[classifier, drafter], bus=bus)

   r1 = sup.delegate("classifier", goal="classify this message")
   r2 = sup.delegate("drafter", goal="draft a response")
   print("classifier response:", r1.payload)
   print("drafter response:   ", r2.payload)
   print(f"bus has {len(bus)} messages")

Independent audit chains
------------------------

Because each worker owns its harness, each owns a separate hash-chained audit
log. The chains verify independently and their heads differ, so a regulator's
question about the drafter never reaches into the classifier's record.

.. code-block:: python

   print("classifier audit verifies:", classifier.harness.audit.verify())
   print("drafter audit verifies:   ", drafter.harness.audit.verify())
   print("classifier chain head[:8]:", classifier.harness.audit.head()[:8])
   print("drafter chain head[:8]:   ", drafter.harness.audit.head()[:8])

Resolve conflict by evidence, not by vote
-----------------------------------------

When two workers on the same base model disagree, a majority vote can let two
correlated errors outvote one correct dissent. The supervisor instead scores each
claim against the shared substrate through
:meth:`~forgeloop.agents.gms_backend.memory.GMSMemory.score_triple` and keeps the
better-grounded claim; a lower geodesic score is more grounded. The substrate,
not a count of agents, arbitrates.

.. code-block:: python

   import torch
   from knowlytix.knowledge.query import DocGMSConfig, GMSExpertStore
   from forgeloop.agents.gms_backend import GMSMemory

   store = GMSExpertStore(
       DocGMSConfig(store_path="data/gms_banking_store"),
       device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
   )
   store.load()
   mem = GMSMemory(store)

   claims = {"worker_A": ("overdraft", "has_fee_amount", "35.0"),
             "worker_B": ("overdraft", "has_fee_amount", "45.0")}
   scored = {w: mem.score_triple(*c) for w, c in claims.items()}
   winner = min(scored, key=scored.get)     # lower geodesic == more grounded

What a split does and does not buy
----------------------------------

The split yields parallel specialization and clean governance boundaries: a
distinct policy set, tool registry and audit chain per worker. It does not yield
better reasoning, and the arbitration above is not a consensus mechanism. The
decision to split reduces to three questions asked in order. Can a single
well-engineered agent do this. Do the workers differ in capability, policy or
audit boundary, or are they one agent in two guises. Are the added latency, the
larger failure surface and the extra model calls justified by what the
specialization returns.

See also
--------

- :doc:`/3-api-reference/modules/multiagent/index` — full signatures for the four types.
- :doc:`/3-api-reference/modules/governance/index` — the
  :class:`~forgeloop.agents.governance.harness.GovernanceHarness` each worker carries.
- :doc:`/3-api-reference/modules/gms_backend/index` — the
  :class:`~forgeloop.agents.gms_backend.memory.GMSMemory` used for evidence-based arbitration.
- :doc:`/4-notebook-examples/agents/index` — the supervisor-worker system run end to end.
