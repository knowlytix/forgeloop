Recording Reasoning as Typed Entries
====================================

This page shows how to record an agent's reasoning as typed, inspectable entries
on a :class:`~forgeloop.agents.reasoning.Scratchpad` rather than as a free-form
paragraph, how the two construction-time invariants reject a malformed entry, and
where an independent verifier attaches above the structural check. The scratchpad
guarantees that a claim names its evidence; it does not by itself decide whether
the evidence supports the claim, which is the boundary the later part of the
section builds on.

The pieces
----------

Four types in :mod:`forgeloop.agents.reasoning` carry the whole surface.

.. list-table::
   :header-rows: 1

   * - Type
     - Role
   * - :class:`~forgeloop.agents.reasoning.TrustLevel`
     - Ordered belief an entry warrants: ``LOW`` to ``HIGHEST``.
   * - :class:`~forgeloop.agents.reasoning.EntryType`
     - Kind of entry: claim, assumption, observation or question.
   * - :class:`~forgeloop.agents.reasoning.Entry`
     - A frozen, validated record: kind, text, evidence, source, trust.
   * - :class:`~forgeloop.agents.reasoning.Scratchpad`
     - Append-only collection with query and assertion helpers.

The free-form baseline
----------------------

A model partway through a case produces a paragraph in which a fact, an inference
and a qualification are all the same kind of object: text the model wrote. The
rest of the system has no way to tell them apart.

.. code-block:: python

   free_form = (
       "The overdraft fee is $35. It applies once per occurrence. "
       "I think the customer was charged twice in the same day. "
       "Bank policy probably allows a one-time goodwill reversal."
   )

The paragraph helps the model write the next sentence and supports no check by
anything downstream.

Record the same content as typed entries
-----------------------------------------

The same content becomes four entries, each with a kind and, where the kind
requires it, a source or an evidence pointer.
:meth:`~forgeloop.agents.reasoning.Scratchpad.render_table` returns row dicts, or
an aligned text table when ``as_string=True``.

.. code-block:: python

   from forgeloop.agents.reasoning import Scratchpad, TrustLevel

   pad = Scratchpad()
   pad.add_claim("The overdraft fee is $35", evidence="overdraft.txt", trust=TrustLevel.HIGH)
   pad.add_claim("It applies once per occurrence", evidence="overdraft.txt", trust=TrustLevel.HIGH)
   pad.add_assumption("The customer was charged twice in the same day")
   pad.add_observation("Customer service log shows two charges on 2026-05-01", source="ticket-1042")
   pad.add_question("Was a goodwill reversal already issued this year?")

   print(pad.render_table(as_string=True))

:meth:`~forgeloop.agents.reasoning.Scratchpad.add_assumption` and
:meth:`~forgeloop.agents.reasoning.Scratchpad.add_question` fix the trust at
``LOW`` and take no evidence, so an assumption cannot be recorded as though it
were established.

Query and assert over the entries
----------------------------------

Because the entries are typed, the checks are ordinary code.
:meth:`~forgeloop.agents.reasoning.Scratchpad.by_type` filters by
:class:`~forgeloop.agents.reasoning.EntryType`,
:meth:`~forgeloop.agents.reasoning.Scratchpad.unsupported_claims` returns claims
with no evidence pointer, and
:meth:`~forgeloop.agents.reasoning.Scratchpad.assert_all_claims_have_evidence`
raises ``AssertionError`` when any claim lacks one.

.. code-block:: python

   from forgeloop.agents.reasoning import EntryType

   print("claims:        ", len(pad.by_type(EntryType.CLAIM)))
   print("observations:  ", len(pad.by_type(EntryType.OBSERVATION)))
   print("unsupported claims:", [e.text for e in pad.unsupported_claims()])

   pad.assert_all_claims_have_evidence()

The two invariants
------------------

An :class:`~forgeloop.agents.reasoning.Entry` is frozen and validates on
construction, so the check runs when the entry is created rather than in a later
corrective pass. Two rules hold: an observation must name a source, and any entry
whose trust is above ``LOW`` must carry an evidence pointer. Either violation
raises ``pydantic.ValidationError``.

.. code-block:: python

   from forgeloop.agents.reasoning import Entry, EntryType, TrustLevel
   from pydantic import ValidationError

   try:
       Entry(kind=EntryType.OBSERVATION, text="x")
   except ValidationError as e:
       print("observation rejected:", e.errors()[0]["msg"])

   try:
       Entry(kind=EntryType.CLAIM, text="x", trust=TrustLevel.HIGH)
   except ValidationError as e:
       print("high-trust no-evidence rejected:", e.errors()[0]["msg"])

Checking at construction matters for the audit log, which hashes each entry: a
late edit would attest to a process that did not run.

Above the structural check
---------------------------

An evidence pointer is a promise, not a check, so structural validity does not
reach the top of the trust hierarchy. An independently checked claim requires a
verifier outside the model whose verdict is reproducible. In the example notebook
that verifier is a GMS store read through
:class:`~forgeloop.agents.gms_backend.GMSMemory`:
:meth:`~forgeloop.agents.gms_backend.GMSMemory.lookup_enm` reads a stored value
back exactly, and :meth:`~forgeloop.agents.gms_backend.GMSMemory.score_triple`
returns a distance that is small when a claim agrees with the store and large
when it conflicts. The following requires a GPU-backed store and ``knowlytix`` and
is reproduced from the notebook rather than executed on this page.

.. code-block:: python

   # separates a correct tail (35.0) from a wrong one (45.0, the wire fee)
   for tail, label in [("35.0", "claimed"), ("45.0", "wrong (wire fee)")]:
       s = mem.score_triple("overdraft", "has_fee_amount", tail)
       print(f"  (overdraft, has_fee_amount, {tail}) -> {s:.3f}  {label}")

The scratchpad establishes that a claim cites something; the verifier decides
whether the citation holds. The model proposes and the substrate approves, which
is the pattern the governance and evaluation chapters reuse.

See also
--------

- :doc:`/3-api-reference/modules/reasoning/index` — full signatures for these types.
- :doc:`04_tasks_state_actions` — where scratchpad entries are stored as opaque
  dicts on :class:`~forgeloop.agents.core.AgentState`.
- :doc:`12_runtime_governance` — the gate that acts on the approved substrate.
- :doc:`/4-notebook-examples/agents/index` — the same scratchpad run end to end.
