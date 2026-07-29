forgeloop
=========

A companion library for the *Beyond Prompt / Ship / Chunk and Pray* trilogy:
one installable package for building governed agents, testing them by design,
and grounding them in geometric memory.

The package is organized into three subpackages, one per book:

- :mod:`forgeloop.agents` — building governed agents (*Beyond Prompt and Pray*):
  the agent loop, typed tools, runtime governance, memory, planning, evaluation
  and a capstone banking agent.
- :mod:`forgeloop.testing` — testing agentic systems by design (*Beyond Ship and
  Pray*): catalog-driven test suites, design-of-experiments composition and
  training-data emission.
- :mod:`forgeloop.rag` — governed retrieval (*Beyond Chunk and Pray*):
  triple-mediated retrieval over a geometric memory store, grounded synthesis
  and calibrated abstention.

The three share a common substrate: the "LLM proposes, GMS approves" pattern, in
which a language model proposes structured actions and a geometric model store
(the licensed ``knowlytix`` backend) approves or rejects them against a
calibrated operating point.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   /2-user-guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   /3-api-reference/modules

.. toctree::
   :maxdepth: 2
   :caption: Examples

   /examples

.. toctree::
   :maxdepth: 1
   :caption: Installation

   /2-user-guide/installation
