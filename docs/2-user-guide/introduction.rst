Introduction
============

forgeloop is the companion library for the *Beyond Prompt and Pray*, *Beyond
Ship and Pray* and *Beyond Chunk and Pray* books. Each book develops one part of
building a trustworthy agentic system, and each part is a subpackage:

- :mod:`forgeloop.agents` builds a governed agent: an inspectable loop over typed
  actions, tools that are admitted by governance gates before they run, memory,
  planning, trajectory evaluation and a capstone banking-complaint agent.
- :mod:`forgeloop.testing` constructs test suites by design of experiments,
  separating what a case asks and what is true from how it is presented, so that
  a change in outcome is attributable to a specific presentation factor.
- :mod:`forgeloop.rag` retrieves through a geometric memory store: the corpus is
  turned into subject-relation-object triples, retrieval runs over bound query
  triples, and answers are synthesized only from retrieved evidence with
  calibrated abstention.

The books
---------

forgeloop is the companion code for a series of published books. The books
develop the theory and method; this library provides the runnable
implementation.

.. list-table::
   :class: book-covers
   :widths: 1 1 1 1

   * - .. image:: covers/kg.png
          :width: 150px
          :target: https://leanpub.com/KG-Embedding
          :alt: Knowledge Graph Embeddings as Geometric Operators
     - .. image:: covers/bpp.png
          :width: 150px
          :target: https://leanpub.com/beyondpromptandpray
          :alt: Beyond "Prompt and Pray"
     - .. image:: covers/bcap.png
          :width: 150px
          :target: https://leanpub.com/beyondchunkandpray
          :alt: Beyond "Chunk and Pray"
     - .. image:: covers/bsp.png
          :width: 150px
          :target: https://leanpub.com/beyondshipandpray
          :alt: Beyond "Ship and Pray"

1. `Knowledge Graph Embeddings as Geometric Operators
   <https://leanpub.com/KG-Embedding>`_ — the geometric foundation the model
   store is built on.
2. `Beyond "Prompt and Pray" <https://leanpub.com/beyondpromptandpray>`_ —
   building governed agentic systems (:mod:`forgeloop.agents`).
3. `Beyond "Chunk and Pray" <https://leanpub.com/beyondchunkandpray>`_ —
   trustworthy RAG with geometric knowledge graphs (:mod:`forgeloop.rag`).
4. `Beyond "Ship and Pray" <https://leanpub.com/beyondshipandpray>`_ —
   testing agentic systems with geometric ground truth
   (:mod:`forgeloop.testing`).

The propose-and-approve pattern
-------------------------------

The three parts share one pattern. A language model proposes a structured
object — a tool call, a triple, an extracted fact, a drafted answer — and a
geometric model store (the GMS, provided by the licensed ``knowlytix`` backend)
approves or rejects the proposal. The store is not asked to generate; it is
asked to judge, at a threshold fixed on a held-out cohort at a chosen error
rate. A proposal that the store cannot support is denied, escalated to a human,
or answered with an abstention rather than a guess.

This division of labor is what the gates, the retrieval and the abstention
mechanisms have in common: the model supplies coverage and fluency, and the
geometric store supplies a calibrated decision to accept or refuse.

What the library assumes
------------------------

The core install is light — ``pydantic``, ``numpy`` and ``pyyaml`` — and the
loop, typed actions, gates, memory, planning and evaluation run on it alone.
Open-weight model tools and the geometric store are optional extras; the GMS
substrate is licensed and installed separately. The :doc:`installation` page
lists the extras and what each pulls in.
