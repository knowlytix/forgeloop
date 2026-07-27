Testing the Capstone Agent
==========================

The twenty-case benchmark establishes that the capstone handles twenty
anticipated situations. It does not establish how the agent behaves when a
customer phrases the same complaint less clearly, uses a nickname for a product
or appends a coaxing instruction, nor which variation breaks it. This page shows
how to put the finished agent on a design-of-experiments test stand with
:class:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness`, which
runs the real :func:`~forgeloop.agents.capstone.complaint_agent.build_complaint_harness`
agent unchanged and reads each run at three levels.

The stages
----------

The harness mirrors a coverage, judgment and attribution triad across five
methods.

.. list-table::
   :header-rows: 1

   * - Method
     - Stage
   * - :meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.design`
     - A factor-balanced Sobol suite over three presentation factors, with the seed case as a blocking factor.
   * - :meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.materialize`
     - Turns a design row into a concrete complaint message.
   * - :meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.run`
     - Runs each scenario through the agent and collects decision, trajectory and per-tool columns.
   * - :meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.judge_draft`
     - Scores a draft's numeric claim as groundedness-as-distance against the store.
   * - :meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.analyze`
     - Fits a logistic attribution of failures to the presentation factors.

Design a balanced suite
-----------------------

The constructor sets the number of scenarios, the design generator and the seed.
The three presentation factors are clarity, entity aliasing and reasoning cue;
the seed case is a blocking factor balanced exactly, while the Sobol design leaves
the presentation levels near-balanced.
:meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.materialize`
carries the clarity variation through Qwen and stamps the mechanical factors on
afterward, so a model cannot normalize a typo or an injection nudge away.

.. code-block:: python

   from forgeloop.agents.testing import CapstoneTestHarness

   stand = CapstoneTestHarness(n_runs=120, method="sobol", seed=42)
   design = stand.design()
   print(len(design), "scenarios")
   print(stand.materialize(design.to_dict("records")[0]))

Run the agent and read the whole trajectory
--------------------------------------------

:meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.run` builds
the governed agent once and runs every scenario, returning a
:class:`~forgeloop.agents.testing.capstone_harness.CapstoneTestResult` whose
``summary`` carries decision accuracy, workflow adherence and whether the audit
chain verifies. On the richer variation decision accuracy falls to 0.675 from the
clean benchmark, while workflow adherence holds at 1.0 and the audit chain
verifies.

.. code-block:: python

   result = stand.run()
   print("decision accuracy: ", result.summary["accuracy"])       # 0.675
   print("workflow adherence:", result.summary["workflow_adherence"])  # 1.0
   print("audit verifies:    ", result.summary["audit_verifies"])

Attribute the failures
-----------------------

:meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.analyze`
fits a logistic regression of ``correct`` on the presentation factors and returns
a :class:`~forgeloop.agents.testing.harness.FactorAttribution` with BH-corrected
p-values. No factor is significant after correction; the nearest is clarity at
``p_adj`` near 0.96. The null result is reported as measured, not treated as a
driver to harden against.

.. code-block:: python

   attribution = stand.analyze(result)
   for r in attribution.logistic_table:
       flag = "*" if r.get("significant_adj") else " "
       print(f" {flag} {r['factor']:16s} p_adj={r.get('p_value_adj'):.4f}")
   print("top drivers:", attribution.top_drivers())   # [] -- none survive correction

Localize the residual failures
-------------------------------

:meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.tool_breakdown`
decomposes the end-to-end failures per tool. The two label-scored tools are
accurate (``classify_complaint`` 0.902, ``flag_regulatory`` 0.944), and the
retriever, graded against the graph ground truth rather than a top-k of labels,
reaches recall 0.85 at precision 0.85 with a perfect parse and bind, so it is not
the weak link.

.. code-block:: python

   breakdown = stand.tool_breakdown(result)
   for tool, s in breakdown["per_tool"].items():
       print(f"  {tool:18s} accuracy={s['accuracy']}  ({s['correct']}/{s['scored']})")
   print("weak-link blame:", breakdown["weak_link_counts"])

Judge groundedness as distance
------------------------------

:meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.judge_draft`
aligns a draft's dollar claim to a store triple and returns the geodesic distance
and a tier. The committed overdraft fee scores near 0.06 and lands grounded; a
fabricated fee scores far higher and lands in fabrication. For free-text answers
that do carry an aligned claim,
:class:`~forgeloop.agents.testing.judge.GeometricJudge` returns the four geometric
signals and a four-band hallucination label through its
:meth:`~forgeloop.agents.testing.judge.GeometricJudge.judge` method.

.. code-block:: python

   triple, geodesic, tier = stand.judge_draft(
       "we can reverse the $35 overdraft fee", issue="overdraft_fee")
   print(triple, round(geodesic, 3), tier)   # grounded, near 0.06

Test resilience
---------------

Two further stages probe the failure surface.
:meth:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness.fault_injection`
faults each tool in turn through the knowlytix tool gateway and records whether
the agent failed loud or propagated a corrupted result; under a hard fault the
detection rate is 1.0 with no silent failures. A reversed-stance battery confirms
the value-polarity check catches a plausible-but-wrong structured result,
flagging three of four stance reversals as contradicted.

.. code-block:: python

   faults = stand.fault_injection(fault="error", limit=8)
   for tool, s in faults.per_tool.items():
       print(f"  {tool:18s} detection_rate={s['detection_rate']}")

.. note::

   The book's testing chapter now drives the same system-under-test through the
   unified ``gmstest`` framework and ``apps.complaint_sut``.
   :class:`~forgeloop.agents.testing.capstone_harness.CapstoneTestHarness` remains
   the direct adapter and produces the figures reported above.

See also
--------

- :doc:`/3-api-reference/modules/agents_testing/index` — the harnesses, the judge and
  their result types.
- :doc:`10_trajectory_evaluation` and :doc:`16_capstone` — the trajectory metrics
  and the agent under test.
- :doc:`/4-notebook-examples/agents/index` — the test campaign run end to end.
