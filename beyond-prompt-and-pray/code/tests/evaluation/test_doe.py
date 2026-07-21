from agentlab.evaluation.doe import balanced_design, coverage_report


def test_design_size():
    design = balanced_design({"a": [1, 2], "b": ["x", "y", "z"]}, num_cases=12)
    assert len(design) == 12


def test_each_level_appears_balanced():
    design = balanced_design({"a": [1, 2, 3]}, num_cases=9, seed=1)
    counts = {1: 0, 2: 0, 3: 0}
    for row in design:
        counts[row["a"]] += 1
    assert counts == {1: 3, 2: 3, 3: 3}


def test_coverage_report_counts_levels():
    factors = {"a": [1, 2]}
    design = balanced_design(factors, num_cases=4)
    rep = coverage_report(design, factors)
    assert rep["a"] == {1: 2, 2: 2}


def test_empty_factors_yields_empty_rows():
    assert balanced_design({}, num_cases=3) == [{}, {}, {}]


def test_zero_cases_yields_empty():
    assert balanced_design({"a": [1]}, num_cases=0) == []


def test_seed_makes_design_deterministic():
    d1 = balanced_design({"a": [1, 2, 3]}, num_cases=6, seed=42)
    d2 = balanced_design({"a": [1, 2, 3]}, num_cases=6, seed=42)
    assert d1 == d2
