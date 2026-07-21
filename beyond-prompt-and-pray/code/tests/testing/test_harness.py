"""Mock-based tests of the GraphDOEHarness adapter.

These tests do NOT require the GMSH harness or graphdoe to be installed.
They verify the adapter's data shape and the FactorAttribution helpers.
"""

from agentlab.testing import FactorAttribution


def test_factor_attribution_top_drivers():
    attr = FactorAttribution(
        logistic_table=[
            {"factor": "clarity",   "p_value_adj": 0.001, "significant_adj": True,  "odds_ratios": {}},
            {"factor": "reasoning", "p_value_adj": 0.20,  "significant_adj": False, "odds_ratios": {}},
            {"factor": "aliasing",  "p_value_adj": 0.04,  "significant_adj": True,  "odds_ratios": {}},
        ]
    )
    drivers = attr.top_drivers(k=2)
    assert len(drivers) == 2
    assert drivers[0]["factor"] == "clarity"
    assert drivers[1]["factor"] == "aliasing"


def test_factor_attribution_top_failures():
    attr = FactorAttribution(
        failure_table=[
            {"factor": "clarity", "level": "ambiguous",  "failure_rate": 0.53},
            {"factor": "clarity", "level": "clear",      "failure_rate": 0.05},
            {"factor": "aliasing","level": "misleading", "failure_rate": 0.62},
        ]
    )
    top = attr.top_failures(k=2)
    assert top[0]["failure_rate"] == 0.62
    assert top[1]["failure_rate"] == 0.53


def test_top_drivers_filters_non_significant():
    attr = FactorAttribution(
        logistic_table=[
            {"factor": "x", "p_value_adj": 0.001, "significant_adj": False, "odds_ratios": {}},
            {"factor": "y", "p_value_adj": 0.01,  "significant_adj": True,  "odds_ratios": {}},
        ]
    )
    drivers = attr.top_drivers()
    assert len(drivers) == 1
    assert drivers[0]["factor"] == "y"


def test_empty_attribution_handles_gracefully():
    attr = FactorAttribution()
    assert attr.top_drivers() == []
    assert attr.top_failures() == []
