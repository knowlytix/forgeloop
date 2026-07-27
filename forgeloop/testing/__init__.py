"""gmstest — catalog-driven test-suite construction for GMS-grounded testing.

Two layers, two catalogs (see catalogs/):

    base_catalog.yaml    WHAT is asked + WHAT is true   (content + behavior)
    factor_catalog.yaml  HOW it is presented            (ground-truth invariant)

A *profile* (profiles.yaml) pairs a base selection with a factor selection and a
scenario mode. Resolution expands families/groups and filters factors to those
whose `applies_to` matches the chosen bases ("enrichment enriches the base").

Base questions come from one of three sources:

    CatalogBaseSource   mine the GMS store with the selected generators
    SeedCaseSource      user-labeled cases w/ per-component ground truth
    UserBaseSource      user supplies (query, answer) pairs directly; no GMS

Scenarios are composed in one of two modes:

    cross      Option 1 — base x design matrix (presentation factors only)
    embedded   Option 2 — base is a factor in the design (JASA Table 1)

and consumed two ways:

    emit_*     Chapter 15 — SFT / fine-tuning data (classifier, draft LoRA)
    evaluate   Chapter 16 — run through a SUT; score + attribute failures
"""
from .catalogs import BaseSpec, Catalog, FactorSpec, Profile
from .resolve import ResolvedSuite, applicable_factors, resolve, resolve_profile
from .sources import (
    BaseSource, CatalogBaseSource, QAItem, SeedCaseSource, UserBaseSource,
)
from .compose import Scenario, compose, graphdoe_design, simple_design
from .emit import (
    draft_prompt, emit_classifier_sft, emit_draft_sft, passes_contract, to_jsonl,
)
from . import evaluate

__all__ = [
    "BaseSpec", "FactorSpec", "Profile", "Catalog",
    "ResolvedSuite", "resolve", "resolve_profile", "applicable_factors",
    "QAItem", "BaseSource", "UserBaseSource", "SeedCaseSource", "CatalogBaseSource",
    "Scenario", "compose", "simple_design", "graphdoe_design",
    "emit_classifier_sft", "emit_draft_sft", "draft_prompt", "passes_contract", "to_jsonl",
    "evaluate",
]
