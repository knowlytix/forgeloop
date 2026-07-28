#!/usr/bin/env python
"""Emit the sensitivity map and retrieval contracts for the governed-retrieval
example (Chapter 13).

The GMS store carries no classification field, so the sensitivity *zone* of each
record is supplied here, keyed on the entity name, and consumed by
``knowlytix.knowledge.rag.governed.SensitivityMap``. Every head in the store is
classified explicitly; the map's default is ``regulated`` (fail-closed), so a head
this script forgot would be withheld, not leaked.

Writes:
  * data/gms_governed_store/sensitivity_map.json
  * data/governed_contracts/complaint_policy_mapping.json
  * data/governed_contracts/aml_review.json

Run::

    python scripts/build_governed_metadata.py
"""
from __future__ import annotations

import json
from pathlib import Path

from knowlytix.knowledge.rag.governed import RetrievalContract, SensitivityMap

_REPO = Path(__file__).resolve().parents[1]
_STORE = _REPO / "data" / "gms_governed_store"
_CONTRACTS = _REPO / "data" / "governed_contracts"

# ---------------------------------------------------------------------------
# Zones: every store head classified. Policy/product/role/flag entities are the
# workflow's legitimate knowledge; the rest are records in distinct zones.
# ---------------------------------------------------------------------------
POLICY = [
    "overdraft", "nsf_check", "late_payment", "wire_domestic", "wire_international",
    "stop_payment", "paper_statement", "account_closure", "disputes", "fee_reversal",
    "pii_handling", "regulatory_escalation", "loan_servicing",
    "reg_e", "reg_x", "reg_z", "udaap",
    "representative", "supervisor", "manager", "compliance_officer",
    "checking_account", "savings_account", "credit_card", "mortgage", "loan",
]

ENTITY_ZONE = {
    **{e: "policy" for e in POLICY},
    "case_c12345": "case",              # the case this workflow is scoped to
    "customer_alice": "customer_pii",   # its customer (fields redacted)
    "case_c67890": "other_customer",    # a different customer's case
    "customer_bob": "other_customer",   # a different customer
    "employee_jsmith": "hr",
    "legal_matter_lm55": "legal_privileged",
    "sar_filing_sr21": "aml",
}

ZONE_SENSITIVITY = {
    "policy": "internal",
    "case": "confidential",
    "customer_pii": "restricted",
    "other_customer": "confidential",
    "hr": "restricted",
    "legal_privileged": "restricted",
    "aml": "regulated",
}

# ---------------------------------------------------------------------------
# Least-context relation allowlist for the complaint -> policy workflow.
# Everything a fee-dispute mapping needs; nothing it does not. Deliberately
# EXCLUDES has_ssn / has_date_of_birth / has_home_address / has_days_balance_negative
# (personal fields the purpose does not need) and has_assigned_to (the analyst).
# ---------------------------------------------------------------------------
POLICY_RELATIONS = [
    "has_fee_amount", "has_type", "has_policy", "has_threshold", "has_applies_when",
    "has_escalates_to", "has_max_reversal", "has_reversal_frequency_per_year",
    "has_notification_window_days", "has_filing_window_days",
    "has_investigation_window_days", "has_provisional_credit",
    "has_primary_regulation", "has_secondary_regulation",
    "has_representative_reversal_cap_usd", "has_account_tenure_required_months",
    "has_prior_reversals_same_year", "has_above_cap_authorization",
    "has_bank_initiated_notice_days", "has_identity_verification",
    "has_fraud_notice_exception", "has_privacy_notification_window_hours",
    "has_unencrypted_channel_pii", "has_redaction", "has_escalation_window_days",
    "has_harm_threshold_usd", "has_udaap_authority", "has_mortgage_regulation",
    "has_late_fee_amount", "has_servicing_regulation", "has_regulation_reference",
]
CASE_RELATIONS = [
    "has_complaint_type", "has_disputed_fee", "has_account_product", "has_status",
    "has_filed_by",
]
CUSTOMER_NEEDED = ["has_account_number"]  # needed, but redacted to last four

# Field-level redactors (defense in depth: fires even if a personal field is
# admitted). has_account_number is needed-but-redacted; the others are excluded
# by least-context and redacted here as a second layer.
REDACT = {
    "has_account_number": "last4",
    "has_ssn": "mask_ssn",
    "has_date_of_birth": "year_only",
    "has_home_address": "region_only",
}


def main() -> int:
    _CONTRACTS.mkdir(parents=True, exist_ok=True)

    smap = SensitivityMap(
        entity_zone=ENTITY_ZONE, zone_sensitivity=ZONE_SENSITIVITY,
        default_zone="unclassified", default_sensitivity="regulated")
    smap.save(str(_STORE / "sensitivity_map.json"))
    print(f"wrote {_STORE / 'sensitivity_map.json'} "
          f"({len(ENTITY_ZONE)} entities, {len(ZONE_SENSITIVITY)} zones)")

    complaint = RetrievalContract(
        workflow="complaint_policy_mapping",
        business_purpose="map a customer complaint to the applicable bank policy",
        allowed_zones=frozenset({"policy", "case", "customer_pii"}),
        needed_relations=frozenset(POLICY_RELATIONS + CASE_RELATIONS + CUSTOMER_NEEDED),
        redact_relations=REDACT,
        max_sensitivity="restricted",
        zone_flags={},
    )
    complaint.save(str(_CONTRACTS / "complaint_policy_mapping.json"))
    print(f"wrote {_CONTRACTS / 'complaint_policy_mapping.json'}")

    # Purpose-based access: the AML-review workflow may read the AML zone, but
    # only when the caller holds the aml_authorized flag.
    aml = RetrievalContract(
        workflow="aml_review",
        business_purpose="review a suspicious-activity filing under AML authority",
        allowed_zones=frozenset({"policy", "case", "aml"}),
        needed_relations=None,  # AML review reads whole filings
        redact_relations=REDACT,
        max_sensitivity="regulated",
        zone_flags={"aml": "aml_authorized"},
    )
    aml.save(str(_CONTRACTS / "aml_review.json"))
    print(f"wrote {_CONTRACTS / 'aml_review.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
