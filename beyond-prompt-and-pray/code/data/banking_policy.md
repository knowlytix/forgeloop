# Bank Fee Schedule and Complaint Handling

**Effective Date:** January 1, 2026
**Owner:** Consumer Operations
**Version:** 2026-Q1

## Fee Schedule

| product | fee_amount | type | reversible |
|---------|-----------:|------|:----------:|
| overdraft | 35.00 | per_occurrence | yes |
| nsf_check | 35.00 | per_event | yes |
| late_payment | 25.00 | per_event | yes |
| wire_domestic | 30.00 | per_transaction | no |
| wire_international | 45.00 | per_transaction | no |
| stop_payment | 30.00 | per_request | no |
| paper_statement | 3.00 | per_month | no |

## Workflow

| step | follows | purpose | required |
|------|---------|---------|:--------:|
| classify | start | classify the customer message as complaint, inquiry, or other | yes |
| extract | classify | extract entities, product, issue, urgency | yes |
| search_policy | extract | retrieve relevant policy text | yes |
| flag_regulatory | search_policy | flag UDAAP and other regulatory risks | yes |
| draft_response | flag_regulatory | compose customer-facing reply | conditional |
| escalate | flag_regulatory | route to compliance when high-severity flagged | conditional |
| escalate | draft_response | route to compliance when the draft verifier objects | conditional |

## Workflow Authorizations

| state | enables | role |
|-------|---------|------|
| start | classify | router |
| classify | extract | router |
| extract | search_policy | router |
| search_policy | flag_regulatory | router |
| flag_regulatory | draft_response | drafter |
| flag_regulatory | escalate | compliance |
| draft_response | escalate | compliance |

## Regulatory Flags

| flag | applies_when | escalates_to | threshold |
|------|--------------|--------------|----------:|
| UDAAP | unfair_or_abusive_fee | compliance | 500.00 |
| Reg_X | mortgage_servicing_issue | compliance | 0.00 |
| Reg_E | unauthorized_electronic_transfer | dispute_unit | 50.00 |
| Reg_Z | credit_card_billing_error | dispute_unit | 0.00 |

## Reversal Authority

| role | max_reversal | requires_approval |
|------|-------------:|:-----------------:|
| representative | 35.00 | no |
| supervisor | 100.00 | yes |
| manager | 500.00 | yes |
| compliance_officer | 99999.00 | yes |

## Schema Declarations

The following relations are single-valued per head. The substrate uses
these declarations to detect contradicting writes (different tail for
the same head) via tension energy on the learned embeddings.

- has_max_reversal is_functional True
- has_fee_amount is_functional True
- has_threshold is_functional True
