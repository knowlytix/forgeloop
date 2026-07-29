# Bank Policy (Chapter 2 excerpt)

A small excerpt of the bank's policy, used to build a knowledge graph and a
geometric memory store for the Chapter 2 primer.

## Fee Schedule

| product | fee_amount | type |
|---------|-----------:|------|
| overdraft | 35.00 | per_occurrence |
| nsf_check | 35.00 | per_event |
| late_payment | 25.00 | per_event |
| wire_domestic | 30.00 | per_transaction |
| wire_international | 45.00 | per_transaction |
| stop_payment | 30.00 | per_request |

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

| role | max_reversal |
|------|-------------:|
| representative | 35.00 |
| supervisor | 100.00 |
| manager | 500.00 |

## Schema Declarations

- has_fee_amount is_functional True
- has_max_reversal is_functional True
- has_threshold is_functional True
