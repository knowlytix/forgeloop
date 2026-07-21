# Bank Fee Policy (canonicalization demo)

A deliberately *fragmented* policy: the overdraft product is extracted under two
surface names -- `overdraft` and the customer term `od_fee` -- in both tables,
with identical facts. A clean ingester would emit one entity; this one emits two.
Canonicalization should merge them (they are v-identical and u-consistent) while
leaving the genuinely distinct products alone.

## Fee Schedule

| product | fee_amount | type | reversible |
|---------|-----------:|------|:----------:|
| overdraft | 35.00 | per_occurrence | yes |
| od_fee | 35.00 | per_occurrence | yes |
| late_payment | 25.00 | per_event | yes |
| wire_domestic | 30.00 | per_transaction | no |
| stop_payment | 30.00 | per_request | no |

## Overdraft Policy Details

| policy | notification_window_days | regulation_reference |
|--------|-------------------------:|:---------------------|
| overdraft | 1 | regulation_e |
| od_fee | 1 | regulation_e |
