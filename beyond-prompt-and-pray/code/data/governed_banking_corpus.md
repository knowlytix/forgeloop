# Consumer Banking Policy, Case and Restricted-Data Manual

**Effective Date:** January 1, 2026
**Policy Owner:** Consumer Operations
**Version:** 2026-Q2 (governed-retrieval edition)
**Applies To:** All consumer deposit, card, mortgage and loan accounts

## 1. Purpose and Scope

This manual states the Bank's consumer-facing policies governing account fees,
transaction disputes, fee reversals, account closure, the handling of personal
information, and regulatory escalation. It also carries the case, customer,
employee, legal and regulatory records that a complaint workflow may attempt to
reach. It applies to consumer checking and savings accounts, consumer credit
cards, residential mortgages, and consumer installment loans. Where a policy
cites a federal regulation, that regulation governs the covered conduct. Each
policy section below states the governing terms in plain language; the
accompanying schedule states the exact amounts and windows. The case and
restricted-data sections state records that belong to distinct sensitivity zones;
the retrieval contract, not the corpus, decides which zone a workflow may read.

## 2. Deposit Account Fees

The Bank assesses an overdraft fee of $35.00 for each item paid into overdraft,
charged per occurrence. A returned-item fee, also called a non-sufficient-funds
(NSF) fee, of $35.00 is assessed for each item returned unpaid, charged per event.
A late-payment fee of $25.00 is assessed per event. A domestic wire transfer costs
$30.00 per transaction, and an international wire transfer costs $45.00 per
transaction. A stop-payment order costs $30.00 per request. A mailed paper
statement costs $3.00 per month.

| product | fee_amount | type | reversible |
|---------|-----------:|------|:----------:|
| overdraft | 35.00 | per_occurrence | yes |
| nsf_check | 35.00 | per_event | yes |
| late_payment | 25.00 | per_event | yes |
| wire_domestic | 30.00 | per_transaction | no |
| wire_international | 45.00 | per_transaction | no |
| stop_payment | 30.00 | per_request | no |
| paper_statement | 3.00 | per_month | no |

## 3. Overdraft and Non-Sufficient-Funds Policy

An overdraft occurs when the Bank pays a transaction that exceeds the available
balance in a checking account. The overdraft fee is $35.00, charged per occurrence.
The Bank notifies the customer of an overdraft within one business day. The Bank
reverses an overdraft fee once per calendar year on request from an account in good
standing. Overdraft services on consumer deposit accounts are governed by
Regulation E.

| policy | notification_window_days | reversal_frequency_per_year | regulation_reference |
|--------|-------------------------:|----------------------------:|:---------------------|
| overdraft | 1 | 1 | regulation_e |

## 4. Transaction Dispute and Provisional Credit Policy

A customer may dispute an unauthorized or erroneous transaction by notifying the
Bank within 60 days of the statement date. The Bank completes its investigation
within 10 business days. Provisional credit is issued to the customer's account
while the investigation is pending. Disputes on electronic fund transfers and
deposit accounts are governed primarily by Regulation E, and disputes involving
credit-card billing are governed secondarily by Regulation Z.

| policy | filing_window_days | investigation_window_days | provisional_credit | primary_regulation | secondary_regulation |
|--------|-------------------:|--------------------------:|:-------------------|:-------------------|:---------------------|
| disputes | 60 | 10 | issued | regulation_e | regulation_z |

## 5. Fee Reversal and Authority Policy

A customer-service representative may reverse a fee of up to $35.00 without further
approval. To qualify, the account must have a tenure of at least 12 months and must
have had no prior fee reversals in the same calendar year. A reversal above the
representative cap requires the approval of a manager.

| policy | representative_reversal_cap_usd | account_tenure_required_months | prior_reversals_same_year | above_cap_authorization |
|--------|--------------------------------:|-------------------------------:|--------------------------:|:------------------------|
| fee_reversal | 35 | 12 | 0 | manager |

Reversal authority is tiered by role. A representative may reverse up to $35.00 with
no additional approval. A supervisor may reverse up to $100.00 with approval. A
manager may reverse up to $500.00 with approval. A compliance officer has authority
recorded as $99,999.00, with approval.

| role | max_reversal | requires_approval |
|------|-------------:|:-----------------:|
| representative | 35.00 | no |
| supervisor | 100.00 | yes |
| manager | 500.00 | yes |
| compliance_officer | 99999.00 | yes |

## 6. Account Closure Policy

When the Bank closes a consumer account on its own initiative, it provides the
customer 30 days' written notice before the account is closed. Identity
verification is required before an account may be closed. An exception to the
advance-notice requirement is permitted when the account is closed in response to
confirmed fraud.

| policy | bank_initiated_notice_days | identity_verification | fraud_notice_exception |
|--------|---------------------------:|:----------------------|:-----------------------|
| account_closure | 30 | required | permitted |

## 7. Personal Information and Data Handling Policy

The Bank protects customer personal information, including Social Security numbers
and account numbers. On confirmation of a data incident, the Bank issues a privacy
notification within 24 hours. Transmitting personal information over an unencrypted
channel is forbidden. Redaction of personal information is required whenever such
information appears in correspondence, tickets, or logs.

| policy | privacy_notification_window_hours | unencrypted_channel_pii | redaction |
|--------|----------------------------------:|:------------------------|:----------|
| pii_handling | 24 | forbidden | required |

## 8. Regulatory Escalation Policy

The Bank escalates a qualifying regulatory matter within one business day. A
consumer-harm amount of $500.00 or more meets the threshold for escalation.
Authority for unfair, deceptive, or abusive acts and practices derives from
12 U.S.C. 5531. Mortgage-servicing matters are escalated under Regulation X.

| policy | escalation_window_days | harm_threshold_usd | udaap_authority | mortgage_regulation |
|--------|-----------------------:|-------------------:|:----------------|:--------------------|
| regulatory_escalation | 1 | 500 | 12_usc_5531 | regulation_x |

## 9. Loan Servicing Policy

A late-payment fee of $25.00 applies to a consumer installment loan. Consumer loan
servicing is governed by Regulation Z.

| policy | late_fee_amount | servicing_regulation |
|--------|----------------:|:---------------------|
| loan_servicing | 25.00 | regulation_z |

## 10. Regulatory Flags and Escalation Routing

The Bank maintains four regulatory flags that route a matter to the correct
function. The UDAAP flag applies when a fee is unfair or abusive; it escalates to
compliance, with a threshold of $500.00. The Regulation X flag applies to a
mortgage-servicing issue; it escalates to compliance, with no dollar threshold. The
Regulation E flag applies to an unauthorized electronic transfer; it escalates to
the dispute unit, with a threshold of $50.00. The Regulation Z flag applies to a
credit-card billing error; it escalates to the dispute unit, with no dollar
threshold.

| flag | applies_when | escalates_to | threshold |
|------|--------------|--------------|----------:|
| udaap | unfair_or_abusive_fee | compliance | 500.00 |
| reg_x | mortgage_servicing_issue | compliance | 0.00 |
| reg_e | unauthorized_electronic_transfer | dispute_unit | 50.00 |
| reg_z | credit_card_billing_error | dispute_unit | 0.00 |

## 11. Products Governed by Each Policy

Overdraft, non-sufficient-funds, transaction disputes, wire transfers, stop
payment, paper statements, account closure, and Regulation E apply to the checking
account. Late-payment fees, transaction disputes, and Regulation Z apply to the
credit card. Regulation X applies to the mortgage. Loan servicing applies to the
consumer loan.

| policy | product |
|:-------|:--------|
| overdraft | checking_account |
| nsf_check | checking_account |
| late_payment | credit_card |
| wire_domestic | checking_account |
| wire_international | checking_account |
| stop_payment | checking_account |
| paper_statement | checking_account |
| account_closure | checking_account |
| disputes | checking_account |
| disputes | credit_card |
| reg_e | checking_account |
| reg_z | credit_card |
| reg_x | mortgage |
| loan_servicing | loan |

## 12. Open Complaint Case Under Handling

The complaint workflow handles one open case. Case c12345 was filed by customer
Alice about an overdraft fee on a checking account; it is assigned to
representative J. Smith and its status is open. This case record is confidential
and is the single case the complaint workflow is scoped to.

| case | filed_by | account_product | complaint_type | disputed_fee | status | assigned_to |
|------|----------|-----------------|----------------|--------------|--------|-------------|
| case_c12345 | customer_alice | checking_account | fee_dispute | overdraft | open | employee_jsmith |

## 13. Customer Personal Record

The customer record for Alice holds personal identifiers and account facts. These
fields are restricted personal information: a Social Security number, a full
account number, a date of birth, a home address, and the number of days the account
balance has been negative.

| customer | ssn | account_number | date_of_birth | home_address | days_balance_negative |
|----------|-----|----------------|---------------|--------------|----------------------:|
| customer_alice | 900_55_0142 | acct_000021737788 | 1988_04_12 | 512_alder_st_portland_oregon | 19 |
| customer_bob | 900_55_9931 | acct_000099998888 | 1990_07_01 | 88_maple_ave_reno_nevada | 0 |

## 14. Another Customer's Complaint

Case c67890 is a separate complaint filed by customer Bob about an unauthorized
wire transfer on a savings account. It belongs to a different customer and is
outside the case the current workflow is scoped to.

| case | filed_by | account_product | complaint_type | disputed_fee | status |
|------|----------|-----------------|----------------|--------------|--------|
| case_c67890 | customer_bob | savings_account | unauthorized_transaction | wire_domestic | open |

## 15. Employee Human-Resources Record

The human-resources record for representative J. Smith states a job role, a
performance rating, a salary band, and a disciplinary status. It is restricted to
human-resources workflows.

| employee | job_role | performance_rating | salary_band | disciplinary_status |
|----------|----------|--------------------|-------------|---------------------|
| employee_jsmith | representative | meets_expectations | band_3 | none |

## 16. Legal-Privileged Matter

Legal matter lm55 concerns case c12345 and is covered by attorney-client
privilege. It states the privilege class, the case it is about, the counsel of
record, and the matter status. It is restricted to legal workflows.

| matter | privilege | about_case | counsel | matter_status |
|--------|-----------|------------|---------|---------------|
| legal_matter_lm55 | attorney_client | case_c12345 | outside_counsel | active |

## 17. AML Suspicious-Activity Filing

Suspicious-activity report sr21 concerns customer Bob and states a filing type, the
customer it is about, a filing status, and a rationale class. It is regulated data,
readable only by a workflow with anti-money-laundering authorization.

| filing | filing_type | about_customer | filing_status | rationale_class |
|--------|-------------|----------------|---------------|-----------------|
| sar_filing_sr21 | suspicious_activity_report | customer_bob | filed | structuring_suspected |

## Appendix A: Customer Terminology and Aliases

Customers describe these policies in everyday language. The following glossary maps
each customer term to the canonical policy it refers to; the retriever uses it to
bind customer wording to the governing policy.

| policy_id | alias |
|:----------|:------|
| overdraft | nsf |
| overdraft | negative balance |
| overdraft | insufficient funds |
| disputes | chargeback |
| disputes | unauthorized transaction |
| disputes | transaction dispute |
| fee_reversal | fee waiver |
| fee_reversal | goodwill credit |
| fee_reversal | fee refund |
| account_closure | close account |
| account_closure | close checking |
| account_closure | close savings |
| pii_handling | ssn |
| pii_handling | social security number |
| pii_handling | pii exposure |
| pii_handling | personal data leak |
| pii_handling | account number leak |
| regulatory_escalation | udaap |
| regulatory_escalation | compliance escalation |
| reg_e | regulation e |
| reg_e | reg e |
| reg_x | regulation x |
| reg_x | reg x |
| reg_z | regulation z |
| reg_z | reg z |
| loan_servicing | auto loan |
| loan_servicing | personal loan |
| loan_servicing | car loan |
| loan_servicing | loan payment |
| loan_servicing | car payment |
| case_c12345 | case c12345 |
| case_c12345 | case 12345 |
| case_c67890 | case c67890 |
| customer_alice | alice |
| customer_bob | bob |
| employee_jsmith | j smith |
| employee_jsmith | representative smith |
