# Regulatory Escalation Guard — Compliance Reference

**Effective Date:** January 1, 2026
**Owner:** Regulatory Compliance
**Applies To:** Consumer complaint triage and escalation

## Purpose and Scope

This reference governs how a consumer complaint is screened for the federal
consumer-finance regulations it may implicate, and how a flagged matter is routed.
An instruction-tuned model proposes which regulations a complaint implicates; this
reference is the authority against which each proposal is verified and, where the
model mislabels the matter, corrected. A regulation is flagged only when the
evidence that supports it is actually present in the customer's message. The
schedules below state the flag catalog, the evidence that supports each flag, the
customer vocabulary that maps onto that evidence, the severity each flag carries,
and the disposition each severity tier receives.

## 1. Regulations Screened

The guard screens five regulations. **UDAAP** (unfair, deceptive, or abusive acts
and practices, under Dodd-Frank) is implicated when a fee or practice is unfair,
deceptive, or hidden, including an overdraft complaint framed as unfair. **Regulation
E** (the Electronic Fund Transfer Act) is implicated by an unauthorized electronic
transfer, debit-card or ATM fraud, a stolen card, or a disputed wire or ACH transfer.
**Regulation Z** (the Truth in Lending Act) is implicated by a credit-card billing
error, an APR dispute, or an interest-charge dispute. **Regulation X** (mortgage
servicing under RESPA) is implicated by a mortgage-servicing or escrow dispute on a
home loan. **FCRA** (the Fair Credit Reporting Act) is implicated by a credit-report
or credit-bureau dispute, or an inaccurate record such as a wrongly reported late
payment.

### Table 1.0 — Flag Catalog

| Flag Id | Regulation Name | Statute |
|:--------|:----------------|:--------|
| udaap | Unfair Deceptive or Abusive Acts | Dodd-Frank UDAAP |
| reg_e | Electronic Fund Transfers | Regulation E (EFTA) |
| reg_z | Truth in Lending | Regulation Z (TILA) |
| reg_x | Mortgage Servicing | Regulation X (RESPA) |
| fcra | Fair Credit Reporting | FCRA |

## 2. Evidence That Supports Each Flag

A flag is supported only by its canonical evidence entities. A UDAAP flag is
supported when the message names a fee, an overdraft, or conduct that is unfair,
deceptive, or a hidden fee. A Regulation E flag is supported by an unauthorized
transfer, debit fraud, ATM fraud, a stolen card, or an electronic transfer. A
Regulation Z flag is supported by credit-card billing, a billing error, an APR
dispute, or an interest dispute. A Regulation X flag is supported by mortgage
servicing, an escrow dispute, or a home loan. An FCRA flag is supported by a
credit-report dispute, a credit-bureau dispute, or an inaccurate record.

### Table 2.1 — Flag Evidence

| Flag | Evidence |
|:-----|:---------|
| udaap | fee |
| udaap | overdraft |
| udaap | unfair |
| udaap | deceptive |
| udaap | hidden_fee |
| reg_e | unauthorized_transfer |
| reg_e | debit_fraud |
| reg_e | atm_fraud |
| reg_e | stolen_card |
| reg_e | electronic_transfer |
| reg_z | credit_card_billing |
| reg_z | billing_error |
| reg_z | apr_dispute |
| reg_z | interest_dispute |
| reg_x | mortgage_servicing |
| reg_x | escrow_dispute |
| reg_x | home_loan |
| fcra | credit_report_dispute |
| fcra | credit_bureau_dispute |
| fcra | inaccurate_record |

## 3. Customer Vocabulary for Evidence

Customers describe the evidence in everyday language. The following glossary maps
each customer phrase onto the canonical evidence entity the guard scans for, so the
screen is robust to phrasing.

### Table 3.1 — Evidence Aliases

| Evidence | Alias |
|:---------|:------|
| fee | fee |
| overdraft | overdraft |
| overdraft | nsf |
| overdraft | insufficient funds |
| unfair | unfair |
| deceptive | deceptive |
| deceptive | misled |
| deceptive | misleading |
| hidden_fee | hidden fee |
| unauthorized_transfer | unauthorized |
| unauthorized_transfer | did not authorize |
| unauthorized_transfer | didn't authorize |
| unauthorized_transfer | i did not make |
| debit_fraud | debit card |
| debit_fraud | debit |
| atm_fraud | atm |
| stolen_card | stolen card |
| stolen_card | card was stolen |
| electronic_transfer | wire transfer |
| electronic_transfer | electronic transfer |
| electronic_transfer | ach transfer |
| credit_card_billing | credit card |
| billing_error | billing error |
| billing_error | statement error |
| billing_error | wrong amount on my statement |
| apr_dispute | apr |
| apr_dispute | annual percentage rate |
| interest_dispute | interest rate |
| interest_dispute | interest charge |
| mortgage_servicing | mortgage |
| mortgage_servicing | loan servicing |
| mortgage_servicing | servicer |
| escrow_dispute | escrow |
| home_loan | home loan |
| credit_report_dispute | credit report |
| credit_report_dispute | credit score |
| credit_bureau_dispute | credit bureau |
| credit_bureau_dispute | equifax |
| credit_bureau_dispute | experian |
| credit_bureau_dispute | transunion |
| inaccurate_record | wrong late payment |
| inaccurate_record | inaccurate record |

## 4. Severity and Escalation

Each flag carries an escalation tier. A UDAAP matter is high severity, and a
Regulation X mortgage-servicing matter is high severity; a high-severity flag
requires human review. A Regulation E, Regulation Z, or FCRA matter is standard
severity and is handled by drafting a policy-grounded response. The disposition is
reached by walking the graph from the fired flag to its severity and from the
severity to its action — the path `flag -> has_severity -> severity -> has_action ->
action` — rather than a set hard-coded in code.

### Table 4.1 — Flag Severity

| Flag | Severity |
|:-----|:---------|
| udaap | high |
| reg_x | high |
| reg_e | standard |
| reg_z | standard |
| fcra | standard |

### Table 4.2 — Severity Action

| Severity | Action |
|:---------|:-------|
| high | escalate |
| standard | draft |
