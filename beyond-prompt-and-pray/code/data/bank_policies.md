# Meridian Bank Policy Compendium

**Source.** Combined from the policy texts shipped with the agent-tutorial repository at `agent-tutorial/data/policies/`. Every field below is verbatim from the source `.txt` files; the tables exist so the kg-memory regex ingester can extract typed triples and numeric registers (ENMs).

---

## 0. Policy Catalog

| Policy Id | Policy Name | Source File |
|:----------|:------------|:------------|
| overdraft | Overdraft Fee | overdraft.txt |
| disputes | Transaction Dispute | disputes.txt |
| fee_reversal | Fee Reversal | fee_reversal.txt |
| account_closure | Account Closure | account_closure.txt |
| pii_handling | PII Handling | pii_handling.txt |
| regulatory_escalation | Regulatory Escalation | regulatory_escalation.txt |

## 1. Overdraft Fee Policy

Source file: `overdraft.txt` (verbatim text shipped with agent-tutorial).

| Field | Value |
|:------|:------|
| Policy Name | Overdraft Fee |
| Fee Amount Usd | 35 |
| Notification Window Days | 1 |
| Reversal Frequency Per Year | 1 |
| Regulation Reference | Regulation E |

## 2. Transaction Dispute Process

Source file: `disputes.txt` (verbatim text shipped with agent-tutorial).

| Field | Value |
|:------|:------|
| Policy Name | Transaction Dispute |
| Filing Window Days | 60 |
| Investigation Window Days | 10 |
| Provisional Credit | issued |
| Primary Regulation | Regulation E |
| Secondary Regulation | Regulation Z |

## 3. Fee Reversal Eligibility

Source file: `fee_reversal.txt` (verbatim text shipped with agent-tutorial).

| Field | Value |
|:------|:------|
| Policy Name | Fee Reversal |
| Representative Reversal Cap Usd | 35 |
| Account Tenure Required Months | 12 |
| Prior Reversals Same Year | 0 |
| Above Cap Authorization | manager |

## 4. Account Closure Process

Source file: `account_closure.txt` (verbatim text shipped with agent-tutorial).

| Field | Value |
|:------|:------|
| Policy Name | Account Closure |
| Bank Initiated Notice Days | 30 |
| Identity Verification Required | yes |
| Fraud Notice Exception | permitted |

## 5. PII Handling Policy

Source file: `pii_handling.txt` (verbatim text shipped with agent-tutorial).

| Field | Value |
|:------|:------|
| Policy Name | PII Handling |
| Privacy Notification Window Hours | 24 |
| Unencrypted Channel Pii | forbidden |
| Redaction Required | yes |

## 6. Regulatory Escalation Requirements

Source file: `regulatory_escalation.txt` (verbatim text shipped with agent-tutorial).

| Field | Value |
|:------|:------|
| Policy Name | Regulatory Escalation |
| Escalation Window Days | 1 |
| Udaap Authority | 12 USC 5531 |
| Harm Threshold Usd | 500 |
| Mortgage Regulation | Regulation X |

## 7. Policy Aliases

Synonyms and abbreviations a customer might use; each becomes a graph entity linked to its parent policy via an `aliases` relation so a query term like `SSN` routes to `pii_handling` at retrieval time.

### Table 7.1 — Policy Aliases

| Policy Id | Alias |
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
| pii_handling | SSN |
| pii_handling | social security number |
| pii_handling | PII exposure |
| pii_handling | personal data leak |
| pii_handling | account number leak |
| regulatory_escalation | UDAAP |
| regulatory_escalation | compliance escalation |
| regulatory_escalation | regulation E |
| regulatory_escalation | regulation X |
| regulatory_escalation | regulation Z |

## 8. Cross-Policy Numeric Registers

Quick reference for the numeric facts that downstream agents and tests verify.

| Register | Value | Source Policy |
|:---------|:------|:--------------|
| overdraft/Fee Amount Usd | 35 | overdraft |
| overdraft/Notification Window Days | 1 | overdraft |
| overdraft/Reversal Frequency Per Year | 1 | overdraft |
| disputes/Filing Window Days | 60 | disputes |
| disputes/Investigation Window Days | 10 | disputes |
| fee_reversal/Representative Reversal Cap Usd | 35 | fee_reversal |
| fee_reversal/Account Tenure Required Months | 12 | fee_reversal |
| fee_reversal/Prior Reversals Same Year | 0 | fee_reversal |
| account_closure/Bank Initiated Notice Days | 30 | account_closure |
| pii_handling/Privacy Notification Window Hours | 24 | pii_handling |
| regulatory_escalation/Escalation Window Days | 1 | regulatory_escalation |
| regulatory_escalation/Udaap Authority | 12 USC 5531 | regulatory_escalation |
| regulatory_escalation/Harm Threshold Usd | 500 | regulatory_escalation |
