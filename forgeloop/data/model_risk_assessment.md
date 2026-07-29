# Model Risk Assessment Report — Enterprise AI Governance

**Meridian Financial Technologies, Inc.**

**Assessment Date:** June 30, 2026

**Review Period:** Q2 2026 (Post-Remediation Review)

---

## 1. Executive Summary

| Item | Detail |
|:-----|:-------|
| Organization | Meridian Financial Technologies, Inc. |
| Total Models in Inventory | 48 |
| Models Under Review | 12 |
| High-Risk Models | 3 |
| Medium-Risk Models | 5 |
| Low-Risk Models | 4 |
| Regulatory Framework | SR 11-7 (Federal Reserve Guidance on Model Risk Management) |
| Chief Model Risk Officer | Dr. Elena Vasquez |
| Assessment Team | Model Validation Unit (MVU), 8 senior validators |

This Q2 2026 assessment updates the Q1 2026 review with the outcomes of the Q1 remediation
program. MDL-001 completed its fairness remediation in April and has been redesignated
**Medium** risk following validation of the revised calibration layer. MDL-005 deteriorated
further and has been escalated to **High** risk pending a full rebuild. MDL-013 was onboarded
in May 2026 and is included in this quarter's review.

## 2. Model Inventory Summary

### Table 2.1 — Models Under Review

| Model ID | Model Name | Domain | Tier | Risk Rating | Status |
|:---------|:-----------|:-------|:----:|:-----------:|:------:|
| MDL-001 | CreditScore-XGB | Credit Decisioning | 1 | Medium | Stable |
| MDL-002 | FraudNet-LSTM | Fraud Detection | 1 | Medium | Watch |
| MDL-003 | AML-GBM | Anti-Money Laundering | 1 | High | Degraded |
| MDL-004 | MarketVaR-GARCH | Market Risk VaR | 1 | High | Stable |
| MDL-005 | ChurnPredict-RF | Customer Analytics | 2 | High | Watch |
| MDL-006 | CollateralVal-NN | Collateral Valuation | 2 | Medium | Watch |
| MDL-007 | PrepayModel-LR | Prepayment Modeling | 2 | Medium | Stable |
| MDL-008 | LGD-Ensemble | Loss Given Default | 2 | Medium | Watch |
| MDL-009 | PD-Logistic | Probability of Default | 2 | Medium | Stable |
| MDL-010 | RateModel-SVR | Interest Rate Pricing | 3 | Low | Stable |
| MDL-011 | SegmentCluster-KM | Customer Segmentation | 3 | Low | Stable |
| MDL-012 | DocClassifier-BERT | Document Classification | 3 | Low | Stable |
| MDL-013 | LiquidityRisk-Transformer | Liquidity Risk | 2 | Medium | Stable |

### Table 2.2 — Model Architecture Details

| Model ID | Algorithm | Features | Training Samples | Training Date | Retrain Freq |
|:---------|:----------|:--------:|:----------------:|:-------------:|:------------:|
| MDL-001 | XGBoost (depth=6) | 142 | 2,450,000 | 2026-04-15 | Quarterly |
| MDL-002 | LSTM (3-layer) | 87 | 8,900,000 | 2025-11-01 | Monthly |
| MDL-003 | Gradient Boosting | 234 | 12,345,678 | 2025-06-20 | Semi-Annual |
| MDL-004 | GARCH(1,1) | 15 | 1,260 (days) | 2025-12-01 | Monthly |
| MDL-005 | Random Forest | 53 | 890,000 | 2025-10-10 | Quarterly |
| MDL-006 | Neural Network (4-layer) | 28 | 156,000 | 2025-08-15 | Semi-Annual |
| MDL-007 | Logistic Regression | 12 | 340,000 | 2025-07-01 | Annual |
| MDL-008 | Stacked Ensemble | 67 | 1,200,000 | 2025-09-01 | Quarterly |
| MDL-009 | Logistic Regression | 18 | 2,100,000 | 2025-10-15 | Quarterly |
| MDL-010 | Support Vector Regression | 8 | 45,000 | 2025-11-20 | Monthly |
| MDL-011 | K-Means (k=7) | 31 | 2,300,000 | 2025-12-10 | Quarterly |
| MDL-012 | BERT-base | 768 | 450,000 | 2025-05-01 | Semi-Annual |
| MDL-013 | Transformer (6-layer) | 96 | 670,000 | 2026-05-01 | Monthly |

## 3. Performance Metrics

### Table 3.1 — Accuracy Diagnostics

| Model ID | Train AUC | Test AUC | Train-Test Gap | Gini | KS Statistic | Brier Score |
|:---------|----------:|---------:|:--------------:|-----:|:------------:|:-----------:|
| MDL-001 | 0.9067 | 0.8689 | 0.0378 | 0.7378 | 0.5214 | 0.1287 |
| MDL-002 | 0.9489 | 0.9156 | 0.0333 | 0.8312 | 0.6534 | 0.0867 |
| MDL-003 | 0.9123 | 0.8756 | 0.0367 | 0.7512 | 0.5234 | 0.1245 |
| MDL-004 | 0.8234 | 0.8012 | 0.0222 | 0.6024 | 0.3891 | 0.1876 |
| MDL-005 | 0.7734 | 0.6912 | 0.0822 | 0.3824 | 0.2789 | 0.2345 |
| MDL-006 | 0.8567 | 0.8123 | 0.0444 | 0.6246 | 0.4234 | 0.1678 |
| MDL-007 | 0.7456 | 0.7389 | 0.0067 | 0.4778 | 0.2891 | 0.2245 |
| MDL-008 | 0.8789 | 0.8345 | 0.0444 | 0.6690 | 0.4567 | 0.1456 |
| MDL-009 | 0.7623 | 0.7534 | 0.0089 | 0.5068 | 0.3012 | 0.2156 |
| MDL-010 | 0.9012 | 0.8891 | 0.0121 | 0.7782 | 0.5891 | 0.0978 |
| MDL-011 | 0.6789 | 0.6534 | 0.0255 | 0.3068 | 0.2012 | 0.2678 |
| MDL-012 | 0.9234 | 0.8567 | 0.0667 | 0.7134 | 0.5123 | 0.1345 |
| MDL-013 | 0.8845 | 0.8612 | 0.0233 | 0.7224 | 0.4923 | 0.1312 |

### Table 3.2 — Robustness Under Perturbation

| Model ID | Noise 1% | Noise 5% | Noise 10% | Noise 20% | Robustness Rating |
|:---------|:--------:|:--------:|:---------:|:---------:|:-----------------:|
| MDL-001 | 0.8578 | 0.8401 | 0.8134 | 0.7489 | Robust |
| MDL-002 | 0.9198 | 0.8945 | 0.8612 | 0.7934 | Robust |
| MDL-003 | 0.8712 | 0.8456 | 0.8012 | 0.7123 | Moderate |
| MDL-004 | 0.7989 | 0.7812 | 0.7534 | 0.7012 | Moderate |
| MDL-005 | 0.6834 | 0.6401 | 0.5712 | 0.4523 | Fragile |
| MDL-006 | 0.8089 | 0.7823 | 0.7412 | 0.6534 | Moderate |
| MDL-007 | 0.7367 | 0.7289 | 0.7156 | 0.6934 | Robust |
| MDL-008 | 0.8301 | 0.8045 | 0.7634 | 0.6712 | Moderate |
| MDL-009 | 0.7512 | 0.7423 | 0.7289 | 0.7034 | Robust |
| MDL-010 | 0.8867 | 0.8712 | 0.8489 | 0.8012 | Robust |
| MDL-011 | 0.6489 | 0.6123 | 0.5612 | 0.4534 | Fragile |
| MDL-012 | 0.8823 | 0.8634 | 0.8289 | 0.7612 | Robust |
| MDL-013 | 0.8512 | 0.8334 | 0.8067 | 0.7445 | Robust |

### Table 3.3 — Fairness Metrics (Credit & Lending Models Only)

| Model ID | Protected Group | Approval Rate | Reference Rate | Adverse Impact Ratio | Status |
|:---------|:---------------|:-------------:|:--------------:|:--------------------:|:------:|
| MDL-001 | Race: Black | 71.23% | 79.45% | 0.8965 | PASS |
| MDL-001 | Race: Hispanic | 72.34% | 79.45% | 0.9106 | PASS |
| MDL-001 | Race: Asian | 80.89% | 79.45% | 1.0181 | PASS |
| MDL-001 | Gender: Female | 76.23% | 78.12% | 0.9758 | PASS |
| MDL-001 | Age: Under 25 | 68.45% | 77.23% | 0.8863 | PASS |
| MDL-009 | Race: Black | 65.78% | 76.34% | 0.8616 | PASS |
| MDL-009 | Race: Hispanic | 63.45% | 76.34% | 0.8312 | PASS |
| MDL-009 | Gender: Female | 73.89% | 75.12% | 0.9836 | PASS |
| MDL-009 | Age: Under 25 | 61.23% | 74.56% | 0.8213 | PASS |

## 4. Data Quality Assessment

### Table 4.1 — Feature Drift Detection (PSI Scores)

| Model ID | Feature Category | Current PSI | Threshold | Drift Status |
|:---------|:-----------------|:-----------:|:---------:|:------------:|
| MDL-001 | Income Features | 0.1234 | 0.20 | Stable |
| MDL-001 | Bureau Features | 0.1567 | 0.20 | Stable |
| MDL-001 | Behavioral Features | 0.0891 | 0.20 | Stable |
| MDL-002 | Transaction Patterns | 0.2834 | 0.25 | ALERT |
| MDL-002 | Device Fingerprints | 0.1567 | 0.25 | Stable |
| MDL-003 | Transaction Volume | 0.0456 | 0.15 | Stable |
| MDL-003 | Customer Profile | 0.1789 | 0.15 | ALERT |
| MDL-005 | Usage Patterns | 0.3412 | 0.20 | ALERT |
| MDL-005 | Demographic Features | 0.0234 | 0.20 | Stable |
| MDL-006 | Market Indices | 0.0678 | 0.15 | Stable |
| MDL-006 | Property Attributes | 0.1923 | 0.15 | ALERT |
| MDL-012 | Text Distribution | 0.1834 | 0.20 | Stable |
| MDL-013 | Liquidity Features | 0.0923 | 0.20 | Stable |

### Table 4.2 — Missing Data Rates

| Model ID | Feature Set | Missing Rate Q1 2026 | Missing Rate Q2 2026 | Delta | Acceptable Threshold |
|:---------|:-----------|:--------------------:|:--------------------:|:-----:|:--------------------:|
| MDL-001 | Core Features | 1.89% | 1.34% | -0.55% | 3.00% |
| MDL-001 | Bureau Features | 5.67% | 3.12% | -2.55% | 5.00% |
| MDL-002 | Real-Time Features | 0.15% | 0.18% | +0.03% | 1.00% |
| MDL-003 | Customer Data | 2.78% | 3.01% | +0.23% | 4.00% |
| MDL-005 | Engagement Data | 7.89% | 4.23% | -3.66% | 5.00% |
| MDL-006 | Valuation Inputs | 2.12% | 2.34% | +0.22% | 3.00% |
| MDL-008 | Loss History | 1.23% | 1.34% | +0.11% | 2.00% |
| MDL-012 | Document Corpus | 0.67% | 0.72% | +0.05% | 2.00% |
| MDL-013 | Liquidity Inputs | 0.89% | 0.91% | +0.02% | 2.00% |

## 5. Risk Findings and Remediation

### Table 5.1 — Critical Findings

| Finding ID | Model ID | Severity | Category | Description | Remediation Deadline |
|:-----------|:---------|:--------:|:---------|:------------|:--------------------:|
| FND-007 | MDL-005 | Critical | Accuracy | Test AUC 0.6912 below minimum 0.75 performance threshold | 2026-07-31 |
| FND-008 | MDL-005 | High | Robustness | AUC degrades to 0.4523 at 20% noise (Fragile) | 2026-08-15 |
| FND-013 | MDL-002 | High | Data Quality | Transaction Patterns PSI 0.2834 exceeds drift threshold | 2026-07-15 |
| FND-014 | MDL-003 | High | Data Quality | Customer Profile PSI 0.1789 exceeds 0.15 threshold | 2026-08-01 |
| FND-015 | MDL-005 | Critical | Data Quality | Usage Patterns PSI 0.3412 exceeds drift threshold | 2026-07-15 |
| FND-016 | MDL-006 | Medium | Data Quality | Property Attributes PSI 0.1923 exceeds 0.15 threshold | 2026-08-31 |

### Table 5.2 — Model Risk Scores (Composite)

| Model ID | Accuracy Score | Robustness Score | Fairness Score | Data Quality Score | Composite Risk Score | Risk Tier |
|:---------|:-------------:|:----------------:|:--------------:|:------------------:|:--------------------:|:---------:|
| MDL-001 | 82 | 78 | 84 | 81 | 81.25 | Medium |
| MDL-002 | 89 | 85 | N/A | 63 | 79.00 | Medium |
| MDL-003 | 84 | 67 | N/A | 58 | 69.67 | High |
| MDL-004 | 76 | 65 | N/A | 89 | 76.67 | Medium |
| MDL-005 | 38 | 28 | N/A | 33 | 33.00 | Critical |
| MDL-006 | 78 | 56 | N/A | 54 | 62.67 | High |
| MDL-007 | 68 | 82 | N/A | 91 | 80.33 | Low |
| MDL-008 | 81 | 61 | N/A | 87 | 76.33 | Medium |
| MDL-009 | 71 | 79 | 88 | 92 | 82.50 | Low |
| MDL-010 | 89 | 88 | N/A | 95 | 90.67 | Low |
| MDL-011 | 42 | 31 | N/A | 78 | 50.33 | High |
| MDL-012 | 82 | 78 | N/A | 71 | 77.00 | Low |
| MDL-013 | 80 | 78 | N/A | 89 | 82.33 | Low |

## 6. Dependency and Impact Analysis

### Table 6.1 — Model Dependencies

| Upstream Model | Downstream Model | Dependency Type | Impact if Upstream Fails |
|:---------------|:-----------------|:----------------|:------------------------|
| MDL-009 | MDL-001 | PD feeds into credit scoring | Credit scores unreliable |
| MDL-009 | MDL-008 | PD used in LGD calibration | Loss estimates biased |
| MDL-001 | MDL-008 | Credit scores segment LGD pools | Pool assignment errors |
| MDL-004 | MDL-010 | VaR informs rate risk premium | Mispriced interest rates |
| MDL-006 | MDL-008 | Collateral values adjust LGD | Recovery rate errors |
| MDL-011 | MDL-005 | Segments define churn cohorts | Wrong churn populations |
| MDL-012 | MDL-003 | Doc classification feeds AML alerts | Missed suspicious activity |
| MDL-013 | MDL-004 | Liquidity signals inform VaR stress | Understated tail risk |

### Table 6.2 — Business Impact Quantification

| Model ID | Annual Revenue Impact | Annual Loss Exposure | Capital Impact | Regulatory Penalty Risk |
|:---------|:---------------------:|:--------------------:|:--------------:|:-----------------------:|
| MDL-001 | $267,890,000 | $21,345,000 | $6,789,000 | $2,340,000 |
| MDL-002 | $0 | $89,123,000 | $0 | $5,600,000 |
| MDL-003 | $0 | $0 | $0 | $34,567,000 |
| MDL-004 | $0 | $23,456,000 | $67,890,000 | $2,345,000 |
| MDL-005 | $56,789,000 | $34,567,000 | $4,567,000 | $3,456,000 |
| MDL-006 | $0 | $34,567,000 | $8,901,000 | $1,234,000 |
| MDL-007 | $12,345,000 | $5,678,000 | $3,456,000 | $0 |
| MDL-008 | $0 | $67,890,000 | $23,456,000 | $4,567,000 |
| MDL-009 | $0 | $78,901,000 | $15,678,000 | $6,789,000 |
| MDL-010 | $45,678,000 | $12,345,000 | $5,678,000 | $0 |
| MDL-011 | $23,456,000 | $0 | $0 | $0 |
| MDL-012 | $0 | $0 | $0 | $12,345,000 |
| MDL-013 | $34,567,000 | $8,901,000 | $4,567,000 | $0 |

## 7. Comparison to Prior Period

### Table 7.1 — Quarter-over-Quarter Performance Changes

| Model ID | Test AUC Q1 2026 | Test AUC Q2 2026 | Delta AUC | Status Change |
|:---------|:----------------:|:----------------:|:---------:|:-------------:|
| MDL-001 | 0.8412 | 0.8689 | +0.0277 | Degraded → Stable |
| MDL-002 | 0.9234 | 0.9156 | -0.0078 | Watch → Watch |
| MDL-003 | 0.8756 | 0.8756 | 0.0000 | Degraded → Degraded |
| MDL-004 | 0.8012 | 0.8012 | 0.0000 | Stable → Stable |
| MDL-005 | 0.7234 | 0.6912 | -0.0322 | Degraded → Watch |
| MDL-006 | 0.8123 | 0.8123 | 0.0000 | Watch → Watch |
| MDL-007 | 0.7389 | 0.7389 | 0.0000 | Stable → Stable |
| MDL-008 | 0.8345 | 0.8345 | 0.0000 | Watch → Watch |
| MDL-009 | 0.7534 | 0.7534 | 0.0000 | Stable → Stable |
| MDL-010 | 0.8891 | 0.8891 | 0.0000 | Stable → Stable |
| MDL-011 | 0.6534 | 0.6534 | 0.0000 | Stable → Stable |
| MDL-012 | 0.8567 | 0.8567 | 0.0000 | Degraded → Stable |
| MDL-013 | N/A | 0.8612 | N/A | New → Stable |

## 8. Remediation Tracking

### Table 8.1 — Closed and Open Remediation Actions

| Action ID | Finding ID | Model ID | Action | Owner | Due Date | Status | % Complete |
|:----------|:-----------|:---------|:-------|:------|:--------:|:------:|:----------:|
| ACT-001 | FND-001 | MDL-001 | Retrain with fairness constraints | Data Science Team | 2026-04-30 | Closed | 100% |
| ACT-002 | FND-002 | MDL-001 | Add age-aware calibration layer | Data Science Team | 2026-04-30 | Closed | 100% |
| ACT-003 | FND-003 | MDL-001 | Reduce depth, add regularization | Data Science Team | 2026-05-15 | Closed | 100% |
| ACT-004 | FND-004 | MDL-001 | Refresh bureau data pipeline | Data Engineering | 2026-04-30 | Closed | 100% |
| ACT-005 | FND-005 | MDL-002 | Retrain on recent transaction data | Fraud Analytics | 2026-04-15 | Closed | 100% |
| ACT-013 | FND-013 | MDL-002 | Q2 PSI monitoring refresh | Fraud Analytics | 2026-07-15 | In Progress | 40% |
| ACT-014 | FND-014 | MDL-003 | AML feature reweighting | AML Team | 2026-08-01 | Planned | 0% |
| ACT-015 | FND-015 | MDL-005 | Full model rebuild with new features | Customer Analytics | 2026-07-31 | In Progress | 65% |
| ACT-016 | FND-016 | MDL-006 | Property attribute refresh | Valuation Team | 2026-08-31 | Planned | 0% |

---

*This report was prepared by the Model Validation Unit (MVU) in accordance with SR 11-7
guidance. All findings are subject to independent review by Internal Audit. This Q2 report
supersedes Q1 findings that have been closed; remaining open findings are tracked in
Table 8.1.*
