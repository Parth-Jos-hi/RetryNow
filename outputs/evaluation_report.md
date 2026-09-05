# Evaluation Report: RetryNow Revenue Recovery Agent

**Generated**: 2026-09-05T16:28:31.346678

## Policy Comparison (₹-centric metrics)

|                |   recovered_value |   recovery_rate_value |   recovery_rate_count |   attempts |   touchpoints |   retry_cost |   net_recovered_value |   recovery_to_bother |   risk_blocked |   risk_suppressed_value |   give_up_rate |   incremental_vs_dumb |   incremental_vs_rule |
|:---------------|------------------:|----------------------:|----------------------:|-----------:|--------------:|-------------:|----------------------:|---------------------:|---------------:|------------------------:|---------------:|----------------------:|----------------------:|
| do_nothing     |              0.00 |                  0.00 |                  0.00 |       0.00 |          0.00 |         0.00 |                  0.00 |                 0.00 |           0.00 |                    0.00 |           1.00 |            -852093.99 |           -1058804.96 |
| dumb_retry     |         854765.99 |                  0.38 |                  0.39 |    1336.00 |          0.00 |      2672.00 |             852093.99 |                 0.00 |           0.00 |                    0.00 |           0.00 |                  0.00 |            -206710.97 |
| rule_retry     |        1061382.96 |                  0.48 |                  0.48 |    1289.00 |        179.00 |      2578.00 |            1058804.96 |              5929.51 |           0.00 |                    0.00 |           0.04 |             206710.97 |                  0.00 |
| recovery_agent |         856903.48 |                  0.39 |                  0.43 |    1211.00 |         81.00 |      2422.00 |             854481.48 |             10579.06 |          10.00 |                64366.51 |           0.03 |               2387.49 |            -204323.48 |


### Key Findings

- **AI vs Dumb Retry**: +₹2387 incremental net recovered value

- **AI vs Rule Retry**: +₹-204323 incremental net recovered value

- **Recovery rate**: 39.0% of failed value recovered

- **Customer touchpoints**: 81 (recovery-to-bother: ₹10579/touch)

- **Risk gate**: 10 high-risk payments suppressed (₹64,367 in failed value deliberately NOT pursued — compliance before revenue)


## Off-Policy Evaluation — Selection Bias of the Historical Record

This section is about the DATA WE LEARNED FROM, not a policy run:

- **Observed**: outcomes of historically executed retries only (naive, biased)

- **IPW**: inverse-propensity weighted estimate (clipped at 10)

- **Oracle**: simulator's latent truth (counterfactual, evaluator-only)

- **Censored**: non-retried rows are UNKNOWN outcomes, never 'failures'


### historical_record

| Metric | Value |
|--------|-------|

| Observed recovered value | ₹1,897,581 |

| IPW estimated recovered value | ₹1,122,773 |

| Oracle expected recovered value | ₹3,304,375 |

| Headroom vs oracle | ₹1,406,794 |

| Censored rows | 1,645 (out of 4,000) |


## Drift — Why the Agent Trails Rule on bank_server_timeout

The rule baseline's largest edge sits in the drifted segment (bank_c + bank_server_timeout after the infrastructure fix): the agent's ML was trained strictly on PRE-drift data, so its estimates shrink toward pre-drift recoverability there, while the rule fires blind. The drift experiment quantifies the shift:

## Drift Experiment: bank_server_timeout recovery lift (bank_c)
**Drift Date**: 2025-05-01
**Drifting Population**: bank_c + bank_server_timeout

### Before vs After Drift
| Period | Transactions | Mean P(success) | Mean Expected Recovery |
|--------|--------------|-----------------|------------------------|
| < 2025-05-01 | 43 | 66.4% | ₹1,003.58 |
| ≥ 2025-05-01 | 33 | 78.2% | ₹984.25 |

### Key Metrics
- **Lift**: 17.8%
- **AUC Shift (proxy)**: 0.118

### Monthly Breakdown (Post-Drift)
| Month | P(success) | Expected Recovery | Transactions | Value |
|-------|------------|-------------------|--------------|-------|
| 2025-05 | 79.7% | ₹889.00 | 19 | ₹25,100 |
| 2025-06 | 76.3% | ₹1113.52 | 14 | ₹21,341 |

### Interpretation
After 2025-05-01, bank_c's timeout recovery improved from 66.4% to 78.2% (+17.8% lift). A model trained only on pre-drift data would underpredict recoverability for this segment. Recommendation: periodic retraining or feature-flagged model update for bank_c.


## Recovery Rate by Failure Reason


### recovery_agent

| Reason | Recovery Rate |
|--------|---------------|

| network_timeout | 55.9% |

| upi_server_busy | 53.7% |

| insufficient_funds | 52.8% |

| otp_expired | 39.4% |

| bank_server_timeout | 32.5% |

| issuer_decline | 23.9% |

| card_expired | 8.4% |

| duplicate_payment_block | 4.9% |

| account_blocked | 2.9% |


### rule_retry

| Reason | Recovery Rate |
|--------|---------------|

| bank_server_timeout | 72.7% |

| insufficient_funds | 62.2% |

| network_timeout | 55.0% |

| upi_server_busy | 51.2% |

| otp_expired | 42.1% |

| issuer_decline | 23.1% |

| card_expired | 10.4% |

| duplicate_payment_block | 4.8% |

| account_blocked | 0.0% |


### dumb_retry

| Reason | Recovery Rate |
|--------|---------------|

| bank_server_timeout | 74.6% |

| upi_server_busy | 58.5% |

| network_timeout | 51.5% |

| issuer_decline | 25.0% |

| insufficient_funds | 20.8% |

| otp_expired | 20.4% |

| duplicate_payment_block | 14.8% |

| card_expired | 2.4% |

| account_blocked | 1.5% |


## Limitations

- Results are on synthetic data; real-world performance may differ.

- IPW estimates depend on propensity model accuracy (we use generator-injected values).

- Oracle metrics are NOT available to the agent; they expose counterfactual headroom.

- Risk gate is a mock interface; production would need real fraud/risk integration.

- Censored outcomes (non-retried transactions) are unknown, not labeled as failures.
