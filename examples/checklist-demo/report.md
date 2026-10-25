# Evaluation analysis report

Status: exploratory; scientist review required.

## Agent interpretation

Model B shows overall improved accuracy over A with approximately 8.2% gain. However, specific weaknesses exist. The prescribed short-inputs check reveals slight degradation on single-word sentiment inputs with a 2.4% accuracy drop and some regressions misclassifying positive or negative words, potentially indicating weaker lexical sensitivity. The negation check passes with a strong positive accuracy delta, showing no regression on inputs containing 'not'. Unsupervised clustering identified several lexical input groups; cluster 6_00 shows no degradation despite many error disagreements, indicating harmless lexical variability without regression. Cluster 6_05, characterized by phrases like 'not say given' in neutral sentences, has isolated failures where model B misclassifies them as negative, despite showing overall accuracy improvement. Coverage checks confirm these clusters and the prescribed slices do not overlap, indicating the playbook may miss emerging failure modes related to complex negations or polite hedging phrases. A proposed new check targeting neutral sentences with negation and hedging phrases such as 'not say' is supported by isolated failures and should be further validated on expanded data.

## Computed overall observations

```json
{
  "group_id": "all",
  "cases": 7082,
  "baseline_accuracy": 0.44083592205591643,
  "candidate_accuracy": 0.5232985032476701,
  "accuracy_delta": 0.08246258119175374,
  "correct_to_wrong": 102,
  "wrong_to_correct": 686,
  "both_wrong": 3274,
  "expected_label_counts": {
    "negative": 2833,
    "neutral": 3290,
    "positive": 959
  },
  "status": "exploratory",
  "confirmation": "No independent confirmation cohort was supplied",
  "uncertainty_flags": [
    "unmeasured template dependence; adaptive slice selection"
  ],
  "descriptive_bootstrap_95": [
    0.07560544087162972,
    0.08993780954587267
  ],
  "interval_interpretation": "Exact-text-group bootstrap only; not selection-adjusted, not a confirmatory interval"
}
```

## Findings

### f001: Degradation on short single-word sentiment inputs

Kind: exploratory_regression

Agent hypothesis (unverified): The candidate model shows degradation for very short input texts (single words) in sentiment classification, misclassifying positive and negative single-word examples.

```json
{
  "group_id": "prescribed_short-inputs",
  "cases": 82,
  "baseline_accuracy": 0.8414634146341463,
  "candidate_accuracy": 0.8170731707317073,
  "accuracy_delta": -0.024390243902439025,
  "correct_to_wrong": 2,
  "wrong_to_correct": 0,
  "both_wrong": 13,
  "expected_label_counts": {
    "negative": 35,
    "neutral": 13,
    "positive": 34
  },
  "status": "exploratory",
  "confirmation": "No independent confirmation cohort was supplied",
  "uncertainty_flags": [
    "few discordant pairs",
    "unmeasured template dependence; adaptive slice selection"
  ],
  "descriptive_bootstrap_95": [
    -0.06097560975609756,
    0.0
  ],
  "interval_interpretation": "Exact-text-group bootstrap only; not selection-adjusted, not a confirmatory interval"
}
```

Evidence: [e0002](evidence/e0002.json), [e0004](evidence/e0004.json)

Cases: checklist-sentiment-000031, checklist-sentiment-000058

Missing evidence: More cases needed to confirm severity and prevalence outside the provided examples

### f002: No group-level regression detected in cluster 6_00 despite many both-wrong cases

Kind: no_detected_group_regression

Agent hypothesis (unverified): Cluster 6_00 has many neutral-labeled cases with different errors but shows a positive accuracy delta, indicating no group-level degradation in model B.

```json
{
  "group_id": "cluster6_00",
  "cases": 2387,
  "baseline_accuracy": 0.5069124423963134,
  "candidate_accuracy": 0.5609551738583997,
  "accuracy_delta": 0.0540427314620863,
  "correct_to_wrong": 41,
  "wrong_to_correct": 170,
  "both_wrong": 1007,
  "expected_label_counts": {
    "negative": 915,
    "neutral": 1128,
    "positive": 344
  },
  "status": "exploratory",
  "confirmation": "No independent confirmation cohort was supplied",
  "uncertainty_flags": [
    "unmeasured template dependence; adaptive slice selection"
  ],
  "descriptive_bootstrap_95": [
    0.04189359028068706,
    0.06535400083787181
  ],
  "interval_interpretation": "Exact-text-group bootstrap only; not selection-adjusted, not a confirmatory interval"
}
```

Evidence: [e0005](evidence/e0005.json)

Cases: checklist-sentiment-000069, checklist-sentiment-000070, checklist-sentiment-000071

Missing evidence: Further inspection of both-wrong cases for potential disagreements on expected labels

### f003: Isolated failures in cluster 6_05 involving 'not say given' neutral sentences misclassified as negative

Kind: isolated_failure

Agent hypothesis (unverified): Cluster 6_05 predominantly contains neutral-labeled sentences with phrases like 'not say given' that the candidate model misclassifies as negative in isolated cases despite overall accuracy improvement.

```json
{
  "group_id": "cluster6_05",
  "cases": 1000,
  "baseline_accuracy": 0.139,
  "candidate_accuracy": 0.352,
  "accuracy_delta": 0.213,
  "correct_to_wrong": 9,
  "wrong_to_correct": 222,
  "both_wrong": 639,
  "expected_label_counts": {
    "negative": 500,
    "neutral": 500,
    "positive": 0
  },
  "status": "exploratory",
  "confirmation": "No independent confirmation cohort was supplied",
  "uncertainty_flags": [
    "unmeasured template dependence; adaptive slice selection"
  ],
  "descriptive_bootstrap_95": [
    0.189,
    0.23902499999999996
  ],
  "interval_interpretation": "Exact-text-group bootstrap only; not selection-adjusted, not a confirmatory interval"
}
```

Evidence: [e0006](evidence/e0006.json)

Cases: checklist-sentiment-084566, checklist-sentiment-084599, checklist-sentiment-084655, checklist-sentiment-084682, checklist-sentiment-084683

Missing evidence: More cases and independent confirmation cohort to assess operational impact and prevalence

## Proposed checks

## Limitations

- Labels are author-specified expectations and not independently verified
- No independent confirmation cohort was supplied
- Adaptive slice selection and template dependence limit confirmatory conclusions
- Small sample sizes in some prescribed and lexical slices limit robustness
- Business impact and severity of errors were not annotated
- Clustering is lexical and does not necessarily imply causal failure modes
- No independent confirmation cohort
- Template dependence and adaptive selection
- Narrative hypotheses are not verified root causes
- Business severity was not supplied
