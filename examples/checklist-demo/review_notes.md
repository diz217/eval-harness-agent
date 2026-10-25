# Human review notes on the live demonstration

The executable agent loop and evidence artifacts were exercised with real model calls. This report is an exploratory draft, not an accepted scientific analysis.

Known narrative-quality issues in this particular run:

- Overall improvement is approximately **8.25 percentage points**, not an 8.2 percent relative gain. Use the computed accuracy values as the authority.
- Cluster `cluster6_05` contains 500 negative and 500 neutral expected labels. Its description as predominantly neutral is inaccurate. The inspected regressed cases are neutral, but that does not describe the entire cluster.
- Coverage is an input-slice measurement, not proof a playbook is inadequate. The exact counts below take precedence over the model's broad coverage statements.
- The summary mentions a proposed check, but this run did not invoke the proposal-saving tool. There is no saved draft in this run. Earlier development output is not a substitute for a proposal from this run.

These are known limitations of the current small-model synthesis, not errors in the recorded numeric observations. The tool layer rejects invalid negative-regression claims about improving groups, but it does not yet validate every natural-language generalization. Report interpretation and root-cause hypotheses require review. Acceptance tests do not establish general diagnostic accuracy.

## Computed coverage observations

```json
[
  {
    "evidence": "e0007",
    "group": "cluster6_05",
    "cases": 1000,
    "covered": 0,
    "uncovered": 1000
  },
  {
    "evidence": "e0014",
    "group": "prescribed_short-inputs",
    "cases": 82,
    "covered": 82,
    "uncovered": 0
  },
  {
    "evidence": "e0015",
    "group": "prescribed_negation",
    "cases": 907,
    "covered": 907,
    "uncovered": 0
  }
]
```
