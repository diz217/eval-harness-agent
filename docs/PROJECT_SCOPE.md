# Agreed project scope

## Goal

Build an analysis agent over existing paired model predictions and ground truth. Scientists supply evaluation directions, metrics, assumptions, and suspected failure modes. The agent performs those analyses, discovers additional observable patterns, tests its interpretations with tools, and proposes concrete changes to evaluation coverage for scientist review.

This is an applied engineering demonstration. Do not expand it into a model-training project, a new clustering research contribution, a general evaluation platform, or a replacement for scientists.

## Responsibilities

1. Verify that A and B evaluate the same cases and expected labels.
2. Run the scientist's specified comparisons and hypotheses using deterministic metrics.
3. Use established clustering/slice-analysis tools to discover groups that were not named in the playbook. Preserve isolated severe failures.
4. Distinguish a group that looks unusual from a group with evidence of worse outcomes. Never manufacture a cause from a cluster name.
5. Track uncertainty, missing evidence, conflicting measurements, and possible label problems.
6. Ground observations in case IDs, tool artifacts, configurations, and computed results.
7. Critique evaluation coverage using concrete counterexamples: aggregation blind spots, missed groups, invalid comparisons, unsupported metrics, and inadequate evidence.
8. Propose a versioned playbook amendment, with its motivation and proposed validation; save as accepted only after explicit scientist review.

## Architecture boundary

Use existing numerical libraries for clustering and statistics. Reuse tracing and evaluation infrastructure such as LangSmith when useful. Custom work is the bounded analytical agent, its typed tools, scientist context, evidence model, and complete application behavior. Workflow execution alone is insufficient.

The LLM chooses analyses and follow-ups using intermediate results. Code computes metrics, verifies artifacts and claims, enforces budgets, and controls accepted playbook versions. The scientist retains ownership of domain objectives and acceptance criteria.

## Demonstration acceptance scenarios

- A known regression is caught by prescribed checks.
- A failure group absent from the supplied playbook is discovered and investigated.
- A harmless cluster is dismissed using outcome evidence.
- A severe isolated failure remains visible for review.
- An inconclusive finding is escalated with the missing evidence identified.
- A proposed playbook addition is reviewed and saved as a new version.

Use published results where available. Keep separately labeled synthetic fixtures for scenarios that cannot be established in the published data. Discovery is exploratory until confirmed under an appropriate procedure that addresses adaptive searching and dependence. No claim of universal unknown-unknown detection.

## Agent evaluation

Measure end-to-end analysis correctness, supported regression precision/recall on labeled incidents, false-alert burden, evidence accuracy, useful findings beyond supplied checks, correct escalation, and investigation cost. An improvement is a verified application outcome; an eloquent summary is insufficient. Acceptance tests and incident replay are required; a research benchmark campaign is not the objective.

## Data decision

Use published CheckList BERT/RoBERTa sentiment results first. Both predictions and explicit expected labels already exist. This avoids training or rerunning the recommendation pipeline. The original recommendation artifacts remain a possible future adapter, but currently contain only aggregate results and would require new case-level export/inference.

A model comparison does not have to be a controlled ablation for this demonstration. Clearly describe A and B as alternative published systems.

## Implemented boundary

The agent runs over `paired_cases.jsonl`, with no access to the source test names in `reference_metadata.jsonl`. Typed tools provide schema checks, paired comparisons, clustering, case retrieval, coverage analysis, descriptive uncertainty, structured findings, and draft check proposals. No independent confirmation cohort is claimed. Keep source-level reference audit information separate from the agent's evidence. Decide confirmation grouping before claiming confirmed discovered slices.
