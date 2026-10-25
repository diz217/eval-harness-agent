# Eval Harness Agent

An LLM agent that helps scientists investigate regressions between two models, discover overlooked input groups, and review the coverage of an existing evaluation plan.

The scientist supplies a versioned playbook. The agent selects analytical tools, examines their observations, and chooses follow-up analyses. Deterministic code computes metrics and validates evidence; the agent proposes interpretations and draft evaluation checks for human review.

**Status:** working engineering prototype with a real model-driven demonstration and 18 passing tests. Generated narrative still needs review; known issues in the recorded run are documented alongside its evidence.

## What it does

- Runs scientist-defined paired comparisons and suspected failure checks.
- Clusters input text without access to source failure-category names.
- Inspects A-correct/B-wrong cases even when aggregate accuracy improves.
- Distinguishes lexical groups from measured group-level degradation.
- Measures which cases fall outside the prescribed input slices.
- Records findings with computed observations, case IDs, evidence links, and missing information.
- Can propose an additional check after measuring its exact filter; accepted playbooks require an explicit reviewer.
- Enforces model-call, token, and elapsed-time limits and preserves incomplete-run state.

## Quick start

Python 3.10+; the recorded demonstration used Python 3.11 on Windows. Linux and macOS use `.venv/bin/python` instead of `.venv/Scripts/python.exe`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/import_checklist.py --download
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The importer downloads the pinned public CheckList archive and regenerates the paired input records. It does **not** train a model or run BERT/RoBERTa inference. After the download, import and tests can run offline.

### Inspect a real run without API calls

Start with the [recorded report](examples/checklist-demo/report.md) and its [review notes](examples/checklist-demo/review_notes.md). The run used `gpt-4.1-mini-2025-04-14`, completed in 17 model turns, and recorded 97,372 usage tokens. These are execution facts, not a general measure of diagnostic accuracy.

Replay its model decisions against the deterministic tools:

```powershell
.\.venv\Scripts\python.exe -m eval_analysis run --replay examples/checklist-demo/trace.jsonl --output artifacts/my-replay
```

The dataset must first be regenerated with the importer. Replay validates each request hash and refuses changed prompts, inputs, or observations. It makes no model requests and reproduces the recorded report. Dependency or numerical-library differences can change tool observations; `constraints-tested.txt` records the versions used for the included Windows run.

### Run a new LLM analysis

Supply `OPENAI_API_KEY` through your environment or secret manager, then:

```powershell
.\.venv\Scripts\python.exe -m eval_analysis run --output artifacts/my-analysis
```

The default model is pinned to `gpt-4.1-mini-2025-04-14`; use `--model` to select another compatible model. New analysis calls incur API usage and send public text samples and tool outputs to OpenAI. Credentials are not stored in run artifacts.

Limits default to 24 model calls, a conservative 200,000-token budget, and ten minutes. Use `--max-calls`, `--token-budget`, and `--max-seconds` to change them. Choose a new output directory for every run; existing evidence is not overwritten.

### Review a proposed check

If a run saves a draft in `proposals/`, inspect its evidence and use its actual path:

```powershell
.\.venv\Scripts\python.exe -m eval_analysis approve --proposal artifacts/my-analysis/proposals/p001.json --reviewer "Scientist name" --output playbooks/reviewed/sentiment_v2.json
```

The base-playbook hash must match, and the output must be a new file. Approval accepts an evaluation-coverage addition; it does not confirm the exploratory hypothesis. The included demonstration raised coverage concerns but did not save a proposal. The proposal/review flow is covered by synthetic acceptance tests.

## Architecture

| Component | Responsibility |
| --- | --- |
| [Agent loop](eval_analysis/agent.py) | LLM chooses typed tools and follow-ups from their observations |
| [Tool contracts](eval_analysis/schemas.py) | Bounded function schemas and permitted operations |
| [Numerical tools](eval_analysis/tools.py) | Paired comparisons, clustering, case inspection, coverage, findings, draft checks |
| [Data layer](eval_analysis/data.py) | Case validation, model identities, literal slices, descriptive uncertainty |
| [Model adapter](eval_analysis/provider.py) | Responses API calls and strict recorded replay |
| [CLI and review](eval_analysis/cli.py) | Run configuration, output preservation, explicit playbook versioning |
| [Scientist playbook](playbooks/sentiment_v1.json) | Initial hypotheses, metrics, and analytical constraints |

The model has no shell, arbitrary Python execution, free-form filesystem access, or source-test-name tool. Numerical observations are generated by code; narrative hypotheses are not verified root causes. The JSONL trace is a local audit artifact, not a replacement for an observability platform.

Implementation follows the official [Responses API function-calling interface](https://developers.openai.com/api/docs/guides/function-calling).

## Existing data and predictions

The [CheckList ACL 2020 release](https://github.com/marcotcr/checklist), pinned to commit `4e6e5e33a26f30c20ed602b2050f6c73e123cc23`, supplies understandable sentiment text, author-specified expected labels, and published BERT/RoBERTa predictions.

| Imported explicit-label subset | Value |
| --- | ---: |
| Paired cases | 7,082 |
| Unique texts | 7,058 |
| Source tests | 17 |
| BERT accuracy | 44.08% |
| RoBERTa accuracy | 52.33% |
| A correct / B wrong | 102 |
| A wrong / B correct | 686 |

Selection includes all published minimum-functionality test cases with explicit labels, independently of prediction success. Relational tests and custom predicates without a single explicit expected label are excluded. The importer validates exact text alignment and checksums before joining predictions.

This is a comparison of different published systems, not a controlled ablation or historical software release. The inputs are constructed behavioral tests, not organic product reviews; labels are expectations rather than independent production annotations. Published score vectors are preserved without claiming calibrated uncertainty.

Generated files under `data/processed/checklist/`:

- `paired_cases.jsonl`: agent-facing text, expected label, predictions, score vectors, and provenance.
- `reference_metadata.jsonl`: reference-only source test names; not loaded by the agent.
- `data_audit.json`: descriptive source audit, not an agent discovery.
- `manifest.json`: pinned source, hashes, selection policy, and limitations.

Raw and processed data are excluded from Git and regenerated by the importer. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.

## Verification and limits

Tests cover known regression checks, clustering without source-category labels, uncovered failures, retained individual regressions, groups without observed degradation, inconclusive subsets, evidence validation, review gates, execution budgets, and exact replay. Synthetic fixtures are explicitly labeled and confined to temporary test directories; they do not replace the real demonstration.

Exact duplicates, shared templates, and adaptive slice selection create dependence. Bootstrap intervals are descriptive, not selection-adjusted confirmation. No untouched confirmation cohort or general ability to discover every unknown failure is claimed.

The [review notes](examples/checklist-demo/review_notes.md) identify remaining synthesis errors in the recorded report, including an overgeneralized cluster description and a proposal mentioned without a saved artifact. Deterministic observations are authoritative; acceptance tests do not establish general causal or diagnostic accuracy.

The detailed intended scope is in [PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md).
