"""CLI for live analysis, offline trace replay, and explicit scientist review."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import jsonschema

from .agent import Agent
from .data import Dataset
from .provider import DEFAULT_MODEL, OpenAIProvider, ReplayProvider
from .schemas import FILTER
from .tools import AnalysisTools, content_hash, save_json

ROOT = Path(__file__).resolve().parents[1]


def approve(proposal_path, playbook_path, output, reviewer):
    proposal = json.loads(proposal_path.read_text(encoding="utf-8-sig"))
    playbook = json.loads(playbook_path.read_text(encoding="utf-8-sig"))
    if output.exists():
        raise ValueError("Refusing to overwrite an existing reviewed playbook")
    if not reviewer.strip():
        raise ValueError("Explicit reviewer identity is required")
    if proposal["status"] != "draft_pending_scientist_review" or proposal["base_playbook_sha256"] != content_hash(playbook):
        raise ValueError("Proposal is not pending or its base playbook has changed")
    jsonschema.validate(proposal["filter"], FILTER)
    amended = json.loads(json.dumps(playbook))
    amended["version"] += 1
    amended["id"] = playbook["id"].rsplit("-v", 1)[0] + "-v" + str(amended["version"])
    amended["checks"].append({"id": "reviewed-" + content_hash(proposal)[:10],
                               "hypothesis": proposal["title"], "filter": proposal["filter"]})
    amended["review"] = {"reviewer": reviewer, "approved_at": datetime.now(timezone.utc).isoformat(),
                         "proposal_sha256": content_hash(proposal), "validation_plan": proposal["validation_plan"],
                         "meaning": "Approved evaluation coverage addition; does not confirm the underlying exploratory hypothesis"}
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, amended)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--data", type=Path, default=ROOT / "data/processed/checklist/paired_cases.jsonl")
    run.add_argument("--playbook", type=Path, default=ROOT / "playbooks/sentiment_v1.json")
    run.add_argument("--output", type=Path)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--request", default="Compare model B against A using the scientist's playbook. Investigate individual regressions and unnamed input-text groups, distinguish harmless clustering from degradation, assess evaluation coverage, and draft a check addition if supported.")
    run.add_argument("--max-calls", type=int, default=24)
    run.add_argument("--token-budget", type=int, default=200000)
    run.add_argument("--max-seconds", type=int, default=600)
    run.add_argument("--replay", type=Path, help="Replay a recorded trace with no model requests")
    review = sub.add_parser("approve")
    review.add_argument("--proposal", type=Path, required=True)
    review.add_argument("--playbook", type=Path, default=ROOT / "playbooks/sentiment_v1.json")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "approve":
        print(approve(args.proposal, args.playbook, args.output, args.reviewer))
        return
    if args.max_calls < 1 or args.max_calls > 60 or args.token_budget < 1000 or args.max_seconds <= 0:
        parser.error("Invalid execution limits")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or ROOT / "artifacts" / ("replay-" if args.replay else "live-") / stamp
    if output.exists():
        parser.error("Output directory already exists; choose a new directory to preserve evidence")
    data = Dataset(args.data)
    playbook = json.loads(args.playbook.read_text(encoding="utf-8-sig"))
    for check in playbook["checks"]:
        jsonschema.validate(check["filter"], FILTER)
    provider = ReplayProvider(args.replay) if args.replay else OpenAIProvider(args.model)
    toolkit = AnalysisTools(data, playbook, output)
    save_json(output / "run_config.json", {"data_sha256": data.sha256, "playbook": playbook,
        "request": args.request, "model": provider.model, "provider": provider.kind})
    try:
        report = Agent(provider, toolkit, args.max_calls, args.token_budget, args.max_seconds).run(args.request)
    except Exception as error:
        print(f"Analysis incomplete ({type(error).__name__}): {error}")
        print(f"Saved diagnostic state: {output / 'state.json'}")
        raise SystemExit(1) from None
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
