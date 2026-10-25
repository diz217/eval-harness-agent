"""Deterministic numerical tools, evidence artifacts, and reviewable findings."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import jsonschema
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

from .schemas import SCHEMAS


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class AnalysisTools:
    def __init__(self, dataset, playbook, run_dir):
        self.data, self.playbook, self.run_dir = dataset, playbook, Path(run_dir)
        self.evidence, self.findings, self.proposals, self.seen = {}, [], [], set()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "evidence").mkdir(exist_ok=True)
        (self.run_dir / "proposals").mkdir(exist_ok=True)
        self.done = False
        self.vectorizer = self.matrix = None
        self.cluster_cache = {}

    def call(self, name, arguments):
        if name not in SCHEMAS:
            raise ValueError("Unknown tool")
        jsonschema.validate(arguments, SCHEMAS[name])
        result = getattr(self, "tool_" + name)(**arguments)
        self.seen.add(name)
        evidence_id = f"e{len(self.evidence) + 1:04d}"
        record = {"evidence_id": evidence_id, "tool": name, "arguments": arguments,
                  "dataset_sha256": self.data.sha256, "result": result}
        path = self.run_dir / "evidence" / (evidence_id + ".json")
        save_json(path, record)
        self.evidence[evidence_id] = record
        return {"evidence_id": evidence_id, "artifact": "evidence/" + path.name, "result": result}

    def tool_overview(self):
        return {"comparison": self.data.stats("all", bootstrap=True),
                "models": {role: self.data.rows[0][role]["model_id"] for role in ("baseline", "candidate")},
                "playbook": self.playbook,
                "limitations": ["Constructed behavioral tests; labels are author-specified expectations",
                                "Different models, not a controlled ablation or release history",
                                "Source test names are not loaded by these tools",
                                "Published score vectors are not assumed calibrated",
                                "Template dependence and adaptive discovery prevent confirmatory claims"]}

    def tool_run_prescribed_checks(self):
        checks = []
        for check in self.playbook["checks"]:
            group_id = "prescribed_" + check["id"]
            indices = self.data.select(check["filter"])
            self.data.register(group_id, indices, check["hypothesis"])
            checks.append({"check": check, "metrics": self.data.stats(group_id, bootstrap=True)})
        return {"checks": checks, "meaning": "These checks do not exhaust possible failures"}

    def tool_compare_slice(self, name, filter):
        if filter["min_words"] > filter["max_words"]:
            raise ValueError("min_words exceeds max_words")
        group_id = "slice_" + content_hash(filter)[:10]
        self.data.register(group_id, self.data.select(filter), name + ": " + json.dumps(filter))
        return {"name": name, "filter": filter, "metrics": self.data.stats(group_id, bootstrap=True)}

    def tool_cluster_cases(self, n_clusters):
        if n_clusters in self.cluster_cache:
            return self.cluster_cache[n_clusters]
        if len(self.cluster_cache) >= 3:
            raise ValueError("At most three clustering configurations per run")
        if n_clusters > len(set(self.data.texts)):
            raise ValueError("Too many clusters for unique inputs")
        if self.matrix is None:
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=8000, sublinear_tf=True)
            self.matrix = self.vectorizer.fit_transform(self.data.texts)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        with threadpool_limits(limits=1):
            assignments = model.fit_predict(self.matrix)
        terms = self.vectorizer.get_feature_names_out()
        clusters = []
        for i in range(n_clusters):
            group_id = f"cluster{n_clusters}_{i:02d}"
            top_terms = terms[np.argsort(model.cluster_centers_[i])[-8:][::-1]].tolist()
            self.data.register(group_id, np.flatnonzero(assignments == i), ", ".join(top_terms))
            clusters.append({"top_terms": top_terms, "metrics": self.data.stats(group_id)})
        clusters.sort(key=lambda c: c["metrics"].get("accuracy_delta", 0))
        result = {"method": "TF-IDF (word unigrams/bigrams) + KMeans over all input texts",
                  "selection": "No outcome or source-test label used in clustering",
                  "clusters": clusters,
                  "warning": "Terms describe lexical similarity, not cause; small and isolated failures are retained in the full dataset"}
        self.cluster_cache[n_clusters] = result
        return result

    def tool_inspect_cases(self, group_id, outcome, limit):
        indices = self.data.indices(group_id)
        if outcome == "regressed":
            indices = indices[self.data.delta[indices] == -1]
        elif outcome == "improved":
            indices = indices[self.data.delta[indices] == 1]
        elif outcome == "both_wrong":
            indices = indices[(self.data.a[indices] == 0) & (self.data.b[indices] == 0)]
        # Preserve source order; no hidden model-based representative selection.
        return {"group_id": group_id, "matching_cases": len(indices), "sampling": "first cases in input order",
                "cases": [self.data.case(i) for i in indices[:limit]]}

    def tool_check_coverage(self, group_id):
        if "run_prescribed_checks" not in self.seen:
            raise ValueError("Run prescribed checks first")
        covered = set()
        for check in self.playbook["checks"]:
            covered.update(self.data.groups["prescribed_" + check["id"]].tolist())
        indices = self.data.indices(group_id)
        uncovered = np.array([i for i in indices if i not in covered], dtype=int)
        uncovered_id = "uncovered_" + group_id
        self.data.register(uncovered_id, uncovered, "Outside prescribed slices within " + group_id)
        return {"group_id": group_id, "cases": len(indices), "covered_cases": len(indices) - len(uncovered),
                "uncovered": self.data.stats(uncovered_id, bootstrap=True),
                "interpretation": "Input-slice coverage only; a case outside these slices is not automatically a meaningful coverage defect"}

    def validate_evidence(self, evidence_ids):
        if not evidence_ids or any(e not in self.evidence for e in evidence_ids):
            raise ValueError("Provide existing evidence IDs")

    def reject_confirmation_claim(self, text):
        if re.search(r"\b(significant|proven|confirmed regression|verified root cause)\b", text, re.I):
            raise ValueError("No independent confirmation exists: avoid significance or certainty claims")

    def tool_record_finding(self, title, kind, group_id, case_ids, evidence_ids, hypothesis, missing_evidence):
        self.reject_confirmation_claim(title + " " + hypothesis)
        self.validate_evidence(evidence_ids)
        indices = self.data.indices(group_id)
        allowed_ids = {self.data.rows[i]["case_id"] for i in indices}
        if any(case_id not in allowed_ids for case_id in case_ids):
            raise ValueError("Finding case is absent from its group")
        observed = set()
        relevant = False
        for e in evidence_ids:
            record = self.evidence[e]
            result = record["result"]
            if record["tool"] == "inspect_cases":
                observed.update(c["case_id"] for c in result["cases"])
            def references_group(value):
                if isinstance(value, dict):
                    return value.get("group_id") == group_id or any(references_group(v) for v in value.values())
                if isinstance(value, list):
                    return any(references_group(v) for v in value)
                return False
            if references_group(result):
                relevant = True
        if not relevant or any(c not in observed for c in case_ids):
            raise ValueError("Cite group metrics and actual inspected cases as evidence")
        stats = self.data.stats(group_id, bootstrap=True)
        if kind == "exploratory_regression" and (stats["cases"] == 0 or stats["accuracy_delta"] >= 0):
            raise ValueError("This group does not show negative mean paired accuracy")
        if kind == "no_detected_group_regression" and (stats["cases"] == 0 or stats["accuracy_delta"] < -self.playbook["minimum_effect"]):
            raise ValueError("Cannot dismiss a group with observed degradation beyond the playbook effect threshold")
        if kind == "isolated_failure":
            if not case_ids or any(self.data.by_id[c]["baseline"]["label"] != self.data.by_id[c]["expected_label"]
                                   or self.data.by_id[c]["candidate"]["label"] == self.data.by_id[c]["expected_label"] for c in case_ids):
                raise ValueError("Isolated regression requires actual A-correct/B-wrong cases")
        finding = {"finding_id": f"f{len(self.findings) + 1:03d}", "title": title, "kind": kind,
                   "group_id": group_id, "measured_observations": stats, "case_ids": case_ids,
                   "evidence_ids": evidence_ids, "agent_hypothesis_not_verified_cause": hypothesis,
                   "missing_evidence": missing_evidence,
                   "confirmation_status": "unconfirmed", "business_severity": "not established"}
        self.findings.append(finding)
        save_json(self.run_dir / "findings.json", self.findings)
        return finding

    def tool_propose_playbook_addition(self, title, filter, rationale, evidence_ids, validation_plan):
        self.validate_evidence(evidence_ids)
        if not validation_plan.strip() or filter["min_words"] > filter["max_words"]:
            raise ValueError("A valid filter and confirmation/review plan are required")
        matching_evidence = [e for e in evidence_ids if self.evidence[e]["tool"] == "compare_slice"
                             and self.evidence[e]["arguments"]["filter"] == filter]
        if not matching_evidence:
            raise ValueError("Compare the exact proposed filter first and cite that compare_slice evidence")
        self.reject_confirmation_claim(title + " " + rationale)
        measured = self.evidence[matching_evidence[0]]["result"]["metrics"]
        matching = measured["cases"]
        proposal = {"proposal_id": f"p{len(self.proposals) + 1:03d}", "status": "draft_pending_scientist_review",
                    "base_playbook_sha256": content_hash(self.playbook), "base_playbook": self.playbook,
                    "title": title, "filter": filter, "rationale": rationale, "evidence_ids": evidence_ids,
                    "matching_cases_in_exploration_data": matching, "measured_observations": measured,
                    "validation_plan": validation_plan}
        self.proposals.append(proposal)
        save_json(self.run_dir / "proposals" / (proposal["proposal_id"] + ".json"), proposal)
        return proposal

    def tool_finish_report(self, summary, limitations, recommendation):
        self.reject_confirmation_claim(summary)
        required = {"overview", "run_prescribed_checks", "cluster_cases", "inspect_cases", "check_coverage"}
        if not required <= self.seen or len(self.findings) < 2:
            raise ValueError("Incomplete analysis: missing required tools or recorded findings")
        if recommendation == "no_regression_observed" and int((self.data.delta == -1).sum()):
            raise ValueError("Individual regressions exist; retain them for scientist review")
        report = {"status": "completed_exploratory_analysis", "dataset_sha256": self.data.sha256,
                  "playbook_id": self.playbook["id"], "playbook_sha256": content_hash(self.playbook),
                  "agent_summary_unverified_narrative": summary, "overall": self.data.stats("all", bootstrap=True),
                  "findings": self.findings, "proposals": self.proposals, "recommendation": recommendation,
                  "limitations": list(dict.fromkeys(limitations + ["No independent confirmation cohort",
                      "Template dependence and adaptive selection", "Narrative hypotheses are not verified root causes",
                      "Business severity was not supplied"])),
                  "evidence_index": {e: {"tool": r["tool"], "artifact": "evidence/" + e + ".json"} for e, r in self.evidence.items()}}
        save_json(self.run_dir / "report.json", report)
        lines = ["# Evaluation analysis report", "", "Status: exploratory; scientist review required.", "",
                 "## Agent interpretation", "", summary, "", "## Computed overall observations", "",
                 "```json", json.dumps(report["overall"], indent=2), "```", "", "## Findings", ""]
        for f in self.findings:
            lines += ["### " + f["finding_id"] + ": " + f["title"], "", "Kind: " + f["kind"], "",
                      "Agent hypothesis (unverified): " + f["agent_hypothesis_not_verified_cause"], "",
                      "```json", json.dumps(f["measured_observations"], indent=2), "```", "",
                      "Evidence: " + ", ".join(f"[{e}](evidence/{e}.json)" for e in f["evidence_ids"]), "",
                      "Cases: " + ", ".join(f["case_ids"]), "",
                      "Missing evidence: " + "; ".join(f["missing_evidence"]), ""]
        lines += ["## Proposed checks", ""]
        for p in self.proposals:
            lines += [f"- [{p['proposal_id']}: {p['title']}](proposals/{p['proposal_id']}.json) — draft pending review", ""]
        lines += ["## Limitations", ""] + ["- " + item for item in report["limitations"]]
        (self.run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
        self.done = True
        return {"status": report["status"], "report": "report.md", "findings": len(self.findings), "draft_proposals": len(self.proposals)}
