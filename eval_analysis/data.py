"""Data validation, safe subset selection, and descriptive paired statistics."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

LABELS = {"negative", "neutral", "positive"}


def validate_rows(rows):
    if not rows:
        raise ValueError("No paired cases")
    ids = set()
    model_ids = {"baseline": set(), "candidate": set()}
    for row in rows:
        if not isinstance(row.get("case_id"), str) or row["case_id"] in ids:
            raise ValueError("Case IDs must be unique strings")
        ids.add(row["case_id"])
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Each case requires nonempty text")
        if row.get("expected_label") not in LABELS:
            raise ValueError("Invalid expected label")
        for role in model_ids:
            prediction = row.get(role, {})
            if prediction.get("label") not in LABELS or not prediction.get("model_id"):
                raise ValueError("Missing paired model prediction or model identity")
            model_ids[role].add(prediction["model_id"])
            probabilities = prediction.get("probabilities", {})
            if set(probabilities) != LABELS:
                raise ValueError("Probability vector must contain all three labels")
            vals = list(probabilities.values())
            if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in vals):
                raise ValueError("Invalid probabilities")
            if abs(sum(vals) - 1) > 1e-4:
                raise ValueError("Probabilities must sum to one")
    if any(len(v) != 1 for v in model_ids.values()):
        raise ValueError("An input must compare exactly one baseline and one candidate")
    # Strip reference-only fields even if a caller accidentally adds them.
    return [{key: r[key] for key in ("case_id", "text", "expected_label", "baseline", "candidate")}
            for r in rows]


class Dataset:
    def __init__(self, path: Path):
        payload = path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.rows = validate_rows([json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()])
        self.by_id = {r["case_id"]: r for r in self.rows}
        self.a = np.array([r["baseline"]["label"] == r["expected_label"] for r in self.rows], dtype=float)
        self.b = np.array([r["candidate"]["label"] == r["expected_label"] for r in self.rows], dtype=float)
        self.delta = self.b - self.a
        self.texts = [r["text"] for r in self.rows]
        self.word_counts = np.array([len(t.split()) for t in self.texts])
        self.labels = np.array([r["expected_label"] for r in self.rows])
        self.groups = {"all": np.arange(len(self.rows), dtype=int)}
        self.group_descriptions = {"all": "All paired cases"}

    def select(self, condition):
        label = condition["expected_label"]
        mask = (self.word_counts >= condition["min_words"]) & (self.word_counts <= condition["max_words"])
        if label != "any":
            mask &= self.labels == label
        for term in condition["contains_all"]:
            if not term.strip():
                raise ValueError("Empty search terms are not allowed")
            mask &= np.array([term.casefold() in text.casefold() for text in self.texts])
        return np.flatnonzero(mask)

    def register(self, group_id, indices, description):
        self.groups[group_id] = np.asarray(indices, dtype=int)
        self.group_descriptions[group_id] = description

    def indices(self, group_id):
        if group_id not in self.groups:
            raise ValueError("Unknown group_id; use an ID returned by a tool")
        return self.groups[group_id]

    def stats(self, group_id, bootstrap=False):
        ix = self.indices(group_id)
        n = len(ix)
        if n == 0:
            return {"group_id": group_id, "cases": 0, "status": "inconclusive", "reason": "empty subset"}
        a, b, delta = self.a[ix], self.b[ix], self.delta[ix]
        regressions, improvements = int((delta == -1).sum()), int((delta == 1).sum())
        result = {
            "group_id": group_id, "cases": n,
            "baseline_accuracy": float(a.mean()), "candidate_accuracy": float(b.mean()),
            "accuracy_delta": float(delta.mean()), "correct_to_wrong": regressions,
            "wrong_to_correct": improvements, "both_wrong": int(((a == 0) & (b == 0)).sum()),
            "expected_label_counts": {label: int((self.labels[ix] == label).sum()) for label in sorted(LABELS)},
            "status": "exploratory", "confirmation": "No independent confirmation cohort was supplied",
            "uncertainty_flags": [],
        }
        if n < 30:
            result["uncertainty_flags"].append("small subset")
        if regressions + improvements < 10:
            result["uncertainty_flags"].append("few discordant pairs")
        result["uncertainty_flags"].append("unmeasured template dependence; adaptive slice selection")
        if bootstrap:
            # Group exact duplicate texts, but explicitly do not claim to solve
            # dependence between similar source templates.
            texts = [self.texts[i] for i in ix]
            unique = sorted(set(texts))
            lookup = {text: i for i, text in enumerate(unique)}
            group = np.array([lookup[t] for t in texts])
            sums = np.bincount(group, weights=delta)
            counts = np.bincount(group)
            rng = np.random.default_rng(42)
            samples = []
            for _ in range(400):
                pick = rng.integers(0, len(unique), len(unique))
                samples.append(float(sums[pick].sum() / counts[pick].sum()))
            result["descriptive_bootstrap_95"] = [float(x) for x in np.quantile(samples, [.025, .975])]
            result["interval_interpretation"] = "Exact-text-group bootstrap only; not selection-adjusted, not a confirmatory interval"
        return result

    def case(self, index):
        row = self.rows[index]
        return {**row, "accuracy_delta": float(self.delta[index]),
                "severity": "not annotated; do not infer business impact from model score"}
