"""Typed tool contracts exposed to the model; no filesystem or shell tools."""

def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def string(description=""):
    return {"type": "string", "description": description}


def strings(description=""):
    return {"type": "array", "items": {"type": "string"}, "description": description, "maxItems": 12}


FILTER = obj({
    "expected_label": {"type": "string", "enum": ["any", "negative", "neutral", "positive"]},
    "contains_all": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 100}, "maxItems": 5,
                     "description": "Case-insensitive literal substrings, all required. Not a regex or token boundary match."},
    "min_words": {"type": "integer", "minimum": 0, "maximum": 1000},
    "max_words": {"type": "integer", "minimum": 0, "maximum": 1000},
})
TOOL_DEFINITIONS = [
    ("overview", "Inspect aggregate paired results and dataset limitations.", obj({})),
    ("run_prescribed_checks", "Execute every scientist-specified playbook check with deterministic statistics.", obj({})),
    ("compare_slice", "Define a new input/label-based group and compute paired metrics. Selection is exploratory.",
     obj({"name": string(), "filter": FILTER})),
    ("cluster_cases", "Cluster ALL input texts with TF-IDF and KMeans, then compare A/B within each group. Does not use source categories or select cases by failure.",
     obj({"n_clusters": {"type": "integer", "minimum": 2, "maximum": 24}})),
    ("inspect_cases", "Inspect actual case evidence, including disagreements and isolated regressions. IDs come from previous tools.",
     obj({"group_id": string(), "outcome": {"type": "string", "enum": ["regressed", "improved", "both_wrong", "any"]},
          "limit": {"type": "integer", "minimum": 1, "maximum": 8}})),
    ("check_coverage", "Check which cases and regressed pairs in a group lie outside the prescribed input slices. A gap is not automatically a validated new failure mode.",
     obj({"group_id": string()})),
    ("record_finding", "Record an evidence-linked analytical finding. Numerical observations are filled in by code; narrative is a hypothesis, never a verified root cause.",
     obj({"title": string(), "kind": {"type": "string", "enum": ["exploratory_regression", "no_detected_group_regression", "isolated_failure", "inconclusive", "label_review"]},
          "group_id": string(), "case_ids": strings(), "evidence_ids": strings("Existing evidence IDs supporting this finding"),
          "hypothesis": string(), "missing_evidence": strings()})),
    ("propose_playbook_addition", "Save a draft new evaluation check, grounded in existing evidence. Cannot approve or modify the scientist's accepted playbook.",
     obj({"title": string(), "filter": FILTER, "rationale": string(), "evidence_ids": strings(), "validation_plan": string()})),
    ("finish_report", "Finish after prescribed checks, clustering, case inspection, coverage analysis, and recorded findings. Unsupported completion requests are rejected.",
     obj({"summary": string(), "limitations": strings(),
          "recommendation": {"type": "string", "enum": ["scientist_review", "insufficient_evidence", "no_regression_observed"]}})),
]
SCHEMAS = {name: schema for name, _, schema in TOOL_DEFINITIONS}
TOOLS = [{"type": "function", "name": name, "description": description, "parameters": schema, "strict": True}
         for name, description, schema in TOOL_DEFINITIONS]
