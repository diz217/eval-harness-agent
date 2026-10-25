"""Import published CheckList predictions without training or inference.

The legacy suite contains serialized functions. This reader replaces all external
symbols with inert records; it never imports CheckList, dill, or numpy or executes
an expectation function. Only MFT cases with explicit expected labels are exported.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path

REVISION = "4e6e5e33a26f30c20ed602b2050f6c73e123cc23"
SOURCE_BASE = f"https://raw.githubusercontent.com/marcotcr/checklist/{REVISION}"
ARCHIVE_SHA256 = "061524b310a641144d2fbccb9087634e5c8d54f32ce79885c7a1c398cadfd202"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
ALLOWED_SYMBOLS = {
    ("checklist.test_suite", "TestSuite"),
    ("checklist.test_types", "MFT"), ("checklist.test_types", "INV"),
    ("checklist.test_types", "DIR"), ("checklist.expect", "Expect"),
    ("checklist.expect", "iter_with_optional"),
    ("dill._dill", "_create_function"), ("dill._dill", "_load_type"),
    ("dill._dill", "_get_attr"), ("dill._dill", "_import_module"),
    ("dill._dill", "_create_cell"), ("numpy", "ndarray"), ("numpy", "dtype"),
}


class Inert:
    symbol = "inert"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args, obj.kwargs, obj.state = args, kwargs, None
        return obj

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __setstate__(self, state):
        self.state = state

    def __call__(self, *args, **kwargs):
        obj = Inert(*args, **kwargs)
        obj.symbol = "call:" + self.symbol
        return obj


class DataOnlyReader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("collections", "defaultdict"):
            return collections.defaultdict
        if (module, name) not in ALLOWED_SYMBOLS:
            raise pickle.UnpicklingError(f"Unapproved symbol: {module}.{name}")
        return type("Symbol", (Inert,), {"symbol": module + "." + name})


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def decode_indices(value):
    if not isinstance(value, Inert) or not isinstance(value.state, tuple):
        raise ValueError("Expected serialized index array")
    version, shape, dtype, fortran, payload = value.state
    if version != 1 or len(shape) != 1 or fortran:
        raise ValueError("Unsupported index array shape/layout")
    if not isinstance(dtype, Inert) or dtype.symbol != "numpy.dtype":
        raise ValueError("Unsupported index dtype")
    if dtype.args[0] != "i8" or dtype.state[1] not in ("<", ">"):
        raise ValueError("Expected explicit-endian int64 indices")
    if len(payload) != 8 * shape[0]:
        raise ValueError("Index array byte length mismatch")
    return struct.unpack(dtype.state[1] + "q" * shape[0], payload)


def decode_text(value):
    if isinstance(value, str):
        return value
    if (isinstance(value, Inert) and len(value.args) == 2
            and isinstance(value.args[0], Inert)):
        dtype, payload = value.args
        code = dtype.args[0]
        if dtype.symbol == "numpy.dtype" and code.startswith("U"):
            endian = dtype.state[1]
            if endian not in ("<", ">") or not isinstance(payload, bytes):
                raise ValueError("Invalid Unicode scalar")
            if len(payload) != int(code[1:]) * 4:
                raise ValueError("Unicode scalar length mismatch")
            return payload.decode("utf-32-le" if endian == "<" else "utf-32-be").rstrip("\0")
    raise ValueError("Unsupported text representation")


def parse_predictions(payload):
    rows = []
    for line_number, line in enumerate(payload.decode("utf8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Prediction line {line_number}: expected four fields")
        label = int(parts[0])
        probs = [float(x) for x in parts[1:]]
        if (label not in LABELS or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
                or abs(sum(probs) - 1) > 1e-4):
            raise ValueError(f"Prediction line {line_number}: invalid probability vector")
        if probs[label] < max(probs) - 1e-6:
            raise ValueError(f"Prediction line {line_number}: label/probability mismatch")
        rows.append((label, probs))
    return rows


def accuracy(rows, model):
    return sum(r[model]["label"] == r["expected_label"] for r in rows) / len(rows)


def summarize(rows):
    a, b = accuracy(rows, "baseline"), accuracy(rows, "candidate")
    return {
        "cases": len(rows), "baseline_accuracy": a, "candidate_accuracy": b,
        "candidate_minus_baseline_accuracy": b - a,
        "correct_to_wrong": sum(r["baseline"]["label"] == r["expected_label"]
                                and r["candidate"]["label"] != r["expected_label"] for r in rows),
        "wrong_to_correct": sum(r["baseline"]["label"] != r["expected_label"]
                                and r["candidate"]["label"] == r["expected_label"] for r in rows),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def write_jsonl(path, rows):
    with path.open("w", encoding="utf8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def import_archive(archive_path, output_dir):
    archive_bytes = archive_path.read_bytes()
    if sha256(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from pinned source; import refused")
    member_hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
        def read(name):
            member = archive.getmember("release_data/sentiment/" + name)
            if not member.isfile() or member.size > 20_000_000:
                raise ValueError("Invalid source member")
            payload = archive.extractfile(member).read()
            member_hashes[name] = sha256(payload)
            return payload
        inputs = read("tests_n500").decode("utf8").splitlines()
        predictions = {name: parse_predictions(read("predictions/" + name))
                       for name in ("bert", "roberta")}
        suite = DataOnlyReader(io.BytesIO(read("sentiment_suite.pkl"))).load()
    if any(len(p) != len(inputs) for p in predictions.values()):
        raise ValueError("Input and prediction row counts do not match")
    if suite.symbol != "checklist.test_suite.TestSuite":
        raise ValueError("Unexpected suite type")
    records, reference, excluded = [], [], collections.Counter()
    grouped = collections.defaultdict(list)
    seen_rows = set()
    for name, test in suite.state["tests"].items():
        start, end = suite.state["test_ranges"][name]
        state = test.state
        if test.symbol != "checklist.test_types.MFT":
            excluded["relational_test_without_single_case_ground_truth"] += end - start
            continue
        labels = state["labels"]
        if labels is None:
            excluded["custom_expectation_without_explicit_label"] += end - start
            continue
        indices = decode_indices(state["run_idxs"])
        if len(indices) != end - start:
            raise ValueError("Selected case count differs from source range")
        for offset, index in enumerate(indices):
            source_row = start + offset
            text = decode_text(state["data"][index])
            if text != inputs[source_row]:
                raise ValueError(f"Input alignment mismatch at source row {source_row}")
            expected = labels[index] if isinstance(labels, list) else labels
            if expected not in LABELS:
                raise ValueError("Unknown explicit label")
            if source_row in seen_rows:
                raise ValueError("Duplicate source row")
            seen_rows.add(source_row)
            case_id = f"checklist-sentiment-{source_row:06d}"
            row = {
                "case_id": case_id, "text": text, "expected_label": LABELS[expected],
                "duplicate_text_group": sha256(text.encode("utf8")),
                "source_row_1based": source_row + 1,
                "baseline": {"model_id": "checklist-published-bert"},
                "candidate": {"model_id": "checklist-published-roberta"},
            }
            for role, model in (("baseline", "bert"), ("candidate", "roberta")):
                predicted, probabilities = predictions[model][source_row]
                row[role].update(label=LABELS[predicted], probabilities=dict(zip(LABELS.values(), probabilities)))
            records.append(row)
            reference.append({"case_id": case_id, "source_test_name": name,
                              "capability": state["capability"], "source_data_index": index,
                              "source_expectation": "explicit-label MFT"})
            grouped[name].append(row)
    if not records:
        raise ValueError("No eligible paired cases")
    records.sort(key=lambda x: x["source_row_1based"])
    reference.sort(key=lambda x: x["case_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "paired_cases.jsonl", records)
    write_jsonl(output_dir / "reference_metadata.jsonl", reference)
    report = {
        "status": "descriptive data audit; not an agent analysis or confirmatory experiment",
        "models": {"baseline": "published BERT", "candidate": "published RoBERTa"},
        "comparison_type": "different published models, not a controlled ablation or release history",
        "label_source": "author-specified behavioral expectations; not independent natural-review annotations",
        "overall": summarize(records),
        "by_source_test_reference_only": {name: summarize(rows) for name, rows in grouped.items()},
        "source_prediction_rows": len(inputs),
        "excluded_rows": dict(excluded),
        "unique_texts": len({r["duplicate_text_group"] for r in records}),
        "label_counts": dict(collections.Counter(r["expected_label"] for r in records)),
    }
    if len(records) + sum(excluded.values()) != len(inputs):
        raise ValueError("Source row accounting is incomplete")
    write_json(output_dir / "data_audit.json", report)
    manifest = {
        "source_repository": "https://github.com/marcotcr/checklist",
        "source_revision": REVISION, "source_archive_url": SOURCE_BASE + "/release_data.tar.gz",
        "archive_sha256": ARCHIVE_SHA256, "member_sha256": member_hashes,
        "files": {name: sha256((output_dir / name).read_bytes()) for name in
                  ("paired_cases.jsonl", "reference_metadata.jsonl", "data_audit.json")},
        "selection": "All published explicit-label MFT cases, independent of predictions",
        "inference_performed": False, "training_performed": False,
        "probability_semantics": "Published three-way sentiment vectors, preserved verbatim; not claimed to be raw logits or calibrated uncertainty",
        "reference_access": "Do not expose reference_metadata or reference-only audit slices to discovery agent",
        "confirmation": "No untouched confirmation set is claimed; template and duplicate dependence must be addressed",
    }
    write_json(output_dir / "manifest.json", manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--download", action="store_true", help="Fetch the pinned public archive if absent")
    args = parser.parse_args()
    raw = args.root / "data" / "raw" / "checklist"
    archive = raw / "release_data.tar.gz"
    if not archive.exists():
        if not args.download:
            parser.error("Archive absent. Use --download to retrieve published predictions.")
        raw.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(SOURCE_BASE + "/release_data.tar.gz", headers={"User-Agent": "EvalAnalysisAgent-data-import"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(25_000_001)
        if len(payload) > 25_000_000 or sha256(payload) != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive fails size/checksum validation")
        archive.write_bytes(payload)
    report = import_archive(archive, args.root / "data" / "processed" / "checklist")
    print(json.dumps({"overall": report["overall"], "unique_texts": report["unique_texts"],
                      "source_tests": len(report["by_source_test_reference_only"]),
                      "excluded_rows": report["excluded_rows"]}, indent=2))


if __name__ == "__main__":
    main()
