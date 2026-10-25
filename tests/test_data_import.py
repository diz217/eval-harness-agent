"""Integrity checks for imported data and the non-executing legacy reader."""
import io
import json
import math
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import import_checklist as importer


class ImportSafetyTests(unittest.TestCase):
    def test_external_executable_symbol_is_rejected(self):
        payload = b'cos\nsystem\n(S\'must never execute\'\ntR.'
        with self.assertRaises(pickle.UnpicklingError):
            importer.DataOnlyReader(io.BytesIO(payload)).load()

    def test_prediction_probability_validation(self):
        for payload in (b'0 nan 0 1\n', b'0 .1 .1 .1\n', b'0 .1 .1 .8\n'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                importer.parse_predictions(payload)

    def test_neutral_predictions_are_preserved(self):
        self.assertEqual(importer.parse_predictions(b'1 0 .75 .25\n'), [(1, [0, .75, .25])])

    def test_changed_archive_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'bad.tar.gz'
            source.write_bytes(b'not the pinned archive')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                importer.import_archive(source, Path(tmp) / 'out')
            self.assertFalse((Path(tmp) / 'out').exists())


class ImportedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = ROOT / 'data' / 'processed' / 'checklist'
        if not (cls.base / 'manifest.json').exists():
            raise unittest.SkipTest('Run scripts/import_checklist.py first')
        cls.rows = [json.loads(x) for x in (cls.base / 'paired_cases.jsonl').read_text(encoding='utf8').splitlines()]

    def test_provenance_hashes_and_reference_join(self):
        manifest = json.loads((self.base / 'manifest.json').read_text(encoding='utf8'))
        for name, expected_hash in manifest['files'].items():
            self.assertEqual(importer.sha256((self.base / name).read_bytes()), expected_hash)
        ref = [json.loads(x) for x in (self.base / 'reference_metadata.jsonl').read_text(encoding='utf8').splitlines()]
        ids = [r['case_id'] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {r['case_id'] for r in ref})
        self.assertTrue(all('source_test_name' not in r and 'capability' not in r for r in self.rows))

    def test_all_ground_truth_classes_and_paired_transitions(self):
        report = json.loads((self.base / 'data_audit.json').read_text(encoding='utf8'))
        self.assertEqual({r['expected_label'] for r in self.rows}, {'negative', 'neutral', 'positive'})
        stats = importer.summarize(self.rows)
        net = stats['wrong_to_correct'] - stats['correct_to_wrong']
        self.assertTrue(math.isclose(stats['candidate_minus_baseline_accuracy'], net / len(self.rows)))
        self.assertEqual(stats, report['overall'])
        self.assertEqual(len(self.rows) + sum(report['excluded_rows'].values()), report['source_prediction_rows'])


if __name__ == '__main__':
    unittest.main()
