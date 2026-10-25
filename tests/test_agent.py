"""Acceptance tests use explicit synthetic fixtures, never fabricated model results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from eval_analysis.agent import Agent, BudgetExceeded
from eval_analysis.cli import approve
from eval_analysis.data import Dataset
from eval_analysis.provider import ReplayProvider
from eval_analysis.tools import AnalysisTools, save_json


ANY = {"expected_label": "any", "contains_all": [], "min_words": 0, "max_words": 1000}


def prediction(label):
    return {"model_id": "synthetic-test-system", "label": label,
            "probabilities": {x: float(x == label) for x in ('negative', 'neutral', 'positive')}}


def fixture_rows():
    rows = []
    for i in range(40):
        text = ('delayed delivery negative review ' if i < 10 else 'pleasant service positive review ') + str(i)
        gold = 'negative' if i < 10 else 'positive'
        rows.append({'case_id': f'fixture-{i}', 'text': text, 'expected_label': gold,
                     'baseline': prediction(gold), 'candidate': prediction('positive'),
                     'source_test_name': 'DO NOT EXPOSE THIS REFERENCE LABEL'})
    return rows


class ScriptedProvider:
    """Offline test double; not advertised as a live LLM or agent result."""
    kind, model = 'scripted_test', 'fixture'

    def __init__(self):
        self.turn = 0

    def respond(self, instructions, messages, tools, max_output_tokens, timeout):
        actions = [
            ('overview', {}), ('run_prescribed_checks', {}), ('cluster_cases', {'n_clusters': 2}),
            ('inspect_cases', {'group_id': 'all', 'outcome': 'regressed', 'limit': 2}),
            ('check_coverage', {'group_id': 'all'}),
            ('record_finding', {'title': 'Retain individual failures', 'kind': 'isolated_failure',
                'group_id': 'all', 'case_ids': ['fixture-0'], 'evidence_ids': ['e0001', 'e0004'],
                'hypothesis': 'Delivery wording warrants investigation; cause not established.',
                'missing_evidence': ['Independent confirmation', 'Severity annotation']}),
            ('compare_slice', {'name': 'Delayed wording', 'filter': {**ANY, 'contains_all': ['delayed']}}),
            ('record_finding', {'title': 'Insufficient confirmation', 'kind': 'inconclusive',
                'group_id': 'all', 'case_ids': [], 'evidence_ids': ['e0001'],
                'hypothesis': 'Template dependence remains unresolved.', 'missing_evidence': ['Independent cohort']}),
            ('propose_playbook_addition', {'title': 'Review delayed-delivery wording',
                'filter': {**ANY, 'contains_all': ['delayed']}, 'rationale': 'Fixture evidence warrants a draft check.',
                'evidence_ids': ['e0004', 'e0007'], 'validation_plan': 'Scientist review and an independent cohort.'}),
            ('finish_report', {'summary': 'Synthetic acceptance run only.',
                'limitations': ['Explicit test fixture'], 'recommendation': 'scientist_review'}),
        ]
        name, args = actions[self.turn]
        self.turn += 1
        return {'output': [{'type': 'function_call', 'call_id': f'call{self.turn}',
                            'name': name, 'arguments': json.dumps(args)}],
                'usage': {'total_tokens': 10}, 'status': 'completed', 'model': self.model}


class ToolAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'synthetic_fixture.jsonl'
        self.path.write_text('\n'.join(json.dumps(r) for r in fixture_rows()), encoding='utf8')
        self.book = {'id': 'fixture-v1', 'version': 1, 'minimum_effect': .01,
                     'checks': [{'id': 'short', 'hypothesis': 'Short-input stability',
                                 'filter': {**ANY, 'max_words': 1}}]}
        self.data = Dataset(self.path)
        self.tools = AnalysisTools(self.data, self.book, self.root / 'run')

    def tearDown(self):
        self.tmp.cleanup()

    def test_prescribed_regression_and_uncovered_failure(self):
        self.tools.call('run_prescribed_checks', {})
        result = self.tools.call('check_coverage', {'group_id': 'all'})['result']
        self.assertEqual(result['uncovered']['correct_to_wrong'], 10)
        self.assertEqual(result['covered_cases'], 0)
        stats = self.tools.call('compare_slice', {'name': 'Delivery wording',
            'filter': {**ANY, 'contains_all': ['delayed']}})['result']['metrics']
        self.assertEqual(stats['accuracy_delta'], -1)

    def test_known_regression_is_detected_by_prescribed_check(self):
        book = copy.deepcopy(self.book)
        book['checks'] = [{'id': 'delivery', 'hypothesis': 'Delivery must remain stable',
                           'filter': {**ANY, 'contains_all': ['delayed']}}]
        tools = AnalysisTools(self.data, book, self.root / 'known-check')
        result = tools.call('run_prescribed_checks', {})['result']['checks'][0]['metrics']
        self.assertEqual(result['cases'], 10)
        self.assertEqual(result['accuracy_delta'], -1)
        self.assertEqual(result['correct_to_wrong'], 10)

    def test_clustering_uses_all_inputs_without_reference_labels(self):
        self.assertTrue(all('source_test_name' not in r for r in self.data.rows))
        clusters = self.tools.call('cluster_cases', {'n_clusters': 2})['result']['clusters']
        self.assertEqual(sum(c['metrics']['cases'] for c in clusters), 40)
        self.assertTrue(any(c['metrics']['accuracy_delta'] < 0 for c in clusters))
        self.assertTrue(any(c['metrics']['accuracy_delta'] == 0 for c in clusters))

    def test_isolated_failure_and_invented_evidence(self):
        self.tools.call('overview', {})
        inspected = self.tools.call('inspect_cases', {'group_id': 'all', 'outcome': 'regressed', 'limit': 1})
        self.assertEqual(inspected['result']['cases'][0]['case_id'], 'fixture-0')
        args = {'title': 'Case retained', 'kind': 'isolated_failure', 'group_id': 'all',
                'case_ids': ['fixture-0'], 'evidence_ids': ['e0001', 'e0002'],
                'hypothesis': 'Unverified hypothesis', 'missing_evidence': ['Severity annotation']}
        self.tools.call('record_finding', args)
        bad = copy.deepcopy(args); bad['evidence_ids'] = ['invented-evidence']
        with self.assertRaises(ValueError): self.tools.call('record_finding', bad)
        bad = copy.deepcopy(args); bad['case_ids'] = ['fixture-39']
        with self.assertRaises(ValueError): self.tools.call('record_finding', bad)

    def test_harmless_group_can_be_dismissed_but_harmful_one_cannot(self):
        comparison = self.tools.call('compare_slice', {'name': 'Pleasant service',
            'filter': {**ANY, 'contains_all': ['pleasant']}})
        group = comparison['result']['metrics']['group_id']
        self.tools.call('record_finding', {'title': 'No detected group regression',
            'kind': 'no_detected_group_regression', 'group_id': group, 'case_ids': [],
            'evidence_ids': [comparison['evidence_id']], 'hypothesis': 'Lexical similarity alone is not harm.',
            'missing_evidence': ['Independent confirmation']})
        harmful = self.tools.call('compare_slice', {'name': 'Delayed service',
            'filter': {**ANY, 'contains_all': ['delayed']}})
        with self.assertRaises(ValueError):
            self.tools.call('record_finding', {'title': 'Incorrect dismissal',
                'kind': 'no_detected_group_regression', 'group_id': harmful['result']['metrics']['group_id'],
                'case_ids': [], 'evidence_ids': [harmful['evidence_id']], 'hypothesis': 'Invalid', 'missing_evidence': []})

    def test_empty_subset_is_inconclusive(self):
        result = self.tools.call('compare_slice', {'name': 'No examples',
            'filter': {**ANY, 'contains_all': ['absent phrase']}})['result']['metrics']
        self.assertEqual(result['status'], 'inconclusive')

    def test_tool_schema_prevents_unbounded_cluster_request(self):
        with self.assertRaises(jsonschema.ValidationError):
            self.tools.call('cluster_cases', {'n_clusters': 100000})

    def test_review_is_explicit_and_creates_new_version(self):
        evidence = self.tools.call('compare_slice', {'name': 'Delayed wording', 'filter': {**ANY, 'contains_all': ['delayed']}})
        self.tools.call('propose_playbook_addition', {'title': 'Delivery wording',
            'filter': {**ANY, 'contains_all': ['delayed']}, 'rationale': 'Synthetic fixture finding',
            'evidence_ids': [evidence['evidence_id']], 'validation_plan': 'Independent validation required'})
        proposal = self.tools.run_dir / 'proposals/p001.json'
        self.assertEqual(json.loads(proposal.read_text())['status'], 'draft_pending_scientist_review')
        original = self.root / 'book.json'; save_json(original, self.book)
        output = self.root / 'book_v2.json'
        approve(proposal, original, output, 'test-reviewer')
        self.assertEqual(json.loads(original.read_text())['version'], 1)
        self.assertEqual(json.loads(output.read_text())['version'], 2)
        with self.assertRaises(ValueError): approve(proposal, original, output, 'test-reviewer')

    def test_end_to_end_tool_loop_and_exact_offline_replay(self):
        provider = ScriptedProvider()
        report = Agent(provider, self.tools).run('Synthetic fixture acceptance run')
        self.assertTrue(report.exists())
        state = json.loads((self.tools.run_dir / 'state.json').read_text())
        self.assertEqual(state['live_api_calls'], 0)
        replay_tools = AnalysisTools(Dataset(self.path), self.book, self.root / 'replay')
        replay = ReplayProvider(self.tools.run_dir / 'trace.jsonl')
        replay_report = Agent(replay, replay_tools).run('Synthetic fixture acceptance run')
        self.assertEqual(report.read_text(), replay_report.read_text())

    def test_proposal_requires_measuring_its_actual_filter(self):
        e = self.tools.call('overview', {})
        with self.assertRaisesRegex(ValueError, 'exact proposed filter'):
            self.tools.call('propose_playbook_addition', {'title': 'Untested slice',
                'filter': {**ANY, 'contains_all': ['delayed']}, 'rationale': 'Unmeasured',
                'evidence_ids': [e['evidence_id']], 'validation_plan': 'Review'})

    def test_unsupported_statistical_certainty_is_rejected(self):
        e = self.tools.call('overview', {})
        with self.assertRaisesRegex(ValueError, 'certainty'):
            self.tools.call('record_finding', {'title': 'Statistically significant regression',
                'kind': 'exploratory_regression', 'group_id': 'all', 'case_ids': [],
                'evidence_ids': [e['evidence_id']], 'hypothesis': 'Unsupported', 'missing_evidence': []})

    def test_call_budget_preserves_incomplete_state(self):
        with self.assertRaises(BudgetExceeded):
            Agent(ScriptedProvider(), self.tools, max_calls=1).run('Fixture budget test')
        self.assertEqual(json.loads((self.tools.run_dir / 'state.json').read_text())['status'], 'incomplete')
        self.assertFalse((self.tools.run_dir / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
