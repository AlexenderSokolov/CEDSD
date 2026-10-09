"""Lossless, atomic epoch evidence controls; scratch outputs are retained.

Run: python -m unittest discover -s uica_exec/tests -p test_full_epoch_evidence.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from uica.full_training import _save_epoch_predictions


class _Budget:
    def __init__(self):
        self.guards = []

    def guard(self, required_bytes=0, force=False):
        self.guards.append((required_bytes, force))


class EpochEvidenceRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Intentionally retain every target and failed-write scratch file.
        cls.scratch = Path(tempfile.mkdtemp(prefix='uica-full-epoch-evidence-')).resolve()

    @classmethod
    def tearDownClass(cls):
        print(f'RETAINED_EPOCH_EVIDENCE_SCRATCH {cls.scratch}')

    def test_gzip_preserves_all_rows_and_exact_float_bits(self):
        folder = self.scratch / 'roundtrip'
        rows = [
            {'sample_id': '样本甲', 'label': 0, 'component_id': 'family1',
             'logits': [math.nextafter(1.0, 2.0), -0.0], 'margin': -math.nextafter(1.0, 2.0),
             'arm': 'log', 'seed': 17, 'data_role': 'validation', 'run_id': 'fixture-run'},
            {'sample_id': '样本乙', 'label': 1, 'component_id': 'family2',
             'logits': [-1e300, 1e300], 'margin': 2e300,
             'arm': 'log', 'seed': 17, 'data_role': 'validation', 'run_id': 'fixture-run'},
            {'sample_id': '样本丙', 'label': 1, 'component_id': 'family3',
             'logits': [5e-324, -5e-324], 'margin': -1e-323,
             'arm': 'log', 'seed': 17, 'data_role': 'validation', 'run_id': 'fixture-run'}]
        budget = _Budget()
        receipt = _save_epoch_predictions(folder, 7, rows, budget)
        path = folder / receipt['epoch_prediction_path']
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            restored = [json.loads(line) for line in stream]
        self.assertEqual(restored, rows)
        for before, after in zip(rows, restored):
            for left, right in zip(before['logits']+[before['margin']], after['logits']+[after['margin']]):
                self.assertEqual(struct.pack('<d', left), struct.pack('<d', right))
        blob = path.read_bytes()
        self.assertEqual(receipt['epoch_prediction_sha256'], hashlib.sha256(blob).hexdigest())
        self.assertEqual(receipt['epoch_prediction_n'], len(rows))
        self.assertEqual(receipt['epoch_prediction_bytes'], len(blob))
        self.assertTrue(budget.guards[0][1])
        self.assertGreaterEqual(budget.guards[0][0], len(gzip.decompress(blob)))
        # The gzip header excludes timestamps and temporary path names.
        same = _save_epoch_predictions(folder, 7, rows, budget)
        self.assertEqual(same['epoch_prediction_sha256'], receipt['epoch_prediction_sha256'])

    def test_failed_compression_retains_previous_target(self):
        folder = self.scratch / 'write-failure'
        budget = _Budget()
        receipt = _save_epoch_predictions(folder, 1, [{'sample_id': 'original', 'logits': [0., 1.]}], budget)
        path = folder / receipt['epoch_prediction_path']
        before = path.read_bytes()
        with patch.object(gzip.GzipFile, 'write', side_effect=OSError('injected gzip write failure')):
            with self.assertRaisesRegex(OSError, 'gzip write failure'):
                _save_epoch_predictions(folder, 1, [{'sample_id': 'replacement', 'logits': [2., 3.]}], budget)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(path.with_name(path.name+'.tmp').exists())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), receipt['epoch_prediction_sha256'])


if __name__ == '__main__':
    unittest.main()
