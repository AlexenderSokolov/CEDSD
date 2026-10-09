"""CPU fault injection for full-fit recovery; writes no files and starts no GPU.

Run: python -m unittest discover -s uica_exec/tests -p test_full_resume.py
The detector/cache are small CPU fixtures. The actual trainer, weighted updates,
paired order, dropout, Adam, scheduler and resume cursor execute unchanged.
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from uica import full_training as training


class _Data:
    def __init__(self, records):
        self.records = records
        self.coverage_identity = 'fixed-fixture-coverage'

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {'sample_id': row['sample_id'], 'label': row['label'], 'record': row,
                'waveform': torch.ones(16)*(row['label']*2-1),
                'affective': torch.zeros(1, 1024), 'text': torch.zeros(2, 768),
                'text_present': True}


class _Model(torch.nn.Module):
    def __init__(self, config, arm, seed):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 2))
        self.dropout = torch.nn.Dropout(.1)
        self.acoustic_backbone = type('BackboneReceipt', (), {
            'published_initialization': {'all_loaded_tensors_equal_published': True}})()

    def cuda(self):
        return self

    def parameter_counts(self):
        return {'total': 2, 'trainable': 2, 'backbone_trainable': 2,
                'relation_branch': 0, 'classifier': 0}

    def trainable_parameter_groups(self, backbone_lr, head_lr):
        return [{'params': list(self.parameters()), 'lr': backbone_lr}]

    def forward(self, batch):
        return {'logits': self.dropout(batch['waveforms'][:, :1]) @ self.weight}


class _Budget:
    def __init__(self, root):
        self.root = Path(root)
        self.state = {'fits': {}, 'sessions': [], 'failures': [],
                      'deadlines': {'train_stop': 1e20, 'delivery': 1e20}}

    def guard(self, *args, **kwargs):
        pass

    def save(self):
        pass

    @contextlib.contextmanager
    def gpu(self, stage):
        yield


class _Harness:
    def __init__(self, fault=None):
        config_path = Path(__file__).resolve().parents[1] / 'configs/six_arm.example.json'
        self.config = json.loads(config_path.read_text(encoding='utf-8-sig'))
        self.config.pop('closeout', None)
        self.config['full'] = {'output_root': 'memory-only', 'started_unix': 1}
        self.config['task'] = {}
        self.train = [{'sample_id': f's{i:03d}', 'role': 'train', 'label': i % 2,
                       'component_id': f'c{i}'} for i in range(24)]
        self.dev = [{'sample_id': f'd{i}', 'role': 'validation', 'label': i % 2,
                     'component_id': f'dc{i}'} for i in range(4)]
        self.tensors, self.json, self.lines = {}, {}, {}
        self.present = set()
        self.fault, self.triggered, self.optimizer_steps = fault, False, 0

    @staticmethod
    def key(path):
        return Path(path).as_posix()

    def read_json(self, path):
        if Path(path).name == 'engineering_published.json':
            return {'published_initialization': {'all_loaded_tensors_equal_published': True}}
        if Path(path).name == 'prepared.json':
            return {'base_identity': 'base', 'manifest_identity': 'sha-primary_manifest.jsonl'}
        return copy.deepcopy(self.json[self.key(path)])

    def read_jsonl(self, path):
        if Path(path).name == 'primary_manifest.jsonl':
            return self.train + self.dev
        return copy.deepcopy(self.lines[self.key(path)])

    def write_json(self, path, value):
        self.json[self.key(path)] = copy.deepcopy(value)
        self.present.add(self.key(path))

    def write_jsonl(self, path, value):
        if self.fault in {'predictions', 'predictions_and_last'} and not self.triggered and Path(path).name == 'best_dev_predictions.jsonl':
            self.triggered = True
            raise OSError('injected selected-prediction write failure')
        self.lines[self.key(path)] = copy.deepcopy(value)
        self.present.add(self.key(path))

    def append(self, path, value):
        self.lines.setdefault(self.key(path), []).append(copy.deepcopy(value))
        self.present.add(self.key(path))

    def save_epoch_predictions(self, folder, epoch, predictions, budget):
        relative = f'epoch_predictions/epoch_{epoch:02d}.jsonl.gz'
        path = Path(folder) / relative
        self.lines[self.key(path)] = copy.deepcopy(predictions)
        self.present.add(self.key(path))
        return {'epoch_prediction_path': relative, 'epoch_prediction_sha256': 'fixture-epoch-sha',
                'epoch_prediction_n': len(predictions), 'epoch_prediction_bytes': 0}

    def atomic_save(self, value, path, budget):
        if self.fault == 'predictions_and_last' and self.triggered and Path(path).name == 'last.pt':
            raise OSError('injected recovery-checkpoint write failure')
        if self.fault == 'best' and not self.triggered and Path(path).name == 'best.pt':
            self.triggered = True
            raise OSError('injected selected-checkpoint write failure')
        self.tensors[self.key(path)] = copy.deepcopy(value)
        self.present.add(self.key(path))

    def torch_load(self, path, *args, **kwargs):
        return copy.deepcopy(self.tensors[self.key(path)])

    def load_delta(self, path, config, budget):
        saved = self.torch_load(path)
        model = _Model(config, saved['arm'], saved['seed'])
        model.load_state_dict(saved['delta'])
        return model, saved

    def infer(self, model, data, arm, seed, checkpoint, budget):
        model.eval()
        rows = []
        with torch.inference_mode():
            for index, record in enumerate(data.records):
                logits = model({'waveforms': data[index]['waveform'][None]})['logits'][0].tolist()
                rows.append({**record, 'data_role': record['role'], 'logits': logits,
                             'margin': float(logits[1])-float(logits[0]), 'arm': arm,
                             'seed': seed, 'checkpoint_id': checkpoint})
        return rows

    @contextlib.contextmanager
    def patches(self):
        real_tensor, real_step = torch.tensor, torch.optim.AdamW.step
        def cpu_tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return real_tensor(*args, **kwargs)
        def optimizer_step(optimizer, *args, **kwargs):
            result = real_step(optimizer, *args, **kwargs)
            self.optimizer_steps += 1
            if self.fault == 'optimizer' and not self.triggered and self.optimizer_steps == 3:
                self.triggered = True
                # Some parameters/moments already changed before an update fails.
                with torch.no_grad():
                    optimizer.param_groups[0]['params'][0].add_(1e6)
                raise RuntimeError('injected partially updated optimizer failure')
            return result
        replacements = [
            (Path, 'mkdir', lambda *args, **kwargs: None),
            (Path, 'exists', lambda path: self.key(path) in self.present),
            (training, '_append', self.append), (training, 'read_json', self.read_json),
            (training, '_save_epoch_predictions', self.save_epoch_predictions),
            (training, 'read_jsonl', self.read_jsonl), (training, 'write_json', self.write_json),
            (training, 'write_jsonl', self.write_jsonl),
            (training, 'file_sha256', lambda path: 'sha-'+Path(path).name),
            (training, 'datasets', lambda budget, config, role, scale='full':
             _Data(self.train if role == 'train' else self.dev)),
            (training, 'CloseoutDetector', _Model), (training, 'atomic_save', self.atomic_save),
            (training, 'load_delta', self.load_delta), (training, 'infer', self.infer),
            (training, 'to_device', lambda batch, device: batch),
            (torch, 'load', self.torch_load), (torch, 'tensor', cpu_tensor),
            (torch, 'autocast', lambda *args, **kwargs: contextlib.nullcontext()),
            (torch.optim.AdamW, 'step', optimizer_step),
            (torch.cuda, 'is_available', lambda: False),
            (torch.cuda, 'empty_cache', lambda: None),
            (torch.cuda, 'max_memory_allocated', lambda: 0)]
        with contextlib.ExitStack() as stack:
            for target, name, value in replacements:
                stack.enter_context(patch.object(target, name, value))
            stack.enter_context(patch('uica.full_runtime.file_sha256', lambda path: 'sha-'+Path(path).name))
            stack.enter_context(patch('builtins.print'))
            yield

    def run(self, root, resume=False):
        config = copy.deepcopy(self.config)
        config['task'] = {'resume': resume}
        with self.patches():
            return training.train(config, _Budget(root), 'acoustic', 17)

    def last(self, root):
        return self.tensors[f'{root}/fits/full/acoustic_seed17/last.pt']


class FullResumeRegression(unittest.TestCase):
    def assert_same_finish(self, harness, root):
        restored = harness.run(root, resume=True)
        uninterrupted = harness.run('reference')
        a, b = harness.last(root), harness.last('reference')
        self.assertEqual(restored['status'], 'complete')
        self.assertEqual(restored['selected_epoch'], uninterrupted['selected_epoch'])
        self.assertEqual(a['global_step'], b['global_step'])
        self.assertEqual(a['resume_epoch'], b['resume_epoch'])
        self.assertTrue(all(torch.equal(a['delta'][key], b['delta'][key]) for key in a['delta']))
        self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))

    def test_optimizer_fault_retains_preceding_safe_boundary(self):
        harness = _Harness('optimizer')
        with self.assertRaisesRegex(RuntimeError, 'partially updated optimizer'):
            harness.run('interrupted')
        saved = harness.last('interrupted')
        self.assertTrue(saved['resume_checkpoint_usable'])
        self.assertEqual((saved['resume_epoch'], saved['next_batch_index'], saved['global_step']), (2, 0, 2))
        self.assertLess(float(saved['delta']['weight'].abs().max()), 10)
        result = harness.json['interrupted/fits/full/acoustic_seed17/result.json']
        self.assertTrue(result['resumable'])
        self.assertFalse(result['current_state_safe'])
        self.assert_same_finish(harness, 'interrupted')

    def test_selection_pair_fault_replays_dev_before_committing_rank(self):
        for fault in ('best', 'predictions'):
            with self.subTest(fault=fault):
                harness = _Harness(fault)
                with self.assertRaisesRegex(OSError, 'write failure'):
                    harness.run('interrupted')
                saved = harness.last('interrupted')
                self.assertEqual((saved['resume_epoch'], saved['next_batch_index'], saved['global_step']), (1, 24, 2))
                self.assertTrue(all(value == float('inf') for value in saved['best_rank']))
                self.assertEqual(saved['bad'], 0)
                self.assertIsNone(saved['metrics'])
                self.assertFalse(saved['selection_pair_complete'])
                result = harness.json['interrupted/fits/full/acoustic_seed17/result.json']
                self.assertTrue(result['resumable'])
                self.assertFalse(result['checkpoint_usable'])
                self.assert_same_finish(harness, 'interrupted')

    def test_stale_usable_result_cannot_hide_incomplete_new_selection(self):
        harness = _Harness('predictions_and_last')
        path = 'interrupted/fits/full/acoustic_seed17/result.json'
        harness.json[path] = {'status':'failed', 'checkpoint_usable':True,
                              'checkpoint_id':'sha-best.pt', 'run_id':'previous-run',
                              'selection_pair_complete':True, 'resumable':False}
        harness.present.add(path)
        with self.assertRaisesRegex(OSError, 'recovery-checkpoint write failure'):
            harness.run('interrupted')
        failure = harness.json[path]
        self.assertFalse(failure['checkpoint_usable'])
        self.assertFalse(failure['selection_pair_complete'])
        self.assertTrue(failure['resumable'])
        self.assertNotEqual(failure['run_id'], 'previous-run')


if __name__ == '__main__':
    unittest.main()
