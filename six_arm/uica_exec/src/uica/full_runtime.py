"""Seven-day full-data execution contract and identity-bound FP32 inference.

The scientific model is the verified closeout model.  This module replaces only
the old data admission, cache addressing and execution budget.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .common import config_digest, file_sha256, read_json, read_jsonl, utc_now, write_json
from .data import collate_features, window_digest
from .metrics import binary_metrics, choose_threshold
from .training import to_device

ARMS = ['acoustic', 'ae', 'pooljoint', 'ca', 'linear', 'log']
SEEDS = [17, 29, 43]
QUEUE = [(a, s) for s in SEEDS for a in ARMS]
SMALL_QUEUE = [(a, s) for s in SEEDS for a in ['acoustic', 'ae', 'log']]
GIB = 1024**3
REPO = Path(__file__).resolve().parents[3]
INFERENCE = {'precision': 'fp32', 'tf32': False, 'cudnn_deterministic': True,
             'cudnn_benchmark': False, 'batch': 4, 'pad_samples': 80000,
             'order': 'sample_id_ascending', 'score': 'float64(logit_spoof-logit_real)',
             'll': 'logaddexp(0,margin)-label*margin'}


def run_identity():
    for parent in REPO.parents:
        if parent.parent.name == 'runs' and len(parent.name) == 36:
            return parent.name
    return os.environ.get('ORX_RUN_ID', 'local-lightweight-check')


def source_identity():
    # Native archives and Windows working trees may differ only in line endings.
    names = {'full_runtime.py', 'full_training.py', 'closeout_model.py',
             'closeout_pretrained.py', 'model.py', 'operators.py', 'data.py',
             'metrics.py', 'common.py', 'training.py'}
    import hashlib
    hashes = {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
              for p in sorted((REPO / 'uica_exec/src/uica').glob('*.py')) if p.name in names}
    if set(hashes) != names:
        raise ValueError('Full scientific source inventory is incomplete')
    return config_digest(hashes)


def science_identity(config):
    # Stage, native run ID, output paths and resume flags cannot change science.
    names = ('audio', 'text', 'model', 'training', 'models', 'selection', 'execution')
    return config_digest({k: config[k] for k in names if k in config})


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, config):
        self.config = config
        spec = config['full']
        self.root = Path(spec['output_root'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'state.json'
        started = float(spec['started_unix'])
        if not math.isfinite(started) or started <= 0:
            raise ValueError('A fixed real execution T0 is required')
        self.state = read_json(self.path) if self.path.exists() else {
            'started_unix': started, 'gpu_seconds': 0.0, 'fits': {}, 'sessions': [],
            'authorization': 'APPROVED_PLAN.md 2026-10-04 full-access',
            'output_root': str(self.root), 'run_ids': [], 'failures': []}
        if float(self.state['started_unix']) != started:
            raise ValueError('T0 cannot reset on resume or a new run')
        self.state['deadlines'] = {'train_stop': started + 144 * 3600,
                                   'delivery': started + 168 * 3600}
        self.state['minimum_free_bytes'] = 30 * GIB
        self.state['output_root'] = str(self.root)
        self.active = None
        self.last_disk = 0.0
        self.save()

    def save(self):
        self.state['updated_at'] = utc_now()
        write_json(self.path, self.state)

    def guard(self, required_bytes=0, force=False, training=False):
        now = time.time()
        if now >= self.state['deadlines']['delivery']:
            raise BudgetStop('T+168h delivery deadline')
        if training and now >= self.state['deadlines']['train_stop']:
            raise BudgetStop('T+144h training cutoff; save partial then evaluate')
        if force or required_bytes or now - self.last_disk >= 30:
            free = shutil.disk_usage(self.root).free
            self.state['disk'] = {'free_bytes': free, 'required_increment_bytes': int(required_bytes),
                                  'minimum_free_bytes': 30 * GIB, 'checked_at': utc_now()}
            self.last_disk = now
            self.save()
            if free - int(required_bytes) < 30 * GIB:
                raise BudgetStop(f'30GiB free-space boundary: need {int(required_bytes)} bytes; free {free}')
        return self.state.get('disk')

    def reconcile_previous(self, run_id, ended_unix, status, evidence):
        """Called only after the parent verified the exact native run terminal.

        A crashed process's wall time is conservatively charged.  PID existence
        alone and a different run's status are insufficient recovery evidence.
        """
        previous = self.state.get('active_session')
        if not previous:
            return
        if previous['run_id'] != run_id or status not in {'failed', 'cancelled', 'completed', 'stopped'} or not evidence:
            raise ValueError('Exact terminal native-run evidence is required for reconciliation')
        end = float(ended_unix)
        if not previous['started_unix'] <= end <= time.time() + 60:
            raise ValueError('Invalid native-run terminal timestamp')
        previous.update(ended_unix=end, elapsed_seconds=end-previous['started_unix'],
                        reconciled_at=utc_now(), native_status=status, evidence=evidence)
        self.state['sessions'].append(previous)
        self.state['gpu_seconds'] += previous['elapsed_seconds']
        self.state.pop('active_session')
        self.save()

    @contextmanager
    def gpu(self, stage):
        if self.state.get('active_session'):
            raise RuntimeError('Unreconciled native GPU run; verify exact original run before resume')
        self.guard(force=True)
        session = {'stage': stage, 'run_id': run_identity(), 'started_unix': time.time()}
        self.active = session
        self.state['active_session'] = session
        if session['run_id'] not in self.state['run_ids']:
            self.state['run_ids'].append(session['run_id'])
        self.save()
        try:
            yield
        finally:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            finally:
                end = time.time()
                session.update(ended_unix=end, elapsed_seconds=end-session['started_unix'])
                self.state['gpu_seconds'] += session['elapsed_seconds']
                self.state['sessions'].append(session)
                self.state.pop('active_session', None)
                self.active = None
                self.save()


@contextmanager
def execution_lock(config):
    """Kernel-owned serial lock shared by all native stages, no background service."""
    import fcntl
    root = Path(config['full']['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.execution.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another native full stage owns this output root') from error
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps({'pid': os.getpid(), 'run_id': run_identity(), 'task': config.get('task')}))
        lock.flush()
        yield


def _processing_science(processing):
    from copy import deepcopy
    value = deepcopy(processing)
    for spec in value.get('models', {}).values():
        for key in ('id', 'path', 'directory'):
            spec.pop(key, None)
    return value


class Dataset:
    """Manifest metadata in memory, one feature item at a time; no corpus tensor memo."""
    def __init__(self, manifest, cache, roles):
        self.records = sorted([r for r in read_jsonl(manifest) if r['role'] in set(roles)],
                              key=lambda r: r['sample_id'])
        if not self.records or len({r['sample_id'] for r in self.records}) != len(self.records):
            raise ValueError('Dataset role is empty or has duplicate sample IDs')
        self.cache_dir = Path(cache)
        index = read_json(self.cache_dir / 'cache_index.json')
        self.index = index.get('entries', index)
        report = read_json(self.cache_dir / 'preprocess.json')
        self.processing = report['processing']
        if config_digest(self.processing) != report['processing_fingerprint']:
            raise ValueError('Full cache processing receipt mismatch')
        self.processing_fingerprint = report['processing_fingerprint']
        for record in self.records:
            if record['sample_id'] not in self.index or record.get('component_id') in (None, ''):
                raise ValueError('Full dataset lacks a cache entry or frozen family ID')
        self.coverage_identity = config_digest([(r['sample_id'], r['label'], r['component_id']) for r in self.records])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        entry = self.index[record['sample_id']]
        if isinstance(entry, str):
            entry = {'path': entry}
        path = Path(entry['path'])
        if not path.is_absolute():
            path = self.cache_dir / path
        item = torch.load(path, map_location='cpu', weights_only=True)
        processing = entry.get('processing', self.processing)
        if _processing_science(processing) != _processing_science(self.processing):
            raise ValueError('Historical/new cache scientific processing differs')
        expected = entry.get('cache_identity', {
            'processing_fingerprint': config_digest(processing), 'window_sha256': record.get('window_sha256'),
            'content_sha256': record.get('content_sha256'), 'models': processing['models']})
        if expected.get('window_sha256') != record.get('window_sha256') or expected.get('content_sha256') != record.get('content_sha256'):
            raise ValueError('Indexed cache content/window differs from the new frozen manifest')
        if expected.get('processing_fingerprint') != config_digest(processing) or expected.get('models') != processing['models']:
            raise ValueError('Indexed cache processing identity mismatch')
        if not item.get('cache_complete') or item.get('sample_id') != record['sample_id'] or item.get('cache_identity') != expected:
            raise ValueError('Frozen feature cache identity mismatch: ' + record['sample_id'])
        waveform = torch.as_tensor(item['waveform'], dtype=torch.float32).cpu().numpy()
        if waveform.ndim != 1 or not 0 < len(waveform) <= 80000 or window_digest(waveform, 16000) != record['window_sha256']:
            raise ValueError('Cached waveform violates five-second audited window identity')
        if item['affective'].shape[-1] != 1024 or item['text'].shape[-1] != 768:
            raise ValueError('Frozen E2V/BERT feature dimensions differ')
        item.update(label=int(record['label']), record=record)
        return item


def datasets(budget, config, role, scale='full'):
    manifest = budget.root / ('primary_manifest.jsonl' if scale == 'full' or role != 'train' else 'small_manifest.jsonl')
    return Dataset(manifest, budget.root / 'cache', [role])


def collate(items):
    batch = collate_features(items)
    length = batch['waveforms'].shape[1]
    if length > 80000:
        raise ValueError('Input exceeds the fixed five-second window')
    batch['waveforms'] = torch.nn.functional.pad(batch['waveforms'], (0, 80000-length))
    return batch


def score_rows(rows, threshold=None):
    if not rows:
        raise ValueError('Cannot score an empty prediction set')
    labels = np.array([r['label'] for r in rows], dtype=np.int64)
    logits = np.array([r['logits'] for r in rows], dtype=np.float64)
    if logits.shape != (len(rows), 2) or not np.isfinite(logits).all():
        raise ValueError('Nonfinite or malformed FP32 logits')
    margin = logits[:, 1] - logits[:, 0]
    if threshold is None:
        threshold = choose_threshold(labels, margin)
    result = binary_metrics(labels, margin, threshold)
    result['ll'] = float(np.mean(np.logaddexp(0, margin) - labels*margin))
    result['fp'] = int(np.sum((labels == 0) & (margin >= threshold)))
    result['fn'] = int(np.sum((labels == 1) & (margin < threshold)))
    return result


def infer(model, dataset, arm, seed, checkpoint, budget):
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.float().eval()
    rows = []
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        for start in range(0, len(dataset), 4):
            budget.guard()
            batch = to_device(collate([dataset[i] for i in range(start, min(start+4, len(dataset)))]), 'cuda')
            logits = model(batch)['logits'].detach().float().cpu().numpy()
            for i, record in enumerate(batch['records']):
                values = logits[i].astype(np.float64)
                rows.append({'sample_id': record['sample_id'], 'data_role': record['role'],
                             'label': int(record['label']), 'component_id': record['component_id'],
                             'arm': arm, 'seed': seed, 'checkpoint_id': checkpoint,
                             'logits': [float(values[0]), float(values[1])], 'margin': float(values[1]-values[0]),
                             'inference_id': config_digest(INFERENCE), 'condition': 'full',
                             'source': record.get('source'), 'generator_id': record.get('generator_id'),
                             'audio_condition': record.get('condition'), 'fad_seen': record.get('fad_seen', record.get('seen')),
                             'run_id': run_identity()})
    validate_predictions(rows, dataset, arm, seed, checkpoint)
    return rows


predict = infer


def validate_predictions(rows, dataset, arm, seed, checkpoint):
    expected = [(r['sample_id'], r['label'], r['component_id']) for r in dataset.records]
    observed = [(r['sample_id'], r['label'], r['component_id']) for r in rows]
    if observed != expected or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Predictions do not cover the exact frozen role and sorted order')
    for row in rows:
        if (row['arm'], row['seed'], row['checkpoint_id'], row['inference_id']) != (arm, seed, checkpoint, config_digest(INFERENCE)):
            raise ValueError('Prediction checkpoint/inference identity mismatch')
        values = np.asarray(row['logits'], dtype=np.float64)
        if values.shape != (2,) or not np.isfinite(values).all() or float(values[1]-values[0]) != row['margin']:
            raise ValueError('Prediction finite logits/float64 margin identity mismatch')


def tensor_bytes(value):
    """Upper bound for incremental atomic checkpoint bytes including optimizer."""
    seen = set()
    def count(item):
        if isinstance(item, torch.Tensor):
            key = (str(item.device), item.untyped_storage().data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return item.untyped_storage().nbytes()
        if isinstance(item, dict):
            return sum(count(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(v) for v in item)
        return 0
    return count(value) + 16 * 1024**2


def atomic_save(value, path, budget):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The existing checkpoint remains on disk until replace; available space
    # must hold a whole additional serialized payload, including Adam moments.
    budget.guard(tensor_bytes(value), force=True)
    temporary = path.with_name(path.name + '.atomic')
    torch.save(value, temporary)
    os.replace(temporary, path)


def delta_payload(model, config, arm, seed, base_identity, manifest_identity, epoch, threshold, metrics, scale='full'):
    state = model.state_dict()
    keys = {n for n, p in model.named_parameters() if p.requires_grad}
    cache = Path(config['full']['output_root']) / 'cache/cache_index.json'
    return {'delta': {n: state[n].detach().float().cpu().clone() for n in sorted(keys)},
            'trainable_keys': sorted(keys), 'config': config, 'arm': arm, 'seed': seed, 'scale': scale,
            'base_identity': base_identity, 'manifest_identity': manifest_identity, 'epoch': epoch,
            'threshold': threshold, 'metrics': metrics, 'inference': INFERENCE,
            'source_identity': source_identity(), 'science_identity': science_identity(config),
            'cache_index_identity': file_sha256(cache)}


def load_delta(path, config, budget):
    from .closeout_model import CloseoutDetector
    saved = torch.load(path, map_location='cpu', weights_only=False)
    prepared = read_json(budget.root / 'prepared.json')
    if saved['base_identity'] != prepared['base_identity']:
        raise ValueError('Published XLS-R base snapshot identity mismatch')
    if saved['manifest_identity'] != file_sha256(budget.root / 'primary_manifest.jsonl'):
        raise ValueError('Checkpoint full manifest identity mismatch')
    if saved['cache_index_identity'] != file_sha256(budget.root / 'cache/cache_index.json'):
        raise ValueError('Frozen feature cache index changed')
    if saved['inference'] != INFERENCE or saved['source_identity'] != source_identity() or saved['science_identity'] != science_identity(config):
        raise ValueError('Checkpoint scientific source/config/inference identity mismatch')
    if saved.get('scale') == 'small' and saved.get('train_manifest_identity') != file_sha256(budget.root / 'small_manifest.jsonl'):
        raise ValueError('Checkpoint small-family subset changed')
    model = CloseoutDetector(config, variant=saved['arm'], seed=saved['seed'])
    state = model.state_dict()
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if expected != set(saved['delta']) or expected != set(saved['trainable_keys']):
        raise ValueError('Delta must contain all and only trainable tail layers and heads')
    if len({n.split('.')[3] for n in expected if n.startswith('acoustic_backbone.encoder.layers.')}) != 4:
        raise ValueError('Delta does not include exactly four XLS-R layers')
    for name, tensor in saved['delta'].items():
        if state[name].shape != tensor.shape or state[name].dtype != tensor.dtype or not torch.isfinite(tensor).all():
            raise ValueError('Delta tensor shape/dtype/finite mismatch')
        state[name] = tensor
    model.load_state_dict(state, strict=True)
    return model.float().cuda().eval(), saved


def engineering(config, budget):
    from .closeout_model import CloseoutDetector, engineering_operator_checks
    from .closeout_pretrained import published_state_negative_controls
    identity = read_json(budget.root / 'prepared.json')
    manifest_id = file_sha256(budget.root / 'primary_manifest.jsonl')
    if identity.get('manifest_identity', identity.get('manifest')) != manifest_id:
        raise ValueError('Prepared/frozen full manifest changed')
    tests = engineering_operator_checks()
    tests['published_loading_negative_controls'] = published_state_negative_controls()
    role_counts = Counter(r['role'] for r in read_jsonl(budget.root / 'primary_manifest.jsonl'))
    tests['input_binding'] = {'counts': dict(role_counts), 'manifest_identity': manifest_id,
                              'cache_index_identity': file_sha256(budget.root / 'cache/cache_index.json')}
    with budget.gpu('engineering_fp32_bf16_delta_reload'):
        data = datasets(budget, config, 'validation')
        probe_records = sorted([next(r for r in data.records if r['label'] == 0)] +
                               [r for r in data.records if r['label'] == 1][:3], key=lambda r: r['sample_id'])
        by_id = {r['sample_id']: i for i, r in enumerate(data.records)}
        batch = to_device(collate([data[by_id[r['sample_id']]] for r in probe_records]), 'cuda')
        model = CloseoutDetector(config, 'acoustic', 17).float().cuda()
        tests['published_initialization'] = model.acoustic_backbone.published_initialization
        if not tests['published_initialization']['all_loaded_tensors_equal_published']:
            raise ValueError('Incomplete published backbone initialization')
        changed_name, changed_parameter = next((n, p) for n, p in model.named_parameters()
                                               if n.startswith('acoustic_backbone.encoder.layers.') and p.requires_grad)
        original = changed_parameter.detach().cpu().clone()
        model.train()
        begin = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch)
            loss = torch.nn.functional.cross_entropy(output['logits'].float(), batch['labels'])
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
            raise ValueError('Engineering BF16 finite gradients failed')
        model.zero_grad(set_to_none=True)
        tests['real_microbatch_seconds'] = time.monotonic()-begin
        with torch.no_grad():
            changed_parameter.add_(1e-5)
        data.records = probe_records
        before = infer(model, data, 'acoustic', 17, 'engineering', budget)
        metrics = score_rows(before)
        payload = delta_payload(model, config, 'acoustic', 17, identity['base_identity'],
                                manifest_id, 0, metrics['threshold'], metrics)
        atomic_save(payload, budget.root / 'engineering_published_delta.pt', budget)
        del model, payload, changed_parameter, batch
        gc.collect()
        torch.cuda.empty_cache()
        restored, saved = load_delta(budget.root / 'engineering_published_delta.pt', config, budget)
        after = infer(restored, data, 'acoustic', 17, 'engineering', budget)
        difference = float(np.max(np.abs(np.asarray([r['logits'] for r in before])-np.asarray([r['logits'] for r in after]))))
        tail_changed = not torch.equal(saved['delta'][changed_name], original)
        if difference > 2e-5 or not tail_changed:
            raise ValueError('Fresh changed-tail FP32 delta reload failed')
        tests['fresh_delta_reload'] = {'n': len(before), 'max_logit_abs_diff': difference,
                                      'tail_changed': tail_changed, 'trainable_tensors': len(saved['delta'])}
        del restored, saved, data
        gc.collect()
        torch.cuda.empty_cache()
    write_json(budget.root / 'engineering_published.json', tests)
    print('PUBLISHED_INITIALIZATION_ENGINEERING ' + json.dumps(tests, ensure_ascii=False), flush=True)
    return tests
