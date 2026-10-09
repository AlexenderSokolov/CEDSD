"""Full-pool six-arm fits with paired order and resumable effective-batch cursors."""
from __future__ import annotations

import gc
import gzip
import json
import math
import os
import time
import traceback

import numpy as np
import torch

from .common import canonical_json, config_digest, file_sha256, read_json, read_jsonl, utc_now, write_json, write_jsonl
from .closeout_model import CloseoutDetector
from .metrics import weighted_ce_sum
from .training import restore_rng, rng_state, seed_everything, to_device
from .full_runtime import (ARMS, SEEDS, QUEUE, SMALL_QUEUE, BudgetStop, INFERENCE,
                           atomic_save, collate, datasets, delta_payload, infer,
                           load_delta, run_identity, score_rows, source_identity)


def fit_key(arm, seed, scale='full'):
    return f'{scale}_{arm}_seed{seed}'


def paired_order(seed, epoch, n):
    return np.random.default_rng(seed*100000+epoch).permutation(n)


def _fixed_training(settings):
    fixed = {'max_epochs': 40, 'min_epochs': 5, 'patience': 7, 'effective_batch': 16,
             'microbatch': 4, 'backbone_lr': 1e-5, 'head_lr': .001,
             'weight_decay': .0001, 'warmup_fraction': .05, 'grad_clip': 1.0,
             'precision': 'bf16'}
    for name, expected in fixed.items():
        if settings.get(name) != expected:
            raise ValueError(f'Full training retains verified training.{name}={expected}')


def _append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def _save_epoch_predictions(folder, epoch, predictions, budget):
    """Keep every development logit losslessly for independent model selection."""
    path = folder / 'epoch_predictions' / f'epoch_{epoch:02d}.jsonl.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b''.join((canonical_json(row)+'\n').encode('utf-8') for row in predictions)
    # Reserve the uncompressed size plus more than deflate's worst expansion.
    # The previous target remains intact until the complete gzip is replaced.
    required = len(raw) + max(64*1024, math.ceil(len(raw)*.01))
    budget.guard(required, force=True)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream,
                           compresslevel=6, mtime=0) as compressed:
            compressed.write(raw)
    os.replace(temporary, path)
    return {'epoch_prediction_path': path.relative_to(folder).as_posix(),
            'epoch_prediction_sha256': file_sha256(path),
            'epoch_prediction_n': len(predictions), 'epoch_prediction_bytes': path.stat().st_size}


def train(config, budget, arm, seed, scale='full'):
    if arm not in ARMS or seed not in SEEDS or scale not in {'full', 'small'}:
        raise ValueError('Fit is outside the fixed full18 / conditional small9 matrix')
    if scale == 'small' and arm not in {'acoustic', 'ae', 'log'}:
        raise ValueError('Only Acoustic/AE/Log have an authorized small-scale comparison')
    budget.guard(training=True, force=True)
    if (budget.root / 'evaluation_lock.json').exists():
        raise RuntimeError('Final evaluation already froze the complete model matrix')
    engineering = read_json(budget.root / 'engineering_published.json')
    if engineering.get('published_initialization', {}).get('all_loaded_tensors_equal_published') is not True:
        raise RuntimeError('Strict published initialization engineering is not verified')
    _fixed_training(config['training'])
    key = fit_key(arm, seed, scale)
    queue = QUEUE if scale == 'full' else SMALL_QUEUE
    earlier = queue[:queue.index((arm, seed))]
    if any(budget.state['fits'].get(fit_key(a, s, scale), {}).get('status') != 'complete' for a, s in earlier):
        raise ValueError('Prescribed seed-block queue requires earlier fits completed or explicitly resolved by the parent')
    if scale == 'small':
        if any(budget.state['fits'].get(fit_key(a, s), {}).get('status') != 'complete' for a, s in QUEUE):
            raise RuntimeError('Full18 takes priority; all full fits must finish before small9')
        if not budget.state.get('small_scale_authorization', {}).get('all_nine_fit_budget_verified'):
            raise RuntimeError('Parent must verify budget for all small9 and final evaluation before starting small fits')
    folder = budget.root / 'fits' / scale / f'{arm}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'result.json'
    if result_path.exists() and read_json(result_path)['status'] == 'complete':
        raise RuntimeError('Completed fits are immutable and cannot be rerun')
    attempted = (folder / 'attempts.jsonl').exists()
    resume = bool(config.get('task', {}).get('resume', False))
    if attempted and not resume:
        raise RuntimeError('Existing fit requires an explicit diagnosed native-run resume')
    if resume and not (folder / 'last.pt').exists():
        raise RuntimeError('No resumable last checkpoint exists')
    if budget.state.get('active_session'):
        raise RuntimeError('Verify/reconcile the exact original native run before resume')
    data = datasets(budget, config, 'train', scale)
    dev = datasets(budget, config, 'validation')
    full_records = read_jsonl(budget.root / 'primary_manifest.jsonl')
    expected_full_ids = sorted(r['sample_id'] for r in full_records if r['role'] == 'train')
    observed_ids = [r['sample_id'] for r in data.records]
    if scale == 'full' and observed_ids != expected_full_ids:
        raise ValueError('Full loader does not cover every qualified frozen training record')
    counts = np.bincount([r['label'] for r in data.records], minlength=2)
    if not counts.all() or len({r['label'] for r in dev.records}) != 2:
        raise ValueError('Train and development must both contain real and spoof')
    manifest_identity = file_sha256(budget.root / 'primary_manifest.jsonl')
    train_manifest_identity = file_sha256(budget.root / ('primary_manifest.jsonl' if scale == 'full' else 'small_manifest.jsonl'))
    prepared = read_json(budget.root / 'prepared.json')
    if prepared.get('manifest_identity', prepared.get('manifest')) != manifest_identity:
        raise ValueError('Prepared full manifest changed before fit')
    attempt = {'arm': arm, 'seed': seed, 'scale': scale, 'run_id': run_identity(),
               'started_at': utc_now(), 'resume': resume, 'source_identity': source_identity(),
               'train_n': len(data), 'train_counts': counts.tolist(),
               'coverage_identity': data.coverage_identity, 'manifest_identity': manifest_identity,
               'train_manifest_identity': train_manifest_identity}
    _append(folder / 'attempts.jsonl', attempt)
    write_json(folder / 'loader_coverage.json', {**attempt, 'complete_frozen_train': scale == 'full',
                                               'sample_ids': observed_ids})
    print('FULL_LOADER_COVERAGE '+json.dumps({k: v for k, v in attempt.items() if k != 'source_identity'}), flush=True)
    seed_everything(seed)
    began = time.time()
    previous_seconds = sum(s['elapsed_seconds'] for s in budget.state['sessions'] if s['stage'] == 'train_'+key)
    settings = config['training']
    model = optimizer = scheduler = None
    stored_last_usable = False
    last_safe_boundary = None
    checkpoint_usable = True
    selection_pair_complete = True
    stop = 'max_epochs'
    interruption = None
    with budget.gpu('train_'+key):
        budget.state['fits'][key] = {**attempt, 'status': 'running'}
        budget.save()
        try:
            model = CloseoutDetector(config, arm, seed).float().cuda()
            write_json(folder / 'published_initialization.json', model.acoustic_backbone.published_initialization)
            parameter_counts = model.parameter_counts()
            parameter_counts['branch'] = parameter_counts['relation_branch']
            write_json(folder / 'parameter_counts.json', parameter_counts)
            optimizer = torch.optim.AdamW(model.trainable_parameter_groups(settings['backbone_lr'], settings['head_lr']),
                                          weight_decay=settings['weight_decay'])
            weights = torch.tensor(len(data)/(2*counts), dtype=torch.float32, device='cuda')
            steps = math.ceil(len(data)/16)*40
            warmup = max(1, round(steps*.05))
            def schedule(step):
                if step < warmup:
                    return (step+1)/warmup
                return .5*(1+math.cos(math.pi*min((step-warmup)/max(1, steps-warmup), 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
            epoch, begin_index, bad, best_rank, global_step = 1, 0, 0, (float('inf'),)*3, 0
            numerator_sum = denominator_sum = 0.0
            last_metrics = None
            resume_rng = rng_state()
            if resume:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model, saved = load_delta(folder / 'last.pt', config, budget)
                if saved.get('train_manifest_identity') != train_manifest_identity or saved.get('train_coverage_identity') != data.coverage_identity:
                    raise ValueError('Resume training sample identity differs')
                optimizer = torch.optim.AdamW(model.trainable_parameter_groups(settings['backbone_lr'], settings['head_lr']),
                                              weight_decay=settings['weight_decay'])
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
                optimizer.load_state_dict(saved['optimizer'])
                scheduler.load_state_dict(saved['scheduler'])
                restore_rng(saved['rng'])
                epoch, begin_index = int(saved['resume_epoch']), int(saved['next_batch_index'])
                bad, best_rank, global_step = saved['bad'], tuple(saved['best_rank']), saved['global_step']
                numerator_sum, denominator_sum = saved['epoch_numerator_sum'], saved['epoch_denominator_sum']
                last_metrics = saved['metrics']
                selection_pair_complete = saved.get('selection_pair_complete', True)
                resume_rng = saved['rng']
                if saved.get('resume_checkpoint_usable') is not True or not 0 <= begin_index <= len(data):
                    raise ValueError('Saved effective-batch cursor is not safely resumable')
                stored_last_usable = True
                last_safe_boundary = {'resume_epoch': epoch, 'next_batch_index': begin_index,
                                      'global_step': global_step}
                del saved
            # Space for a whole additional Adam checkpoint at atomic replacement.
            trainable_bytes = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
            checkpoint_headroom = 3*trainable_bytes + 32*1024**2
            budget.guard(checkpoint_headroom, force=True)
            last_checkpoint_at = time.time()
            next_epoch, next_batch_index = epoch, begin_index
            checkpoint_usable = True
            def save_last():
                nonlocal stored_last_usable, last_safe_boundary
                if not checkpoint_usable:
                    raise RuntimeError('An interrupted optimizer update may not replace the safe last checkpoint')
                payload = delta_payload(model, config, arm, seed, prepared['base_identity'], manifest_identity,
                                        epoch, None if last_metrics is None else last_metrics['threshold'], last_metrics, scale)
                payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=resume_rng,
                               resume_epoch=next_epoch, next_batch_index=next_batch_index,
                               bad=bad, best_rank=best_rank, global_step=global_step,
                               epoch_numerator_sum=numerator_sum, epoch_denominator_sum=denominator_sum,
                               train_n=len(data), train_coverage_identity=data.coverage_identity,
                               train_manifest_identity=train_manifest_identity,
                               resume_checkpoint_usable=checkpoint_usable,
                               selection_pair_complete=selection_pair_complete,
                               fit_seconds=previous_seconds+time.time()-began)
                atomic_save(payload, folder / 'last.pt', budget)
                stored_last_usable = True
                last_safe_boundary = {'resume_epoch': next_epoch, 'next_batch_index': next_batch_index,
                                      'global_step': global_step}
            def guard_fit():
                budget.guard(checkpoint_headroom, training=True)
                # Leave time to safely serialize rather than start another update
                # whose CUDA work could cross the absolute 144-hour boundary.
                if time.time() >= budget.state['deadlines']['train_stop']-120:
                    raise BudgetStop('Training cutoff safety interval for safe checkpoint save')
            print('FIT_CONFIG '+json.dumps({'arm': arm, 'seed': seed, 'scale': scale, 'training': settings,
                                            'train_n': len(data), 'train_counts': counts.tolist(), 'dev_n': len(dev),
                                            'parameters': parameter_counts, 'inference': INFERENCE,
                                            'resume_epoch': epoch, 'next_batch_index': begin_index,
                                            'run_id': run_identity()}), flush=True)
            # First-step failures also need a valid published-initialization
            # restore point. Existing verified resume checkpoints stay intact.
            if not stored_last_usable:
                save_last()
            try:
                if epoch > 5 and bad >= 7:
                    stop = 'early_stopping'
                else:
                    for epoch in range(epoch, 41):
                        epoch_started = time.time()
                        model.train()
                        order = paired_order(seed, epoch, len(data))
                        next_epoch = epoch
                        if begin_index == 0:
                            numerator_sum = denominator_sum = 0.0
                        for begin in range(begin_index, len(order), 16):
                            next_batch_index = begin
                            guard_fit()
                            resume_rng = rng_state()
                            indices = order[begin:begin+16]
                            items = [data[int(i)] for i in indices]
                            denominator = weights[torch.tensor([i['label'] for i in items], device='cuda')].sum()
                            optimizer.zero_grad(set_to_none=True)
                            batch_numerator = 0.0
                            for micro in range(0, len(items), 4):
                                batch = to_device(collate(items[micro:micro+4]), 'cuda')
                                with torch.autocast('cuda', dtype=torch.bfloat16):
                                    output = model(batch)
                                    loss = weighted_ce_sum(output['logits'].float(), batch['labels'], weights)
                                if not torch.isfinite(loss):
                                    raise ValueError('Nonfinite BF16 training weighted CE')
                                (loss/denominator).backward()
                                batch_numerator += float(loss.detach())
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                            checkpoint_usable = False
                            optimizer.step()
                            scheduler.step()
                            checkpoint_usable = True
                            global_step += 1
                            numerator_sum += batch_numerator
                            denominator_sum += float(denominator)
                            next_batch_index = min(begin+16, len(data))
                            resume_rng = rng_state()
                            if global_step % 10 == 0:
                                progress = {'epoch': epoch, 'next_batch_index': next_batch_index,
                                            'train_n': len(data), 'global_step': global_step,
                                            'elapsed_seconds': previous_seconds+time.time()-began,
                                            'run_id': run_identity(), 'updated_at': utc_now()}
                                write_json(folder / 'progress.json', progress)
                                if global_step % 100 == 0:
                                    print('FIT_PROGRESS '+json.dumps(progress), flush=True)
                            if time.time()-last_checkpoint_at >= 900:
                                save_last()
                                last_checkpoint_at = time.time()
                        # A resume with cursor==N reruns development only, never
                        # silently omits part of the training epoch.
                        save_last()
                        if next_batch_index != len(data) or global_step < math.ceil(len(data)/16):
                            raise ValueError('Training epoch did not consume the complete frozen train role')
                        predictions = infer(model, dev, arm, seed, f'epoch-{epoch}-step-{global_step}', budget)
                        metrics = score_rows(predictions)
                        if metrics['eer'] is None:
                            raise ValueError('Single-class development cannot select a model')
                        epoch_evidence = _save_epoch_predictions(folder, epoch, predictions, budget)
                        rank = (metrics['eer'], metrics['ll'], epoch)
                        improved = rank < best_rank
                        next_bad = 0 if metrics['eer'] < best_rank[0] else bad+1
                        entry = {'epoch': epoch, 'global_step': global_step, 'train_n': len(data),
                                 'train_weighted_ce': numerator_sum/denominator_sum, 'dev': metrics,
                                 'bad_epochs': next_bad, 'epoch_seconds': time.time()-epoch_started,
                                 'fit_seconds': previous_seconds+time.time()-began,
                                 'coverage_identity': data.coverage_identity, 'run_id': run_identity(),
                                 **epoch_evidence}
                        if improved:
                            payload = delta_payload(model, config, arm, seed, prepared['base_identity'], manifest_identity,
                                                    epoch, metrics['threshold'], metrics, scale)
                            payload.update(train_manifest_identity=train_manifest_identity, train_n=len(data),
                                           train_coverage_identity=data.coverage_identity)
                            try:
                                atomic_save(payload, folder / 'best.pt', budget)
                                write_jsonl(folder / 'best_dev_predictions.jsonl', predictions)
                            except BaseException:
                                selection_pair_complete = False
                                raise
                            finally:
                                del payload
                            selection_pair_complete = True
                            best_rank = rank
                        # Commit selection only after the best checkpoint and
                        # its reference predictions are both durable. A failed
                        # pair leaves cursor==N and the old rank, so a resumed
                        # development pass can repair the incomplete pair.
                        bad = next_bad
                        last_metrics = metrics
                        next_epoch, next_batch_index = epoch+1, 0
                        numerator_sum = denominator_sum = 0.0
                        resume_rng = rng_state()
                        save_last()
                        _append(folder / 'history.jsonl', entry)
                        print('EPOCH '+json.dumps(entry), flush=True)
                        begin_index = 0
                        if epoch >= 5 and bad >= 7:
                            stop = 'early_stopping'
                            break
                        if time.time() >= budget.state['deadlines']['train_stop']-120 and epoch < 40:
                            raise BudgetStop('Training cutoff after completed development selection')
            except BaseException as error:
                interruption = error
                stop = 'training_cutoff' if isinstance(error, BudgetStop) else 'interrupted'
                optimizer.zero_grad(set_to_none=True)
                # Forward/backward faults leave parameters unchanged. A failed
                # optimizer/scheduler update may mutate a subset of parameters
                # or moments; preserve the preceding durable restore boundary.
                gc.collect()
                torch.cuda.empty_cache()
                if checkpoint_usable:
                    save_last()
            del model, optimizer, scheduler
            model = optimizer = scheduler = None
            gc.collect()
            torch.cuda.empty_cache()
            result = {'status': 'partial' if isinstance(interruption, BudgetStop) else ('failed' if interruption else 'complete'),
                      'arm': arm, 'seed': seed, 'scale': scale, 'train_n': len(data), 'dev_n': len(dev),
                      'manifest_identity': manifest_identity, 'train_manifest_identity': train_manifest_identity,
                      'train_coverage_identity': data.coverage_identity, 'stop_reason': stop,
                      'elapsed_seconds': previous_seconds+time.time()-began, 'run_id': run_identity(),
                      'parameters': parameter_counts, 'source_identity': source_identity(),
                      'checkpoint_usable': False, 'current_state_safe': checkpoint_usable,
                      'selection_pair_complete': selection_pair_complete,
                      'resumable': (folder / 'last.pt').exists() and stored_last_usable,
                      'last_safe_boundary': last_safe_boundary}
            if interruption:
                result.update(reason=str(interruption), error_type=type(interruption).__name__)
            if (folder / 'best.pt').exists() and selection_pair_complete:
                # Safe cutoff saves optimization first; FP32 development reload
                # is evaluation and may continue during the final 24 hours.
                model, saved = load_delta(folder / 'best.pt', config, budget)
                checkpoint_id = file_sha256(folder / 'best.pt')
                restored = infer(model, dev, arm, seed, checkpoint_id, budget)
                original = read_jsonl(folder / 'best_dev_predictions.jsonl')
                expected_ids = [(r['sample_id'], r['label'], r['component_id']) for r in original]
                actual_ids = [(r['sample_id'], r['label'], r['component_id']) for r in restored]
                if expected_ids != actual_ids:
                    raise ValueError('Selected/reloaded development prediction sample identity differs')
                difference = float(np.max(np.abs(np.asarray([r['logits'] for r in original])-np.asarray([r['logits'] for r in restored]))))
                if difference > 2e-5:
                    raise ValueError('Fresh checkpoint reload logits differ from model selection')
                write_jsonl(folder / 'dev_predictions.jsonl', restored)
                result.update(selected_epoch=saved['epoch'], dev=score_rows(restored, saved['threshold']),
                              checkpoint_id=checkpoint_id, reload_max_logit_abs_diff=difference,
                              checkpoint_usable=True, peak_cuda_bytes=torch.cuda.max_memory_allocated())
                del model, saved
                model = None
                gc.collect()
                torch.cuda.empty_cache()
            elif result['status'] == 'complete':
                raise RuntimeError('A completed scientific fit has no development-selected best checkpoint')
            write_json(result_path, result)
            budget.state['fits'][key] = result
            budget.save()
            print('FIT_RESULT '+json.dumps(result), flush=True)
            if interruption and not isinstance(interruption, BudgetStop):
                raise interruption
            return result
        except BaseException as error:
            # Do not promote a failed reload or incomplete fit to complete.
            failure = {'status': 'partial' if isinstance(error, BudgetStop) else 'failed',
                       'arm': arm, 'seed': seed, 'scale': scale, 'train_n': len(data),
                       'manifest_identity': manifest_identity, 'run_id': run_identity(),
                       'reason': str(error), 'error_type': type(error).__name__,
                       'traceback': traceback.format_exc(), 'checkpoint_usable': False,
                       'current_state_safe': checkpoint_usable,
                       'selection_pair_complete': selection_pair_complete,
                       'resumable': (folder / 'last.pt').exists() and stored_last_usable,
                       'last_safe_boundary': last_safe_boundary, 'updated_at': utc_now()}
            if result_path.exists():
                previous = read_json(result_path)
                best_path = folder / 'best.pt'
                if (previous.get('checkpoint_usable') and selection_pair_complete and best_path.exists()
                        and previous.get('checkpoint_id') == file_sha256(best_path)):
                    # Preserve independently replayed best metadata only while the
                    # exact checkpoint remains intact. Current recovery evidence
                    # takes priority over any earlier result's status or cursor.
                    failure = {**previous, **failure, 'checkpoint_usable': True}
            write_json(result_path, failure)
            budget.state['fits'][key] = failure
            budget.state['failures'].append({'fit': key, 'run_id': run_identity(), 'reason': str(error), 'at': utc_now()})
            budget.save()
            print('FIT_FAILURE '+json.dumps(failure), flush=True)
            raise
        finally:
            del model, optimizer, scheduler
            gc.collect()
            torch.cuda.empty_cache()
