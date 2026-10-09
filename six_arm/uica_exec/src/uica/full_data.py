"""Full-corpus accounting and frozen recording-family split; no old admission policy."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .common import config_digest, file_sha256, read_json, read_jsonl, write_json, write_jsonl, utc_now
from .data import _path_record, _read_audio, _window_from_audio, resolve_audio_path, window_digest

REPO = Path(__file__).resolve().parents[3]
PRIOR = REPO / 'uica_exec/inputs/full_manifest.jsonl'


def metadata(filename, label, ordinal):
    # Only parsing/identity fields are reused. Old role/exclusion policy is replaced.
    row = _path_record(filename, label)
    row.update(manifest_row=ordinal + 2, csv_ordinal=ordinal, role='pending', exclusion_reason=None,
               identity_source='path_inferred' if row['parent_ids'] else 'unknown',
               original_fad_role=row['original_partition'] if row['source'] == 'FAD' else None)
    reasons = []
    if row['label'] is None:
        reasons.append('unknown_csv_label')
    if row['source'] in {'MDPE', 'CommonVoice'} and row['label'] == 1:
        reasons.append('unresolved_source_label_300')
    if any('replaceonce' in p.lower() or 'partial' in p.lower() for p in Path(filename).parts):
        reasons.append('partial_spoof_missing_position_812')
    if row['source'] == 'FAD':
        expected = {0 if p in {'real','real_noise','bonafide'} else 1 for p in filename.lower().split('/')
                    if p in {'real','real_noise','bonafide','fake','fake_noise','spoof'}}
        if len(expected) != 1 or row['label'] not in expected:
            reasons.append('path_label_conflict')
        if row['original_fad_role'] not in {'train','validation','test'}:
            reasons.append('missing_fad_original_role')
    # Explicit CommonVoice recording IDs also occur in SeedVC filenames.
    cv = re.findall(r'common_voice_zh[-_]CN_(\d+)', filename, re.I)
    keys = list(row['parent_ids']) + ['commonvoice:zh-CN:' + v for v in cv]
    # THCHS30's A/D utterance IDs are scoped to the corpus, not the generator.
    if 'thchs30' in filename.lower():
        keys += ['thchs30:' + x.upper() for x in re.findall(r'(?<![A-Za-z0-9])([ABCD]\d+_\d+)(?!\d)', filename, re.I)]
    if row['source']=='FAD' and not row['parent_ids']:
        parts=filename.lower().split('/')
        corpus=next((p for p in parts if p in {'thchs30','magicread','magicconversa','nieshuaireal'}),None)
        if corpus:
            stem=Path(filename).stem
            if row['condition']=='noise':
                stem=re.sub(r'_(?:n\d+|metro|tram|bus|street|public|shopping|airport|park)_snr\d+$','',stem,flags=re.I)
            keys.append(corpus+':'+stem)
            row['base_corpus']=corpus
    if row['source']=='SeedVC':
        observed=re.match(r'([^/]+)_common_voice_',Path(filename).stem,re.I)
        if observed:
            keys.append('observed_possible_shared_reference:SeedVC:'+observed.group(1))
            row['possible_reference_role']='unknown_conservatively_grouped_observed_prefix'
    if row['source'] in {'CosyVoice_VC','OpenVoice','OpenVoice_VC','GPTSoVITS_VC','SeedVC','Fish_VC'}:
        for identity in row['identity_roles']:
            identity.update(role='unknown',provenance='path_token_only_no_verified_role')
    row['recording_keys'] = sorted(set(keys))
    row['identity_source'] = 'path_inferred' if keys else 'unknown'
    row['observed_recording_tokens_unknown_role'] = row['parent_ids'] if len(row['parent_ids']) > 1 else []
    row['known_reference_keys'] = []
    row['family_relation_evidence'] = 'explicit_recording_ids_in_path' if keys else 'unknown_parent_singleton_unless_exact_duplicate'
    row['reasons'] = reasons
    row['fad_seen'] = ('unseen' if any(p in {'unseen','test_unseen'} for p in filename.lower().split('/')) else 'seen') if row['source'] == 'FAD' else None
    return row


def decode(row, root, audio):
    result = {'sample_id':row['sample_id'], 'file':row['file']}
    try:
        path = resolve_audio_path(root, row['file'])
        result['file_bytes'] = path.stat().st_size
        result['content_sha256'] = file_sha256(path)
        values, rate = _read_audio(path)
        digest = hashlib.sha256(f'decoded-float32:{rate}:{values.shape[1]}:'.encode())
        digest.update(np.asarray(values, dtype='<f4').tobytes())
        window = _window_from_audio(values, rate, audio['sample_rate'], audio['max_seconds'])
        result.update(pcm_sha256=digest.hexdigest(), window_sha256=window_digest(window,audio['sample_rate']),
                      window_end=len(window)/audio['sample_rate'], original_sample_rate=rate,
                      original_channels=values.shape[1], duration_seconds=len(values)/rate,
                      window_num_samples=len(window), decode_status='complete')
    except Exception as error:
        result.update(decode_status='failed',audio_error=f'{type(error).__name__}: {error}')
    return result


def family_split(rows, seed):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        parent[find(b)] = find(a)
    edges, duplicates = {}, defaultdict(list)
    label_conflicts = set()
    for i,row in enumerate(rows):
        keys = [('recording',v) for v in row['recording_keys']]
        keys += [(k,row[k]) for k in ('content_sha256','pcm_sha256','window_sha256') if row.get(k)]
        for key in keys:
            if key in edges: union(edges[key],i)
            else: edges[key] = i
        if row.get('window_sha256'):
            duplicates[row['window_sha256']].append(i)
    groups = defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    priority = {'train':0,'validation':1,'test':2}
    conflicts = []
    for indices in groups.values():
        family = 'family_' + config_digest(sorted(rows[i]['sample_id'] for i in indices))[:24]
        anchors = {rows[i]['original_fad_role'] for i in indices if rows[i]['original_fad_role'] in priority}
        if anchors:
            role = max(anchors,key=priority.get)
            assignment = 'FAD_anchor_highest_priority'
        else:
            number = int(hashlib.sha256(f'{seed}:{family}'.encode()).hexdigest()[:16],16) / 2**64
            role = 'train' if number < .8 else 'validation' if number < .9 else 'test'
            assignment = 'seed1701_family_hash_80_10_10'
        if len(anchors)>1:
            conflicts.append({'family_id':family,'anchor_roles':sorted(anchors),'retained_role':role,'n':len(indices)})
        for i in indices:
            row=rows[i]
            row.update(family_id=family,component_id=family,assigned_partition=role,split_evidence=assignment)
            if row['original_fad_role'] in priority and row['original_fad_role'] != role:
                row['reasons'].append('lower_priority_fad_role_conflict')
            row['role'] = role
    # Inconsistent labels for exactly the same model input cannot be repaired by relabeling.
    for indices in duplicates.values():
        if len({rows[i]['label'] for i in indices})>1:
            for i in indices:
                rows[i]['reasons'].append('exact_input_label_conflict')
                label_conflicts.add(i)
        elif len(indices)>1:
            eligible = [i for i in indices if not rows[i]['reasons']]
            eligible.sort(key=lambda i:(-priority[rows[i]['role']], rows[i]['condition']!='clean',
                                        rows[i]['source']!='FAD',rows[i]['file']))
            for i in eligible[1:]:
                rows[i]['reasons'].append('redundant_exact_input_duplicate')
    for row in rows:
        row['reasons'] = sorted(set(row['reasons']))
        row['exclusion_reason'] = ';'.join(row['reasons']) or None
        if row['reasons']: row['role']='quarantine'
    roles_by_family=defaultdict(set)
    for row in rows:
        if row['role']!='quarantine': roles_by_family[row['family_id']].add(row['role'])
    if any(len(v)>1 for v in roles_by_family.values()):
        raise ValueError('Family crosses frozen partitions')
    return {'families':len(groups),'largest_family':max(map(len,groups.values())),
            'anchor_conflicts':conflicts,'exact_window_duplicate_groups':sum(len(v)>1 for v in duplicates.values()),
            'exact_input_label_conflict_rows':len(label_conflicts),'cross_partition_family_violations':0}


def audit(config,budget):
    root = Path(config['paths']['data_root'])
    csv_path = root/'zh-label.csv'
    sha = file_sha256(csv_path)
    expected_sha = file_sha256(REPO/'research/uica-full-20261004/source_full.csv')
    if sha != expected_sha: raise ValueError('Full source CSV changed after preflight copy')
    finished = budget.root/'data_audit.json'
    if finished.exists():
        result=read_json(finished)
        if result['source_csv_sha256']!=sha or result['allocation_sha256']!=file_sha256(budget.root/'allocation_manifest.jsonl'):
            raise ValueError('Frozen audit identity changed')
        if result['manifest_identity']!=file_sha256(budget.root/'primary_manifest.jsonl'):
            raise ValueError('Frozen primary manifest changed')
        print('FULL_AUDIT_ALREADY_FROZEN '+json.dumps(result,ensure_ascii=False),flush=True)
        return result
    with csv_path.open(encoding='utf-8-sig',newline='') as stream:
        raw = list(csv.DictReader(stream))
    if len(raw)!=config['full']['expected_rows']: raise ValueError('Must account for all69700 rows')
    rows = [metadata(r['file'],r['label'],i) for i,r in enumerate(raw)]
    if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate CSV file rows require explicit accounting')
    old = {r['sample_id']:r for r in read_jsonl(PRIOR)}
    for row in rows:
        prior=old.get(row['sample_id'])
        if prior and (prior['file'],prior['label'])!=(row['file'],row['label']):
            raise ValueError('Prior/full path-label disagreement')
        row['history_usage'] = {'mini_present':bool(prior), 'prior_role':prior['role'] if prior else None,
                                'known_prior_training':bool(prior and prior['role']=='train'),
                                'project_history_coverage':'partial_known_mini_snapshot_only'}
    journal=budget.root/'decode_ledger.jsonl'
    receipt=budget.root/'decode_source.json'
    if receipt.exists() and read_json(receipt)['source_csv_sha256']!=sha:
        raise ValueError('Decode journal source identity changed')
    write_json(receipt,{'source_csv_sha256':sha,'rows':len(rows),'audio':config['audio']})
    previous = {r['sample_id']:r for r in read_jsonl(journal)} if journal.exists() else {}
    pending = [r for r in rows if r['sample_id'] not in previous]
    begin=time.monotonic()
    with journal.open('a',encoding='utf-8',newline='\n') as stream, ThreadPoolExecutor(max_workers=config['full'].get('decode_workers',4)) as pool:
        for result in pool.map(lambda r:decode(r,root,config['audio']),pending):
            stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            previous[result['sample_id']]=result
            if len(previous)%500==0:
                stream.flush()
                budget.guard()
                progress={'decoded':len(previous),'total':len(rows),'failed':sum(r['decode_status']=='failed' for r in previous.values()),
                          'elapsed_seconds':time.monotonic()-begin,'run_id':budget.state.get('current_run_id')}
                write_json(budget.root/'decode_progress.json',progress)
                print('FULL_DECODE_PROGRESS '+json.dumps(progress),flush=True)
    for row in rows:
        row.update(previous[row['sample_id']])
        if row['decode_status']!='complete': row['reasons'].append('full_audio_decode_failure')
    graph=family_split(rows,config['full']['split_seed'])
    primary=sorted((r for r in rows if r['role']!='quarantine'),key=lambda r:r['sample_id'])
    counts=Counter(r['role'] for r in rows)
    if sum(counts.values())!=69700: raise ValueError('Incomplete per-row accounting')
    if any(sorted({r['label'] for r in primary if r['role']==role}) != [0,1] for role in ('train','validation','test')):
        raise ValueError('A main partition lacks both classes')
    write_jsonl(budget.root/'allocation_manifest.jsonl',rows)
    write_jsonl(budget.root/'primary_manifest.jsonl',primary)
    columns=['csv_ordinal','sample_id','file','original_label','label','source','role','original_fad_role','assigned_partition',
             'family_id','recording_keys','known_reference_keys','speaker_id','identity_source','history_usage','condition',
             'fad_seen','duration_seconds','window_end','decode_status','content_sha256','pcm_sha256','window_sha256','exclusion_reason']
    with (budget.root/'全量分配表.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in rows: writer.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list)) else r.get(k) for k in columns})
    summary={'status':'frozen','frozen_at':utc_now(),'source_csv_sha256':sha,'record_count':len(rows),
             'role_counts':dict(counts),'label_counts':dict(Counter(str(r['label']) for r in rows)),
             'source_counts':dict(Counter(r['source'] for r in rows)),
             'source_role_label_counts':[{'source':k[0],'role':k[1],'label':k[2],'n':n} for k,n in
                                         sorted(Counter((r['source'],r['role'],r['label']) for r in rows).items(),key=lambda x:str(x[0]))],
             'source_duration_seconds':{s:sum(r.get('duration_seconds',0) for r in rows if r['source']==s) for s in sorted({r['source'] for r in rows})},
             'role_usable_window_seconds':{s:sum(r.get('window_end',0) for r in rows if r['role']==s) for s in sorted(counts)},
             'duration_seconds':sum(r.get('duration_seconds',0) for r in rows),
             'usable_window_seconds':sum(r.get('window_end',0) for r in primary),
             'exclusion_counts':dict(Counter(reason for r in rows for reason in r['reasons'])),
             'decode_failures':sum(r['decode_status']!='complete' for r in rows),'family_graph':graph,
             'manifest_identity':file_sha256(budget.root/'primary_manifest.jsonl'),
             'allocation_sha256':file_sha256(budget.root/'allocation_manifest.jsonl'),
             'limitations':['Known recording/reference and exact PCM/window relations only; transformed near-duplicates may remain.',
                            'No strict speaker OOD claim; speaker IDs are reporting fields, never grouping edges.',
                            'Historical usage coverage is partial; exposed materials are not new confirmation data.']}
    write_json(finished,summary)
    print('FULL_AUDIT_FROZEN '+json.dumps(summary,ensure_ascii=False),flush=True)
    return summary
