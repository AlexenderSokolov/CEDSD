"""Verified old-cache references and streamed full frozen-feature completion."""
from __future__ import annotations

from collections import Counter
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

from .common import config_digest, file_sha256, read_json, read_jsonl, write_json, utc_now
from .data import window_digest
from .preprocess import (_processing_spec, _identity, _load_checked_window, _frozen_auto_model,
                         TextFeatureEncoder, clean_asr_text, _release_models)
from .training import seed_everything

REPO = Path(__file__).resolve().parents[3]
GIB=1024**3


def store_guard(budget,store,required_bytes=0):
    free=shutil.disk_usage(store).free
    receipt={'path':str(store),'free_bytes':free,'required_increment_bytes':int(required_bytes),
             'minimum_free_bytes':30*GIB,'checked_at':utc_now()}
    budget.state['feature_store_disk']=receipt
    budget.save()
    if free-int(required_bytes)<30*GIB:
        from .full_runtime import BudgetStop
        raise BudgetStop(f'Feature filesystem30GiB boundary: need {int(required_bytes)} bytes; free {free}; path {store}')
    return receipt


def semantic_processing(processing):
    value=copy.deepcopy(processing)
    for spec in value['models'].values(): spec.pop('id',None)
    return value


def verify_item(value,row,identity):
    if not value.get('cache_complete') or value.get('sample_id')!=row['sample_id'] or value.get('cache_identity')!=identity:
        raise ValueError('Cache identity/completeness mismatch '+row['sample_id'])
    wave=value['waveform'].float().cpu().numpy()
    if wave.ndim!=1 or window_digest(wave,16000)!=row['window_sha256']:
        raise ValueError('Cache input-window mismatch')
    if value['affective'].ndim!=2 or value['affective'].shape[1]!=1024 or value['text'].ndim!=2 or value['text'].shape[1]!=768:
        raise ValueError('Frozen encoder feature dimensions mismatch')
    for key in ('waveform','affective','text'):
        if not torch.isfinite(value[key]).all(): raise ValueError('Nonfinite frozen cache')


def model_identity(config):
    locked=read_json(REPO/'uica_exec/inputs/model-lock.json')['models']
    models={}
    for name,spec in locked.items():
        folder=Path(config['models'][name])
        for rel,sha in spec['asset_sha256'].items():
            if file_sha256(folder/rel)!=sha: raise ValueError('Published asset changed '+name+'/'+rel)
        files={str(p.relative_to(folder)):file_sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in {'.bin','.safetensors','.pt'}}
        models[name]={'revision':config['models']['revisions'][name],'files':files}
    return models,config_digest(models['xlsr'])


def prepare(config,budget):
    budget.guard(training=True)
    audit=read_json(budget.root/'data_audit.json')
    manifest=budget.root/'primary_manifest.jsonl'
    if audit['manifest_identity']!=file_sha256(manifest): raise ValueError('Frozen full manifest changed')
    rows=read_jsonl(manifest)
    root=Path(config['paths']['data_root'])
    cache=budget.root/'cache';cache.mkdir(exist_ok=True)
    store=Path(config['full'].get('feature_store',str(cache)))
    store.mkdir(parents=True,exist_ok=True,mode=0o700)
    if store.stat().st_uid!=os.getuid(): raise ValueError('Feature store must belong to the execution user')
    separate_volume=store.stat().st_dev!=budget.root.stat().st_dev
    asr_dir=cache/'asr';asr_dir.mkdir(exist_ok=True)
    processing=_processing_spec(config);fingerprint=config_digest(processing)
    historical=Path(config['full']['historical_cache'])
    old_receipt=read_json(historical/'preprocess.json')
    old_processing=old_receipt['processing'];old_fingerprint=old_receipt['processing_fingerprint']
    if config_digest(old_processing)!=old_fingerprint: raise ValueError('Historical cache receipt fingerprint invalid')
    if semantic_processing(processing)!=semantic_processing(old_processing):
        raise ValueError('Historical cache scientific processing differs; cannot reuse by path')
    models,base_identity=model_identity(config)
    index_path=cache/'cache_index.json'
    index=read_json(index_path) if index_path.exists() else {}
    report={'status':'checking','processing':processing,'processing_fingerprint':fingerprint,
            'manifest_sha256':audit['manifest_identity'],'requested_count':len(rows),'completed_count':0,
            'cached_count':0,'new_count':0,'asr_seed_policy':'new rows sha256(sample_id) first8 mod2**31; legacy text preserved',
            'historical_processing_fingerprint':old_fingerprint,'updated_at':utc_now()}
    pending=[];checked={};reuse_bytes=0;rejected_reuse=[];old_transcripts={}
    for i,row in enumerate(rows):
        entry=index.get(row['sample_id'])
        if entry:
            path=Path(entry['path']);p=entry['processing'];expected=_identity(row,config_digest(p),p)
            if entry['cache_identity']!=expected or semantic_processing(p)!=semantic_processing(processing):
                raise ValueError('Cache index identity changed')
        else:
            old_path=historical/(row['sample_id']+'.pt')
            new_path=store/(row['sample_id']+'.pt')
            path=old_path if old_path.exists() else new_path
            p=old_processing if path==old_path else processing
            expected=_identity(row,config_digest(p),p)
            entry={'path':str(path),'cache_identity':expected,'processing':p,'origin':'historical_verified' if path==old_path else 'full_new'}
        if path.exists():
            value=torch.load(path,map_location='cpu',weights_only=True)
            try:
                verify_item(value,row,expected)
            except ValueError as error:
                if entry['origin']!='historical_verified': raise
                # A present legacy path is only a candidate. Mismatched source/window
                # identity calls for new features, never a rewrite of the old cache.
                rejected_reuse.append({'sample_id':row['sample_id'],'legacy_path':str(path),'reason':str(error)})
                old_transcripts[row['sample_id']]=value.get('transcript')
                replacement=store/(row['sample_id']+'.pt')
                if replacement.exists():
                    replacement_identity=_identity(row,fingerprint,processing)
                    fresh=torch.load(replacement,map_location='cpu',weights_only=True)
                    verify_item(fresh,row,replacement_identity)
                    _load_checked_window(row,root,config['audio'])
                    checked[row['sample_id']]={'path':str(replacement),'cache_identity':replacement_identity,
                                               'processing':processing,'origin':'full_new'}
                    report['new_count']+=1
                    continue
                pending.append(row)
                continue
            # Compare the actual present source, not just path or a past receipt.
            _load_checked_window(row,root,config['audio'])
            checked[row['sample_id']]=entry
            if entry['origin']=='historical_verified':
                report['cached_count']+=1;reuse_bytes+=path.stat().st_size
            else: report['new_count']+=1
        else: pending.append(row)
        if (i+1)%500==0:
            budget.guard(training=True)
            print('FULL_CACHE_CHECK '+json.dumps({'checked':i+1,'total':len(rows),'reused':report['cached_count'],'pending':len(pending)}),flush=True)
    index=checked
    write_json(index_path,index)
    write_json(cache/'rejected_legacy_reuse.json',rejected_reuse)
    report['completed_count']=len(index)
    # Full disk projection includes all18 best+last/Adam deltas and atomic-save headroom.
    old_files=[p for p in historical.glob('*.pt') if p.is_file()]
    old_mean=sum(p.stat().st_size for p in old_files)/max(1,len(old_files))
    estimated_cache_bytes=int(len(pending)*old_mean*1.15)
    # Derive storage from the actual largest matched arm, not a per-fit guess.
    # best stores one FP32 delta; last stores delta plus two Adam moments.
    # Allow per-fit metadata and one extra last for atomic replacement, then
    # another2GiB for feature-index, engineering delta, predictions and transport.
    from .closeout_model import CloseoutDetector
    probe=CloseoutDetector(config,'log',17).float()
    trainable_bytes=sum(p.numel()*p.element_size() for p in probe.parameters() if p.requires_grad)
    parameter_counts=probe.parameter_counts()
    del probe;gc.collect()
    per_fit_bytes=4*trainable_bytes+16*1024**2
    atomic_headroom=3*trainable_bytes+32*1024**2
    checkpoint_reserve=18*per_fit_bytes+atomic_headroom+2*GIB
    required=estimated_cache_bytes+checkpoint_reserve
    disk=budget.guard(checkpoint_reserve if separate_volume else required,force=True)
    feature_disk=store_guard(budget,store,estimated_cache_bytes if separate_volume else required)
    projection={'pending_rows':len(pending),'historical_mean_bytes':old_mean,'estimated_new_cache_bytes':estimated_cache_bytes,
                'checkpoint_and_atomic_reserve_bytes':checkpoint_reserve,'required_bytes':required,
                'trainable_parameter_bytes':trainable_bytes,'parameter_counts':parameter_counts,
                'per_fit_best_plus_last_upper_bytes':per_fit_bytes,'atomic_headroom_bytes':atomic_headroom,
                'feature_store':str(store),'separate_feature_filesystem':separate_volume,
                'output_disk':disk,'feature_disk':feature_disk,
                'old_features_referenced_bytes':reuse_bytes,'min_free_gib':30,'checked_at':utc_now()}
    write_json(budget.root/'disk_projection.json',projection)
    print('FULL_DISK_PROJECTION '+json.dumps(projection),flush=True)
    report['status']='running';write_json(cache/'preprocess.json',report)
    asr=affect=encoder=None
    begin=time.monotonic()
    try:
        if pending:
            # One GPU; resident frozen encoders avoid duplicate full intermediates on disk.
            asr=_frozen_auto_model(config,'sensevoice','cuda')
            affect=_frozen_auto_model(config,'emotion2vec','cuda')
            encoder=TextFeatureEncoder(config,'cuda')
        for row in pending:
            budget.guard(training=True)
            values=_load_checked_window(row,root,config['audio'])
            identity=_identity(row,fingerprint,processing)
            transcript_path=asr_dir/(row['sample_id']+'.json')
            if transcript_path.exists():
                transcript=read_json(transcript_path)
                if transcript['cache_identity']!=identity: raise ValueError('ASR transcript identity changed')
            else:
                sample_seed=int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16)%(2**31)
                seed_everything(sample_seed)
                with torch.inference_mode():
                    result=asr.generate(input=values,cache={},language='zh',use_itn=True,batch_size_s=60,merge_vad=False,fs=16000)
                if not isinstance(result,list) or len(result)!=1 or not isinstance(result[0].get('text'),str):
                    raise ValueError('ASR failure is not empty text '+row['sample_id'])
                text=clean_asr_text(result[0]['text']);old_text=old_transcripts.get(row['sample_id'])
                transcript={'sample_id':row['sample_id'],'raw_text':result[0]['text'],'transcript':text,
                            'seed':sample_seed,'cache_identity':identity,'old_text':old_text,
                            'text_drift':('changed' if old_text!=text else 'unchanged') if old_text is not None else 'new_no_prior_cache'}
                write_json(transcript_path,transcript)
            with torch.inference_mode():
                result=affect.generate(input=values,granularity='frame',extract_embedding=True,fs=16000)
            if not isinstance(result,list) or len(result)!=1 or 'feats' not in result[0]: raise ValueError('E2V failed')
            features=torch.as_tensor(result[0]['feats']).detach().float().cpu()
            if features.ndim!=2 or not len(features) or features.shape[1]!=1024: raise ValueError('E2V frame shape')
            encoded=encoder.encode(transcript['transcript'])
            value={'waveform':torch.from_numpy(values.copy()),'affective':features,**encoded,
                   'sample_id':row['sample_id'],'cache_identity':identity,'cache_complete':True}
            verify_item(value,row,identity)
            size=sum(v.numel()*v.element_size() for v in value.values() if isinstance(v,torch.Tensor))
            budget.guard(checkpoint_reserve if separate_volume else size+4096+checkpoint_reserve,force=True)
            store_guard(budget,store,size+4096 if separate_volume else size+4096+checkpoint_reserve)
            path=store/(row['sample_id']+'.pt');tmp=path.with_suffix('.atomic')
            torch.save(value,tmp);tmp.replace(path)
            index[row['sample_id']]={'path':str(path),'cache_identity':identity,'processing':processing,'origin':'full_new'}
            report['new_count']+=1;report['completed_count']=len(index)
            if len(index)%100==0:
                report['updated_at']=utc_now();write_json(cache/'preprocess.json',report)
                # Only durable features in this index; an interruption can discover later files.
                write_json(index_path,index)
                print('FULL_CACHE_PROGRESS '+json.dumps({'complete':len(index),'total':len(rows),'new':report['new_count'],
                      'legacy':report['cached_count'],'elapsed_seconds':time.monotonic()-begin}),flush=True)
    finally:
        del asr,affect,encoder;_release_models()
        write_json(index_path,index);write_json(cache/'preprocess.json',report)
    if set(index)!={r['sample_id'] for r in rows}: raise ValueError('Full cache coverage incomplete')
    report.update(status='completed',completed_count=len(rows),updated_at=utc_now())
    write_json(cache/'preprocess.json',report)
    prepared={'base_identity':base_identity,'models':models,'manifest_identity':audit['manifest_identity'],
              'manifest':audit['manifest_identity'],'cache_index_identity':file_sha256(index_path),
              'counts':dict(Counter(r['role'] for r in rows)),'cache_status':report,'prepared_at':utc_now()}
    write_json(budget.root/'prepared.json',prepared)
    print('FULL_PREPARED '+json.dumps({'counts':prepared['counts'],'cache_completed':len(rows),'legacy':report['cached_count'],
                                     'new':report['new_count'],'base_identity':base_identity,'manifest_identity':audit['manifest_identity']}),flush=True)
    return prepared
