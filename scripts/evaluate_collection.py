#!/usr/bin/env python3
"""Оценка готовых моделей с сохранением предсказаний по батчам."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import unicodedata
from functools import lru_cache

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Нативный matmul Metal без fast math.
os.environ.setdefault("PYTORCH_MPS_PREFER_METAL", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from rapidfuzz.distance import Levenshtein
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, EncoderDecoderModel, VisionEncoderDecoderModel

from sumbase.charset import glyph_cer_tokens
from sumbase.corpus.acquire import checksum, dump
from sumbase.metrics import text_metrics
from sumbase.normalize import to_cdli_ascii
from sumbase.pretrained import tokens_to_glyphs, tokens_to_translit, to_glyph_input, pad_square_resize
from sumbase.utils import device_summary, pick_device

REVISIONS = {
    "praeclarum/cuneiform":"02d6e0940c949f88c70ac3e49dbbf072cf645b92",
    "colesimmons/sumerian-transliteration":"afb0fb5c33380024caefa6c92044d524a8d1975b",
    "colesimmons/lineart-ocr":"1ffba5eb34bab891e7363dd419a1eb62daf6c57a",
    "colesimmons/SumerianGlyphTokenizer_Roberta":"75fadd99143f2d2fc690c285b7228d262c524be1",
    "colesimmons/SumerianTransliterationTokenizer_Roberta":"b2de6664d98f29604bdf23ca1f686d7bc216b4e2",
}


def read(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_sample(rows, n, salt):
    return sorted(rows, key=lambda r:hashlib.sha256((salt+str(r['id'])).encode()).hexdigest())[:n]


def source_of(row):
    return row.get("source") or next(iter(row.get("sources", [])), "unknown")


def translit_sample(rows, cap=150):
    by_source = collections.defaultdict(list)
    for r in rows:
        by_source[source_of(r)].append(r)
    selected = []
    for source, members in sorted(by_source.items()):
        members.sort(key=lambda r:(len(r['src']),r['id']))
        if len(members)<=cap:
            selected.extend(members)
        else:
            for k in range(3):
                bucket = members[len(members)*k//3:len(members)*(k+1)//3]
                selected.extend(stable_sample(bucket,cap//3,source+str(k)+":42"))
    return selected


def evaluation_id(task, row):
    key = row['cohort']+':'+str(row['id'])
    if task=='translit':
        # Пустой line_id даёт одинаковый ID разным строкам.
        signature=json.dumps([row['src'],row['tgt']],ensure_ascii=False)
        key+=':'+hashlib.sha256(signature.encode()).hexdigest()[:16]
    return key


def expanded_translit_sample(rows, fraction):
    """Дополнение исходной выборки по источникам и длине."""
    if not 0 < fraction <= 1:
        raise ValueError('Transliteration fraction must be in (0, 1]')
    baseline=translit_sample(rows)
    key=lambda r:evaluation_id('translit',dict(r,cohort='external'))
    selected={key(r):r for r in baseline}
    target=max(len(selected),math.ceil(fraction*len(rows)))
    by_source=collections.defaultdict(list)
    for r in rows:by_source[source_of(r)].append(r)
    buckets={}
    for source,members in sorted(by_source.items()):
        members.sort(key=lambda r:(len(r['src']),r['id']))
        for k in range(3):
            part=members[len(members)*k//3:len(members)*(k+1)//3]
            buckets[f'{source}:{k}']=[r for r in part if key(r) not in selected]
    remaining=sum(map(len,buckets.values()))
    needed=target-len(selected)
    allocations={label:divmod(needed*len(members),remaining) if remaining else (0,0)
                 for label,members in buckets.items()}
    leftover=needed-sum(n for n,_ in allocations.values())
    extra=set(sorted(allocations,key=lambda label:(-allocations[label][1],label))[:leftover])
    for label,members in sorted(buckets.items()):
        n=allocations[label][0]+(label in extra)
        members.sort(key=lambda r:hashlib.sha256(('expand:42:'+label+key(r)).encode()).hexdigest())
        for r in members[:n]:selected[key(r)]=r
    assert len(selected)==target
    return list(selected.values())


def plans(task, translit_fraction=None, control_limit=100):
    new = read(f"data/collection/ready/{task}/test.jsonl")
    if task == "translit":
        new = translit_sample(new) if translit_fraction is None else expanded_translit_sample(new,translit_fraction)
    old_paths = {"mt":"data/lite/mt/test.jsonl", "translit":"data/nogpu/translit/test.jsonl", "ocr":"data/nogpu/ocr_lineart/test.jsonl"}
    controls=read(old_paths[task])
    old = controls if control_limit is None else stable_sample(controls,control_limit,"control:42")
    out = []
    for cohort, members in [("external",new),("legacy_control",old)]:
        for original in members:
            r = dict(original)
            r.update(cohort=cohort, source=source_of(r) if cohort=="external" else "legacy")
            r['eval_id']=evaluation_id(task,r)
            out.append(r)
    assert len(out)==len({r['eval_id'] for r in out}),f'Duplicate evaluation keys: {task}'
    return out, old_paths[task]


def norm_text(value, task):
    value = unicodedata.normalize("NFC",value).casefold()
    if task=="translit":
        value=to_cdli_ascii(value)
        value=re.sub(r"[\[\]⸢⸣⌈⌉#?!]","",value)
    elif task=="mt":
        value=" ".join(re.findall(r"\w+",value,flags=re.UNICODE))
    return " ".join(value.split())


def sign_tokens(text):
    tokens = glyph_cer_tokens(text)
    return [x for x in tokens if x not in {" ","<SURFACE>","<COLUMN>","<BLANK_SPACE>","<RULING>"}]


def summarize(records, task):
    complete=[r for r in records if r.get('status')=='ok']
    result={"planned":len(records),"evaluated":len(complete),"skipped":dict(collections.Counter(r.get('status') for r in records if r.get('status')!='ok'))}
    if not complete:
        return result
    preds=[r['prediction'] for r in complete];refs=[r['tgt'] for r in complete]
    if task=='ocr':
        p=[sign_tokens(x) for x in preds];t=[sign_tokens(x) for x in refs]
        result.update(sign_cer=round(100*sum(Levenshtein.distance(a,b) for a,b in zip(p,t))/max(1,sum(map(len,t))),2),
                      sign_exact=round(100*sum(a==b for a,b in zip(p,t))/len(t),2),
                      raw_glyph_cer=round(100*sum(Levenshtein.distance(glyph_cer_tokens(a),glyph_cer_tokens(b)) for a,b in zip(preds,refs))/max(1,sum(len(glyph_cer_tokens(b)) for b in refs)),2),
                      single_sign_output_pct=round(100*sum(len(x)==1 for x in p)/len(p),2))
    else:
        result['raw']=text_metrics(preds,refs)
        result['normalized']=text_metrics([norm_text(x,task) for x in preds],[norm_text(x,task) for x in refs])
    result.update(empty_pct=round(100*sum(not x.strip() for x in preds)/len(preds),2),
                  hit_generation_limit_pct=round(100*sum(r.get('hit_generation_limit',False) for r in complete)/len(complete),2),
                  input_unk_examples=sum(r.get('input_unk_count',0)>0 for r in complete),
                  seconds=sum(r.get('seconds_share',0) for r in complete))
    if task=='translit':
        result.update(input_signs_dropped_examples=sum(r.get('input_dropped_signs',0)>0 for r in complete),
                      input_dropped_sign_pct=round(100*sum(r.get('input_dropped_signs',0) for r in complete)/max(1,sum(r.get('input_sign_count',0) for r in complete)),2))
    if task=='ocr':result['reference_oov_examples']=sum(r.get('reference_oov_signs',0)>0 for r in complete)
    return result


@lru_cache(maxsize=4)
def load_tok(name):
    return AutoTokenizer.from_pretrained(name,revision=REVISIONS[name],local_files_only=True)


def cuneiform_chars(text):
    return [c for c in text if 0x12000<=ord(c)<=0x1254f]


def glyph_coverage(text):
    tok=load_tok('colesimmons/SumerianGlyphTokenizer_Roberta')
    original=cuneiform_chars(text)
    encoded=cuneiform_chars(''.join(tok.convert_ids_to_tokens(tok(text,truncation=False)['input_ids'])))
    lost=sum((collections.Counter(original)-collections.Counter(encoded)).values())
    return len(original),lost


def audit_record(row,task):
    if task=='translit':
        n,lost=glyph_coverage(row['src'])
        row.update(input_sign_count=n,input_dropped_signs=lost)
    if task=='ocr':
        n,lost=glyph_coverage(row['tgt'])
        row.update(reference_sign_count=n,reference_oov_signs=lost)


@torch.inference_mode()
def evaluate(task, args):
    folder=args.output/task;folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else {}
    fraction=args.translit_fraction if task=='translit' else None
    if task=='translit' and fraction is None and manifest.exists():
        fraction=json.loads(manifest.read_text()).get('sampling',{}).get('requested_fraction')
    control_limit=None if args.full_control or previous.get('control_selection')=='all' else 100
    rows, old_path=plans(task,translit_fraction=fraction,control_limit=control_limit)
    full_rows=rows
    if args.phase=='smoke':
        buckets=collections.defaultdict(list)
        for r in rows:
            buckets[(r['cohort'],r['source'])].append(r)
        rows=[]
        for (cohort,source),members in sorted(buckets.items()):
            n=16 if task=='ocr' else 12 if cohort=='legacy_control' else 2
            rows.extend(stable_sample(members,n,'smoke:42'))
    spec={"task":task,"revisions":REVISIONS,"device":device_summary(),"seed":42,
          "operation":"inference_only_no_training", "new_data_sha256":checksum(Path(f'data/collection/ready/{task}/test.jsonl')),
          "legacy_data_sha256":checksum(Path(old_path)),"planned_external":sum(r['cohort']=='external' for r in full_rows),
          "control_selection":"all" if control_limit is None else control_limit,
          "source_max_sample":150 if task=='translit' and fraction is None else None,
          "generation":{"max_input":512,"max_new_tokens":256 if task=='mt' else 510 if task=='translit' else {'external':32,'legacy_control':258},"num_beams":4 if task=='mt' else 1,"do_sample":False}}
    if task=='mt':spec['generation']['encoder_padding_to']=512
    if task=='translit' and fraction is not None:
        spec['sampling']={"method":"retain_150_per_source_baseline_add_proportional_source_length_strata",
                          "requested_fraction":fraction,"population":len(read('data/collection/ready/translit/test.jsonl')),
                          "planned_external":sum(r['cohort']=='external' for r in full_rows),
                          "baseline_retained":len(translit_sample(read('data/collection/ready/translit/test.jsonl')))}
    if manifest.exists():
        prior=json.loads(manifest.read_text())
        for key in ('task','revisions','new_data_sha256','legacy_data_sha256','generation'):
            if prior[key]!=spec[key]:raise ValueError(f'Checkpoint protocol mismatch: {task}/{key}')
        if prior.get('sampling')!=spec.get('sampling'):raise ValueError('Sampling plan mismatch; use a separate output directory')
        if prior.get('control_selection',100)!=spec['control_selection']:raise ValueError('Control plan mismatch; use a separate output directory')
    else:dump(manifest,spec)
    checkpoint=folder/'predictions.jsonl'
    existing={}
    if checkpoint.exists():
        for r in read(checkpoint):
            r['eval_id']=evaluation_id(task,r)
            existing[r['eval_id']]=r
    planned_by_id={r['eval_id']:r for r in rows}
    for key,r in existing.items():
        if key in planned_by_id and task!='ocr':
            assert (r['src'],r['tgt'])==(planned_by_id[key]['src'],planned_by_id[key]['tgt'])
    pending=[r for r in rows if r['eval_id'] not in existing]
    print(f'{task}: planned={len(rows)} cached={len(rows)-len(pending)} remaining={len(pending)}',flush=True)
    device=pick_device();t0=time.time()
    mps_fraction=.6 if task=='mt' else .4
    if device=='mps':torch.mps.set_per_process_memory_fraction(mps_fraction)
    if pending:
        with (folder/'execution_sessions.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({"started_utc":datetime.now(timezone.utc).isoformat(),
                                "device":device_summary(),"pending":len(pending),
                                "cached":len(rows)-len(pending),
                                "batch_size":4 if task=='mt' else 8,
                                "mps_memory_fraction":mps_fraction if device=='mps' else None,
                                "mps_prefer_metal":os.environ.get('PYTORCH_MPS_PREFER_METAL'),
                                "script_sha256":checksum(Path(__file__))})+'\n')
        if task=='mt':
            name='praeclarum/cuneiform';enc=dec=load_tok(name)
            model=AutoModelForSeq2SeqLM.from_pretrained(name,revision=REVISIONS[name],local_files_only=True)
        elif task=='translit':
            name='colesimmons/sumerian-transliteration';enc=load_tok('colesimmons/SumerianGlyphTokenizer_Roberta');dec=load_tok('colesimmons/SumerianTransliterationTokenizer_Roberta')
            model=EncoderDecoderModel.from_pretrained(name,revision=REVISIONS[name],local_files_only=True)
        else:
            name='colesimmons/lineart-ocr';dec=load_tok('colesimmons/SumerianGlyphTokenizer_Roberta');enc=None
            model=VisionEncoderDecoderModel.from_pretrained(name,revision=REVISIONS[name],local_files_only=True)
        model.to(device).eval()
        assert not model.training and not torch.is_grad_enabled()
        if task!='ocr':
            for r in pending:
                r['model_input']='translate Sumerian to English: '+r['src'] if task=='mt' else to_glyph_input(r['src'])
                ids=enc(r['model_input'],truncation=False)['input_ids']
                r['input_tokens']=len(ids);r['input_unk_count']=ids.count(enc.unk_token_id)
            pending.sort(key=lambda r:(r['input_tokens'],r['eval_id']))
        else:
            pending.sort(key=lambda r:(r['cohort'],r['eval_id']))
        with checkpoint.open('a',encoding='utf-8') as stream:
            bs=4 if task=='mt' else 8
            for start in range(0,len(pending),bs):
                batch=pending[start:start+bs]
                # Для одиночных знаков и целых табличек нужны разные лимиты.
                subbatches=[batch] if task!='ocr' else [[r for r in batch if r['cohort']==c] for c in ('external','legacy_control')]
                for part in subbatches:
                    if not part:continue
                    valid=[r for r in part if task=='ocr' or r['input_tokens']<=512]
                    for r in part:
                        if r not in valid:
                            r.update(status='input_exceeds_512_tokens');stream.write(json.dumps(r,ensure_ascii=False)+'\n');existing[r['eval_id']]=r
                    if not valid:continue
                    max_new=256 if task=='mt' else 510 if task=='translit' else 32 if valid[0]['cohort']=='external' else 258
                    kw=dict(num_beams=4 if task=='mt' else 1,max_new_tokens=max_new,do_sample=False)
                    begin=time.time()
                    if task=='ocr':
                        import torchvision.transforms.functional as F
                        pixels=torch.stack([F.normalize(F.to_tensor(pad_square_resize(r.get('image',r.get('image_ref')),384)),[.5]*3,[.5]*3) for r in valid]).to(device)
                        generated=model.generate(pixel_values=pixels,decoder_start_token_id=dec.bos_token_id,eos_token_id=dec.eos_token_id,pad_token_id=dec.pad_token_id,**kw)
                    else:
                        # Фиксированная длина уменьшает число компиляций на MPS.
                        padding = dict(padding='max_length',max_length=512) if task=='mt' else dict(padding=True)
                        inputs=enc([r['model_input'] for r in valid],return_tensors='pt',truncation=False,**padding).to(device)
                        if task=='translit':kw.update(decoder_start_token_id=dec.bos_token_id,bos_token_id=dec.bos_token_id,eos_token_id=dec.eos_token_id,pad_token_id=dec.pad_token_id)
                        generated=model.generate(**inputs,**kw)
                    ids=generated.cpu().tolist();elapsed=time.time()-begin
                    for r,g in zip(valid,ids):
                        if task=='mt':pred=dec.decode(g,skip_special_tokens=True)
                        else:
                            tokens=dec.convert_ids_to_tokens(g);pred=tokens_to_translit(tokens) if task=='translit' else tokens_to_glyphs(tokens)
                        tail=g[1:];ended=dec.eos_token_id in tail
                        r.update(status='ok',prediction=pred,generated_tokens=len(tail[:tail.index(dec.eos_token_id)+1]) if ended else len(tail),
                                 hit_generation_limit=not ended and len(tail)>=max_new,seconds_share=elapsed/len(valid),
                                 inference_device=device,inference_batch_size=len(valid))
                        stream.write(json.dumps(r,ensure_ascii=False)+'\n');existing[r['eval_id']]=r
                    stream.flush()
                print(f'{task}: {min(start+bs,len(pending))}/{len(pending)} elapsed={time.time()-t0:.1f}s',flush=True)
                if device=='mps' and start%(bs*32)==0:torch.mps.empty_cache()
        del model;gc.collect()
        if device=='mps':torch.mps.empty_cache()
    current=[existing[r['eval_id']] for r in rows]
    for r in current:audit_record(r,task)
    # Сохраняем также ответы из предыдущих фаз.
    temporary=checkpoint.with_suffix('.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        for r in existing.values():f.write(json.dumps(r,ensure_ascii=False)+'\n')
    temporary.replace(checkpoint)
    summary={"task":task,"phase":args.phase,"cohorts":{},"by_source":{},"by_length":{}}
    if 'sampling' in spec:summary['sampling']=spec['sampling']
    summary['control_selection']=spec['control_selection']
    for cohort in ('external','legacy_control'):
        subset=[r for r in current if r['cohort']==cohort]
        summary['cohorts'][cohort]=summarize(subset,task)
    for source in sorted({r['source'] for r in current if r['cohort']=='external'}):
        summary['by_source'][source]=summarize([r for r in current if r['cohort']=='external' and r['source']==source],task)
    if task!='ocr':
        for label,lo,hi in [('1-40',0,40),('41-120',40,120),('121+',120,10**9)]:
            summary['by_length'][label]=summarize([r for r in current if r['cohort']=='external' and lo<len(r['src'])<=hi],task)
    if task=='translit':
        summary['fully_representable_input']=summarize([r for r in current if r['cohort']=='external' and not r.get('input_dropped_signs')],task)
        if fraction is not None:
            baseline,_=plans('translit')
            baseline_ids={r['eval_id'] for r in baseline if r['cohort']=='external'}
            summary['baseline_subset']=summarize([r for r in current if r['eval_id'] in baseline_ids],task)
    dump(folder/f'summary_{args.phase}.json',summary)
    print(json.dumps(summary['cohorts'],ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['mt','translit','ocr','all'],default='all')
    parser.add_argument('--phase',choices=['smoke','full'],default='smoke')
    parser.add_argument('--output',type=Path,default=Path('legacy/technical_runs/new_data'))
    parser.add_argument('--translit-fraction',type=float,default=None,
                        help='Retain the baseline and extend to at least this fraction of the external transliteration pool')
    parser.add_argument('--full-control',action='store_true',help='Evaluate every example of the original control test')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
    for task in (['mt','translit','ocr'] if args.task=='all' else [args.task]):evaluate(task,args)


if __name__=='__main__':main()
