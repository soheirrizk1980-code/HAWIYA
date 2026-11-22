"""Compute and cache ORIGINAL YOLO + ArcIris outputs for distinct genuine iris photos.

One pass per photograph. Subsequent site visits only READ the cache.
No database, private biometrics, third-party servers, or artificial scores.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import os
import json
import numpy as np
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    root=args.project.resolve()
    sys.path.insert(0,str(root))
    os.environ['HAWIYA_AI_ROOT']=str(root/'HAWIYA_AI')
    from FETCH_REAL_EYES import ensure_gallery
    ensure_gallery(root)
    from server import SAMPLES, sample_path, demo_signature, CACHE_FILE
    import engine
    print('\nHAWIYA / PREPARING MODEL RESULTS FOR THE FULL GALLERY',flush=True)
    print('Real YOLO and ArcIris outputs only. Skipped photos are not displayed as analyzed.',flush=True)
    data={'signature':demo_signature(), 'analyses':{}, 'pairs':{}, 'sample_origin':'Distinct genuine public-domain/CC0 iris photographs (see IMAGE_SOURCES.txt)'}
    embeddings={}
    total=len(SAMPLES)
    for index,key in enumerate(SAMPLES,1):
        print(f'[{index}/{total}] {SAMPLES[key]["title"]}: YOLO segmentation + geometry + ArcIris...',flush=True)
        image=sample_path(key).read_bytes()
        try:
            with engine._LOCK:
                result,vectors=engine._single_locked(image,include_embedding=True)
        except Exception as exc:
            print(f'FAILED ON {key}: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
            if key == 'original':
                raise
            print('    Eye could not be analyzed by the trained models; skipped for preloaded results.',flush=True)
            continue
        if result.get('status')!='detected' or vectors is None:
            raise RuntimeError('Real inference did not produce an analysis')
        data['analyses'][key]=result
        embeddings[key]=vectors
        print(f'    OK: iris {result["geometry"]["iris"]} / pupil {result["geometry"]["pupil"]}',flush=True)
    keys=list(data['analyses'])
    if len(keys)<2:
        raise RuntimeError('No second distinct iris passed the model pipeline; a preloaded comparison cannot be published. Choose new images and rerun.')
    for a in keys:
        data['pairs'][a]={}
        for b in keys:
            data['pairs'][a][b]={
                'grayscale':round(float(np.dot(embeddings[a]['grayscale'],embeddings[b]['grayscale'])),6),
                'gamma':round(float(np.dot(embeddings[a]['gamma'],embeddings[b]['gamma'])),6)
            }
    CACHE_FILE.parent.mkdir(parents=True,exist_ok=True)
    tmp=CACHE_FILE.with_suffix('.writing')
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    tmp.replace(CACHE_FILE)
    print(f'\nSUCCESS: {len(keys)} distinct source photos + {len(keys)*len(keys)} pair results saved from REAL MODELS.',flush=True)
    print('SITE: http://127.0.0.1:8800\n',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('PREPARATION FAILED:',str(exc),file=sys.stderr)
        raise SystemExit(1)
