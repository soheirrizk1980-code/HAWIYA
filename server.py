"""HAWIYA learning interface with truthful stage events from the original models.

Only run on 127.0.0.1. Uploaded images and their embeddings are not persisted. Only the public demo is cached.
Gallery photos are bundled with the project and served locally.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from engine import ImageProblem, SetupProblem, MAX_BYTES, model_status, analyze_bytes, analyze_with_progress, compare_bytes, compare_with_progress, DEMO_FILE, read_reference_cache, ensure_blue_reference
ROOT = Path(__file__).resolve().parent
app = FastAPI(title='HAWIYA — Biometric Recognition', docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')
app.mount('/fingerprint', StaticFiles(directory=ROOT / 'static' / 'fingerprint', html=True), name='fingerprint')
SAMPLES = {'original': {'title': 'HAWIYA reference · blue iris', 'credit': 'HAWIYA reference photograph', 'license': 'Existing project photograph'}, 'green': {'title': 'Green iris · Skitterphoto', 'credit': 'Skitterphoto via Wikimedia Commons', 'license': 'CC0'}, 'brown': {'title': 'Brown iris · Entertheninja', 'credit': 'Entertheninja via Wikimedia Commons', 'license': 'CC0'}, 'amber': {'title': 'Amber iris · Bobbyphotographer', 'credit': 'Bobbyphotographer via Wikimedia Commons', 'license': 'CC0'}, 'hazel': {'title': 'Hazel iris · Dzenanz', 'credit': 'Dzenanz via Wikimedia Commons', 'license': 'Public domain'}}

def sample_path(key: str) -> Path:
    if key not in SAMPLES:
        raise HTTPException(404, 'Unknown eye photograph')
    path = DEMO_FILE if key == 'original' else ROOT / 'static' / 'gallery' / f'{key}.jpg'
    if not path.is_file() or path.stat().st_size < 10000:
        raise HTTPException(404, f'Sample photo {key} is missing from the project.')
    return path

@app.get('/api/prepared')
async def prepared():
    data = await run_in_threadpool(read_reference_cache)
    if data is None:
        return {'ready': False, 'reason': 'Start HAWIYA with START_WINDOWS.bat or python server.py --ensure-blue to prepare the blue reference.'}
    return {'ready': True, 'samples': ['original'], 'analyses': {'original': data['analyses']['original']}, 'pairs': {}, 'defaultA': 'original', 'defaultB': 'green'}

@app.get('/')
async def home():
    return FileResponse(ROOT / 'static' / 'home.html')

@app.get('/iris')
@app.get('/iris/')
async def iris_lab():
    return FileResponse(ROOT / 'static' / 'index.html')

@app.get('/api/status')
async def status():
    return model_status()

@app.get('/api/samples')
async def samples():
    return [{'id': k, 'title': v['title'], 'credit': v['credit'], 'license': v['license']} for k, v in SAMPLES.items()]

@app.get('/api/samples/{key}')
async def sample_photo(key: str):
    path = await run_in_threadpool(sample_path, key)
    return FileResponse(path, media_type='image/jpeg', headers={'Cache-Control': 'public, max-age=1800'})

@app.get('/api/demo-eye')
async def demo_eye():
    return await sample_photo('original')

async def read_photo(file: UploadFile) -> bytes:
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'application/octet-stream'):
        raise HTTPException(415, 'Choose a JPEG, PNG, BMP, or WebP eye photo.')
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, 'Photo must be smaller than 9 MB.')
    return raw

@app.post('/api/analyze')
async def analyze(file: UploadFile=File(...)):
    data = await read_photo(file)
    try:
        return await run_in_threadpool(analyze_bytes, data)
    except ImageProblem as exc:
        raise HTTPException(422, str(exc)) from exc
    except SetupProblem as exc:
        raise HTTPException(503, str(exc)) from exc

@app.post('/api/compare')
async def compare(first: UploadFile=File(...), second: UploadFile=File(...)):
    a, b = (await read_photo(first), await read_photo(second))
    try:
        return await run_in_threadpool(compare_bytes, a, b)
    except ImageProblem as exc:
        raise HTTPException(422, str(exc)) from exc
    except SetupProblem as exc:
        raise HTTPException(503, str(exc)) from exc
_jobs = {}
_jobs_lock = RLock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='hawiya-lab')

def _clear_stale():
    expire = datetime.now(timezone.utc) - timedelta(minutes=25)
    with _jobs_lock:
        for k, v in list(_jobs.items()):
            if v['created'] < expire and v['status'] != 'running':
                _jobs.pop(k, None)
        if len(_jobs) > 10:
            for k, v in sorted(_jobs.items(), key=lambda x: x[1]['created'])[:len(_jobs) - 10]:
                if v['status'] != 'running':
                    _jobs.pop(k, None)

def _publish(jobid, phase, data=None):
    with _jobs_lock:
        rec = _jobs.get(jobid)
        if rec is None:
            return
        rec['phase'] = phase
        if data:
            rec['outputs'].update(data)
        rec['history'].append({'phase': phase, 'time': datetime.now(timezone.utc).isoformat()})

def _create_job(kind, work):
    _clear_stale()
    with _jobs_lock:
        active = sum((j['status'] == 'running' for j in _jobs.values()))
        if active >= 2:
            raise HTTPException(429, 'The AI is handling another image. Try again shortly.')
        jobid = uuid4().hex
        _jobs[jobid] = {'created': datetime.now(timezone.utc), 'status': 'running', 'kind': kind, 'phase': 'queued', 'history': [], 'outputs': {}, 'result': None, 'error': None}

    def run():
        try:
            result = work(jobid)
            with _jobs_lock:
                _jobs[jobid]['result'] = result
                _jobs[jobid]['status'] = 'done'
                _jobs[jobid]['phase'] = 'done'
        except Exception as e:
            with _jobs_lock:
                _jobs[jobid]['status'] = 'error'
                _jobs[jobid]['phase'] = 'error'
                _jobs[jobid]['error'] = str(e) if isinstance(e, (ImageProblem, SetupProblem)) else 'Processing failed. See the server log for details.'
    _executor.submit(run)
    return {'id': jobid}

@app.post('/api/jobs/analyze')
async def start_analysis(file: UploadFile=File(...)):
    raw = await read_photo(file)

    def work(jobid):
        return analyze_with_progress(raw, lambda phase, data: _publish(jobid, phase, data))
    return _create_job('analyze', work)

@app.post('/api/jobs/compare')
async def start_comparison(first: UploadFile=File(...), second: UploadFile=File(...)):
    a, b = (await read_photo(first), await read_photo(second))

    def work(jobid):
        return compare_with_progress(a, b, lambda phase, data: _publish(jobid, phase, data))
    return _create_job('compare', work)

@app.get('/api/jobs/{jobid}')
async def job_state(jobid: str):
    _clear_stale()
    with _jobs_lock:
        rec = _jobs.get(jobid)
        if rec is None:
            raise HTTPException(404, 'This analysis has expired. Start a new one.')
        return {'id': jobid, 'status': rec['status'], 'kind': rec['kind'], 'phase': rec['phase'], 'history': rec['history'][:], 'outputs': rec['outputs'].copy(), 'result': rec['result'], 'error': rec['error']}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HAWIYA one-time reference setup')
    parser.add_argument('--ensure-blue', action='store_true', help='Prepare the blue reference only if necessary')
    args = parser.parse_args()
    if args.ensure_blue:
        ensure_blue_reference()
        print('HAWIYA ready: blue reference cached; other eyes process live.', flush=True)
    else:
        parser.print_help()
