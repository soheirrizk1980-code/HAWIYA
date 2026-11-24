"""Original HAWIYA model adapter. Uploaded biometric images are never persisted.

Only the bundled public blue-eye demonstration has a private reusable cache.
The YOLO masks, hybrid crop, robust circle fit, Daugman-style normalization,
ArcIris transforms, and 512-D features are delegated to the user's original
hawiya_arciris_cpu_validation.py, not replaced with simulated scores.
"""
from __future__ import annotations
import base64
import hashlib
import json
from functools import lru_cache
import importlib.util
import io
import os
import threading
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parent
AI_ROOT = (Path(os.environ['HAWIYA_AI_ROOT']) if os.getenv('HAWIYA_AI_ROOT') else ROOT / 'HAWIYA_AI').resolve()
YOLO_FILE = AI_ROOT / 'models' / 'yolo_iris_v2.pt'
ARC_FILE = AI_ROOT / 'models' / 'ResNet100_154000.pt'
DRIVER_FILE = AI_ROOT / 'hawiya_arciris_cpu_validation.py'
NETWORK_FILE = AI_ROOT / 'OpenSourceIrisRecognition' / 'methods' / 'ArcIris' / 'Python' / 'modules' / 'network.py'
DEMO_FILE = AI_ROOT.parent / 'hawiya_eye_stock.jpg'
BLUE_CACHE_FILE = ROOT / 'demo_cache' / 'prepared_samples.json'
BLUE_CACHE_VERSION = 'HAWIYA-blue-reference-v2'
MAX_BYTES = 9 * 1024 * 1024
MAX_PIXELS = 16000000
_LOCK = threading.Lock()
_DRIVER = None
_ARC_MODEL = None

class ImageProblem(Exception):
    pass

class SetupProblem(Exception):
    pass

def model_status() -> dict:
    return {'ai_root': str(AI_ROOT), 'yolo': YOLO_FILE.exists(), 'arciris': ARC_FILE.exists(), 'driver': DRIVER_FILE.exists(), 'network': NETWORK_FILE.exists(), 'demo': DEMO_FILE.exists(), 'ready': all((p.is_file() for p in [YOLO_FILE, ARC_FILE, DRIVER_FILE, NETWORK_FILE]))}

def _driver():
    global _DRIVER
    if _DRIVER is not None:
        return _DRIVER
    missing = [str(p) for p in [YOLO_FILE, ARC_FILE, DRIVER_FILE, NETWORK_FILE] if not p.is_file()]
    if missing:
        raise SetupProblem('Required original HAWIYA files missing: ' + ', '.join(missing))
    from ultralytics import YOLO
    spec = importlib.util.spec_from_file_location('hawiya_original_core_v03', str(DRIVER_FILE))
    if spec is None or spec.loader is None:
        raise SetupProblem('Cannot load HAWIYA original image processing source')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = AI_ROOT
    mod.CKPT = ARC_FILE
    mod.YOLO = YOLO_FILE
    mod.REPO = NETWORK_FILE.parents[1]
    mod.PRED_DEVICE = 'cpu'
    mod.iris_yolo = YOLO(str(YOLO_FILE))
    if getattr(mod.iris_yolo, 'task', None) != 'segment':
        raise SetupProblem('The YOLO checkpoint is not a segmentation model; HAWIYA requires iris and pupil masks')
    names = getattr(mod.iris_yolo, 'names', {})
    name0 = str(names.get(0, '')).lower() if isinstance(names, dict) else ''
    name1 = str(names.get(1, '')).lower() if isinstance(names, dict) else ''
    if 'iris' not in name0 or 'pupil' not in name1:
        raise SetupProblem(f'Unexpected YOLO labels: {names}. Expected class 0 iris, class 1 pupil. Check before using.')
    _DRIVER = mod
    return mod

def _arciris():
    global _ARC_MODEL
    if _ARC_MODEL is None:
        mod = _driver()
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
        _ARC_MODEL = mod.load_arciris()
    return _ARC_MODEL

def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ImageProblem('Photo is empty or larger than 9 MB. Please retake or use a smaller image.')
    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            if img.width * img.height > MAX_PIXELS:
                raise ImageProblem('Photo is too large. Use an image below 16 megapixels.')
            if min(img.size) < 100:
                raise ImageProblem('Photo is too small. Please take a close-up of the eye.')
            rgb = np.array(img.convert('RGB'))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageProblem('Please select a valid JPEG or PNG eye photo.') from exc
    if not np.isfinite(rgb).all():
        raise ImageProblem('The photo contains invalid pixels.')
    return rgb

def _png_data_uri(image_rgb: np.ndarray, max_width: int=760) -> str:
    h, w = image_rgb.shape[:2]
    if w > max_width:
        new_h = max(1, int(round(h * max_width / w)))
        image_rgb = cv2.resize(image_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    good, buf = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not good:
        raise RuntimeError('Could not render model output')
    return 'data:image/png;base64,' + base64.b64encode(buf).decode('ascii')

def _image_report(rgb: np.ndarray) -> dict:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    messages = []
    if brightness < 48:
        messages.append('The eye looks dark. Try brighter, soft light.')
    if brightness > 215:
        messages.append('The eye may be overexposed. Avoid strong direct light.')
    if sharpness < 22:
        messages.append('The photo may be blurry. Try closer focus and hold the phone steady.')
    if contrast < 22:
        messages.append('Very low contrast. Try a clearer close-up.')
    return {'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]), 'brightness': round(brightness, 1), 'contrast': round(contrast, 1), 'sharpness': round(sharpness, 1), 'warnings': messages, 'note': 'Quality checks are basic image statistics, not a certified biometric quality score.'}

def _render_overlay(crop: np.ndarray, iris: tuple, pupil: tuple) -> np.ndarray:
    bgr = cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR)
    for circle, color in ((iris, (114, 217, 102)), (pupil, (66, 188, 255))):
        cx, cy, r = circle
        cv2.circle(bgr, (round(cx), round(cy)), round(r), color, 2, cv2.LINE_AA)
        cv2.circle(bgr, (round(cx), round(cy)), 3, color, -1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _render_segmentation_overlay(crop_rgb, iris_mask, pupil_mask):
    """Visualize the model's true masks; not manually drawn circles."""
    canvas = crop_rgb.copy()
    hh, ww = canvas.shape[:2]
    ir = np.asarray(iris_mask)
    pu = np.asarray(pupil_mask)
    if ir.shape != (hh, ww):
        ir = cv2.resize(ir.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)
    if pu.shape != (hh, ww):
        pu = cv2.resize(pu.astype(np.uint8), (ww, hh), interpolation=cv2.INTER_NEAREST)
    imask = ir > 0
    pmask = pu > 0
    if imask.any():
        green = np.array([48, 243, 190], dtype=np.float32)
        canvas[imask] = (0.62 * canvas[imask].astype(np.float32) + 0.38 * green).astype(np.uint8)
    if pmask.any():
        amber = np.array([255, 185, 99], dtype=np.float32)
        canvas[pmask] = (0.5 * canvas[pmask].astype(np.float32) + 0.5 * amber).astype(np.uint8)
    return canvas

def _feature_profile(embeddings):
    """Sixteen means of adjacent 32-value groups; no raw embedding returned."""
    if not embeddings or 'grayscale' not in embeddings:
        return None
    vector = np.asarray(embeddings['grayscale'], dtype=np.float32).reshape(-1)
    if vector.shape != (512,):
        return None
    return [round(float(np.mean(vector[i:i + 32])), 6) for i in range(0, 512, 32)]

def _single_locked(data: bytes, include_embedding: bool=True, progress=None) -> tuple[dict, dict | None]:
    rgb = decode_image(data)
    quality = _image_report(rgb)
    if progress is not None:
        progress('quality', {'quality': quality})
    mod = _driver()
    try:
        crop, pair, bbox = mod.hybrid_eye_crop_live(rgb)
        crop_uri = segmentation_uri = None
        if progress is not None:
            crop_uri = _png_data_uri(crop)
            segmentation_uri = _png_data_uri(_render_segmentation_overlay(crop, pair['iris']['mask'], pair['pupil']['mask']))
            progress('segmentation', {'crop': crop_uri, 'segmentation': segmentation_uri, 'confidence': {'iris': round(float(pair['iris']['confidence']), 4), 'pupil': round(float(pair['pupil']['confidence']), 4)}})
        normalized, iris, pupil = mod.normalize_iris_rgb(crop, pair['iris']['mask'], pair['pupil']['mask'])
    except (ValueError, AssertionError, cv2.error) as exc:
        raise ImageProblem('YOLO could not find a valid iris and pupil in this photo. ' + 'Use a clear close-up with the whole iris visible. Detail: ' + str(exc)[:160]) from exc
    detected_uri = None
    if progress is not None:
        detected_uri = _png_data_uri(_render_overlay(crop, iris, pupil))
        progress('circles', {'detected': detected_uri, 'geometry': {'iris': list(map(float, iris)), 'pupil': list(map(float, pupil))}})
    gray = mod.to_grayscale3(normalized)
    gamma = mod.to_gamma075(normalized)
    gray_uri = gamma_uri = None
    if progress is not None:
        gray_uri, gamma_uri = (_png_data_uri(gray, 1024), _png_data_uri(gamma, 1024))
        progress('unwrap', {'normalized_gray': gray_uri, 'normalized_gamma': gamma_uri})
    if include_embedding:
        arc = _arciris()
        embs = {'grayscale': mod.to_embedding(arc, gray), 'gamma': mod.to_embedding(arc, gamma)}
        for vec in embs.values():
            if vec.shape != (512,):
                raise SetupProblem('ArcIris returned an unexpected embedding shape')
    else:
        embs = None
    profile = _feature_profile(embs)
    if progress is not None and embs is not None:
        progress('encoded', {'feature_profile': profile, 'embedding_dimensions': 512})
    crop_uri = crop_uri if crop_uri is not None else _png_data_uri(crop)
    segmentation_uri = segmentation_uri if segmentation_uri is not None else _png_data_uri(_render_segmentation_overlay(crop, pair['iris']['mask'], pair['pupil']['mask']))
    detected_uri = detected_uri if detected_uri is not None else _png_data_uri(_render_overlay(crop, iris, pupil))
    gray_uri = gray_uri if gray_uri is not None else _png_data_uri(gray, 1024)
    gamma_uri = gamma_uri if gamma_uri is not None else _png_data_uri(gamma, 1024)
    payload = {'status': 'detected', 'model': 'Original HAWIYA YOLO iris V2 + ArcIris ResNet100', 'quality': quality, 'confidence': {'iris': round(float(pair['iris']['confidence']), 4), 'pupil': round(float(pair['pupil']['confidence']), 4)}, 'geometry': {'iris': [round(float(x), 2) for x in iris], 'pupil': [round(float(x), 2) for x in pupil], 'crop_box': [int(x) for x in bbox]}, 'crop': crop_uri, 'segmentation': segmentation_uri, 'detected': detected_uri, 'normalized_gray': gray_uri, 'normalized_gamma': gamma_uri, 'embedding_dimensions': 512 if embs is not None else None, 'feature_profile': profile, 'method': ['Original trained YOLO segmentation', 'Original HAWIYA hybrid crop', 'Robust circle fit', '64x512 rubber-sheet normalization', 'ArcIris ResNet100 (512-D)'] if include_embedding else ['YOLO', 'Original rubber-sheet normalization']}
    return (payload, embs)

def _reference_files():
    return (('driver', DRIVER_FILE), ('network', NETWORK_FILE), ('YOLO', YOLO_FILE), ('ArcIris', ARC_FILE), ('blue-image', DEMO_FILE))

@lru_cache(maxsize=2)
def _hash_reference_files(fingerprint):
    """Hash content once per process/file version, not on every /api/prepared call."""
    h = hashlib.sha256(BLUE_CACHE_VERSION.encode('utf-8') + b'\x00')
    for label, path in _reference_files():
        h.update(label.encode('utf-8') + b'\x00')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
    return h.hexdigest()

def reference_signature():
    try:
        versions = tuple(((str(path), path.stat().st_size, path.stat().st_mtime_ns) for _, path in _reference_files()))
        return _hash_reference_files(versions)
    except OSError:
        return 'missing'

@lru_cache(maxsize=2)
def _read_reference_file(signature, size, mtime_ns):
    try:
        data = json.loads(BLUE_CACHE_FILE.read_text(encoding='utf-8'))
        if data.get('cache_version') != BLUE_CACHE_VERSION or data.get('signature') != signature:
            return None
        if set(data.get('analyses', {})) != {'original'} or data.get('pairs') != {}:
            return None
        if data['analyses']['original'].get('status') != 'detected':
            return None
        internal = data.get('_internal', {})
        if set(internal.get('embeddings', {})) != {'grayscale', 'gamma'}:
            return None
        if any((len(values) != 512 for values in internal['embeddings'].values())):
            return None
        if not isinstance(internal.get('stages'), list):
            return None
        return data
    except (ValueError, OSError, KeyError, TypeError):
        return None

def read_reference_cache():
    signature = reference_signature()
    if signature == 'missing':
        return None
    try:
        stat = BLUE_CACHE_FILE.stat()
    except OSError:
        return None
    return _read_reference_file(signature, stat.st_size, stat.st_mtime_ns)

def ensure_blue_reference():
    """Compute exactly one real blue-eye analysis if a valid private cache is absent."""
    if read_reference_cache() is not None:
        return
    with _LOCK:
        if read_reference_cache() is not None:
            return
        raw = DEMO_FILE.read_bytes()
        stages = []
        payload, embeddings = _single_locked(raw, progress=lambda phase, data: stages.append({'phase': phase, 'data': data}))
        if payload.get('status') != 'detected' or payload.get('embedding_dimensions') != 512:
            raise SetupProblem('Blue reference analysis did not produce a 512-D embedding.')
        cache = {'cache_version': BLUE_CACHE_VERSION, 'signature': reference_signature(), 'image_sha256': hashlib.sha256(raw).hexdigest(), 'analyses': {'original': payload}, 'pairs': {}, 'sample_origin': 'Original HAWIYA reference photograph', '_internal': {'embeddings': {key: value.astype(np.float32).tolist() for key, value in embeddings.items()}, 'stages': stages}}
        BLUE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        target = BLUE_CACHE_FILE.with_suffix('.writing')
        target.write_text(json.dumps(cache, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        os.chmod(target, 384)
        target.replace(BLUE_CACHE_FILE)
        _read_reference_file.cache_clear()
    if read_reference_cache() is None:
        raise SetupProblem('Blue reference cache could not be verified.')

@lru_cache(maxsize=2)
def _blue_source_hash(size, mtime_ns):
    return hashlib.sha256(DEMO_FILE.read_bytes()).digest()

def _single_or_blue_cache(data, progress=None):
    """Reuse saved outputs + exact float32 vectors for the unchanged blue demo."""
    try:
        stat = DEMO_FILE.stat()
        is_blue = len(data) == stat.st_size and hashlib.sha256(data).digest() == _blue_source_hash(stat.st_size, stat.st_mtime_ns)
    except OSError:
        is_blue = False
    if is_blue:
        saved = read_reference_cache()
        if saved is not None:
            if progress is not None:
                for event in saved['_internal']['stages']:
                    progress(event['phase'], event['data'])
            embeddings = {key: np.asarray(value, dtype=np.float32) for key, value in saved['_internal']['embeddings'].items()}
            return (saved['analyses']['original'], embeddings)
    return _single_locked(data, progress=progress)

def analyze_bytes(data: bytes) -> dict:
    with _LOCK:
        return _single_or_blue_cache(data)[0]

def analyze_with_progress(data: bytes, progress):
    with _LOCK:
        return _single_or_blue_cache(data, progress=progress)[0]

def _compare(a: bytes, b: bytes, progress=None) -> dict:
    with _LOCK:
        stage_a = (lambda phase, data: progress('eye_a_' + phase, data)) if progress else None
        stage_b = (lambda phase, data: progress('eye_b_' + phase, data)) if progress else None
        first, f1 = _single_or_blue_cache(a, progress=stage_a)
        second, f2 = _single_or_blue_cache(b, progress=stage_b)
    sim_g = float(np.dot(f1['grayscale'], f2['grayscale']))
    sim_ga = float(np.dot(f1['gamma'], f2['gamma']))
    if sim_g >= 0.3078 and sim_ga >= 0.31653:
        category = 'HIGH_SIMILARITY_IN_DEVELOPMENT_TEST'
    elif sim_g < 0.218898 and sim_ga < 0.22556365:
        category = 'LOW_SIMILARITY_IN_DEVELOPMENT_TEST'
    else:
        category = 'INCONCLUSIVE'
    explanation = '512-dimensional features per view were computed by ArcIris; cosine similarity is not an identity probability.' if progress is not None else 'ArcIris produced two 512-number vectors for each eye. Their cosine similarity is the dot product of L2-normalized vectors. Experimental cutoffs come from a tiny developmental dataset. Human review is required.'
    return {'first': first, 'second': second, 'similarity': {'grayscale': round(sim_g, 6), 'gamma': round(sim_ga, 6)}, 'research_category': category, 'verified_identity': False, 'explanation': explanation}

def compare_bytes(a: bytes, b: bytes) -> dict:
    return _compare(a, b)

def compare_with_progress(first_image: bytes, second_image: bytes, progress):
    return _compare(first_image, second_image, progress=progress)
