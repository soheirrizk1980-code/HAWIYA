"""Original HAWIYA model adapter, Windows-local. No biometric data stored.

The YOLO masks, hybrid crop, robust circle fit, Daugman-style normalization,
ArcIris transforms, and 512-D features are delegated to the user's original
hawiya_arciris_cpu_validation.py, not replaced with simulated scores.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import math
import os
import sys
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

MAX_BYTES = 9 * 1024 * 1024
MAX_PIXELS = 16_000_000
_LOCK = threading.Lock()
_DRIVER = None
_ARC_MODEL = None

class ImageProblem(Exception):
    pass

class SetupProblem(Exception):
    pass


def model_status() -> dict:
    return {
        'ai_root': str(AI_ROOT),
        'yolo': YOLO_FILE.exists(),
        'arciris': ARC_FILE.exists(),
        'driver': DRIVER_FILE.exists(),
        'network': NETWORK_FILE.exists(),
        'demo': DEMO_FILE.exists(),
        'ready': all(p.is_file() for p in [YOLO_FILE, ARC_FILE, DRIVER_FILE, NETWORK_FILE]),
    }


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
        _ARC_MODEL = mod.load_arciris()  # includes original official checkpoint SHA256 validation
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


def _png_data_uri(image_rgb: np.ndarray, max_width: int = 760) -> str:
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
    if brightness < 48: messages.append('The eye looks dark. Try brighter, soft light.')
    if brightness > 215: messages.append('The eye may be overexposed. Avoid strong direct light.')
    if sharpness < 22: messages.append('The photo may be blurry. Try closer focus and hold the phone steady.')
    if contrast < 22: messages.append('Very low contrast. Try a clearer close-up.')
    return {'width': int(rgb.shape[1]), 'height': int(rgb.shape[0]),
            'brightness': round(brightness, 1), 'contrast': round(contrast, 1),
            'sharpness': round(sharpness, 1), 'warnings': messages,
            'note': 'Quality checks are basic image statistics, not a certified biometric quality score.'}


def _render_overlay(crop: np.ndarray, iris: tuple, pupil: tuple) -> np.ndarray:
    bgr = cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR)
    for circle, color in ((iris, (114, 217, 102)), (pupil, (66, 188, 255))):
        cx, cy, r = circle
        cv2.circle(bgr, (round(cx),round(cy)),round(r), color, 2, cv2.LINE_AA)
        cv2.circle(bgr,(round(cx),round(cy)), 3, color, -1, cv2.LINE_AA)
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
        canvas[pmask] = (0.50 * canvas[pmask].astype(np.float32) + 0.50 * amber).astype(np.uint8)
    return canvas


def _feature_profile(embeddings):
    """Sixteen means of adjacent 32-value groups; no raw embedding returned."""
    if not embeddings or 'grayscale' not in embeddings:
        return None
    vector = np.asarray(embeddings['grayscale'], dtype=np.float32).reshape(-1)
    if vector.shape != (512,):
        return None
    return [round(float(np.mean(vector[i:i+32])), 6) for i in range(0, 512, 32)]

def _single_locked(data: bytes, include_embedding: bool = True, progress=None) -> tuple[dict, dict | None]:
    rgb = decode_image(data)
    quality = _image_report(rgb)
    if progress is not None:
        progress('quality', {'quality': quality})
    mod = _driver()
    try:
        crop, pair, bbox = mod.hybrid_eye_crop_live(rgb)
        if progress is not None:
            progress('segmentation', {
                'crop': _png_data_uri(crop),
                'segmentation': _png_data_uri(_render_segmentation_overlay(crop, pair['iris']['mask'], pair['pupil']['mask'])),
                'confidence': {'iris': round(float(pair['iris']['confidence']),4), 'pupil': round(float(pair['pupil']['confidence']),4)}
            })
        normalized, iris, pupil = mod.normalize_iris_rgb(crop, pair['iris']['mask'], pair['pupil']['mask'])
    except (ValueError, AssertionError, cv2.error) as exc:
        raise ImageProblem('YOLO could not find a valid iris and pupil in this photo. ' +
                           'Use a clear close-up with the whole iris visible. Detail: ' + str(exc)[:160]) from exc
    if progress is not None:
        progress('circles', {'detected': _png_data_uri(_render_overlay(crop, iris, pupil)),
            'geometry': {'iris': list(map(float,iris)), 'pupil': list(map(float,pupil))}})
    gray = mod.to_grayscale3(normalized)
    gamma = mod.to_gamma075(normalized)
    if progress is not None:
        progress('unwrap', {'normalized_gray': _png_data_uri(gray,1024), 'normalized_gamma': _png_data_uri(gamma,1024)})
    score = None
    if include_embedding:
        arc = _arciris()
        embs = {'grayscale': mod.to_embedding(arc, gray), 'gamma': mod.to_embedding(arc, gamma)}
        for vec in embs.values():
            if vec.shape != (512,):
                raise SetupProblem('ArcIris returned an unexpected embedding shape')
    else:
        embs = None
    if progress is not None and embs is not None:
        progress('encoded', {'feature_profile': _feature_profile(embs), 'embedding_dimensions': 512})
    payload = {
        'status': 'detected', 'model': 'Original HAWIYA YOLO iris V2 + ArcIris ResNet100',
        'quality': quality,
        'confidence': {'iris': round(float(pair['iris']['confidence']), 4), 'pupil': round(float(pair['pupil']['confidence']), 4)},
        'geometry': {'iris': [round(float(x), 2) for x in iris], 'pupil': [round(float(x),2) for x in pupil],
                     'crop_box': [int(x) for x in bbox]},
        'crop': _png_data_uri(crop), 'segmentation': _png_data_uri(_render_segmentation_overlay(crop, pair['iris']['mask'], pair['pupil']['mask'])), 'detected': _png_data_uri(_render_overlay(crop, iris, pupil)),
        'normalized_gray': _png_data_uri(gray, 1024), 'normalized_gamma': _png_data_uri(gamma, 1024),
        'embedding_dimensions': 512 if embs is not None else None,
        'feature_profile': _feature_profile(embs),
        'method': ['Original trained YOLO segmentation', 'Original HAWIYA hybrid crop',
                   'Robust circle fit', '64x512 rubber-sheet normalization',
                   'ArcIris ResNet100 (512-D)'] if include_embedding else ['YOLO', 'Original rubber-sheet normalization'],
    }
    return payload, embs


def analyze_bytes(data: bytes) -> dict:
    with _LOCK:
        result, _ = _single_locked(data)
        return result


def compare_bytes(a: bytes, b: bytes) -> dict:
    with _LOCK:
        first, f1 = _single_locked(a)
        second, f2 = _single_locked(b)
    sim_g = float(np.dot(f1['grayscale'], f2['grayscale']))
    sim_ga = float(np.dot(f1['gamma'], f2['gamma']))
    # These experimental cutoffs are taken from the original HAWIYA CPU development tests.
    # They are NOT population-calibrated, so we never output a verified identity claim.
    if sim_g >= .30780 and sim_ga >= .31653:
        category = 'HIGH_SIMILARITY_IN_DEVELOPMENT_TEST'
    elif sim_g < .218898 and sim_ga < .22556365:
        category = 'LOW_SIMILARITY_IN_DEVELOPMENT_TEST'
    else:
        category = 'INCONCLUSIVE'
    return {'first': first, 'second': second,
            'similarity': {'grayscale': round(sim_g, 6), 'gamma': round(sim_ga, 6)},
            'research_category': category, 'verified_identity': False,
            'explanation': 'ArcIris produced two 512-number vectors for each eye. Their cosine similarity is the dot product of L2-normalized vectors. Experimental cutoffs come from a tiny developmental dataset. Human review is required.'}


def analyze_with_progress(data: bytes, progress):
    """Publish each stage only after computing it using the original models."""
    with _LOCK:
        result, _ = _single_locked(data, progress=progress)
        return result


def compare_with_progress(first_image: bytes, second_image: bytes, progress):
    """Compare two eyes with real model stages; scores are not identity probabilities."""
    with _LOCK:
        first, f1 = _single_locked(first_image, progress=lambda phase, data: progress('eye_a_' + phase, data))
        second, f2 = _single_locked(second_image, progress=lambda phase, data: progress('eye_b_' + phase, data))
    sim_g = float(np.dot(f1['grayscale'], f2['grayscale']))
    sim_ga = float(np.dot(f1['gamma'], f2['gamma']))
    if sim_g >= .30780 and sim_ga >= .31653:
        category = 'HIGH_SIMILARITY_IN_DEVELOPMENT_TEST'
    elif sim_g < .218898 and sim_ga < .22556365:
        category = 'LOW_SIMILARITY_IN_DEVELOPMENT_TEST'
    else:
        category = 'INCONCLUSIVE'
    return {'first': first, 'second': second,
            'similarity': {'grayscale': round(sim_g, 6), 'gamma': round(sim_ga, 6)},
            'research_category': category, 'verified_identity': False,
            'explanation': '512-dimensional features per view were computed by ArcIris; cosine similarity is not an identity probability.'}
