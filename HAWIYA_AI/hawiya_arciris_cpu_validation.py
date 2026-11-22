#!/usr/bin/env python3
"""HAWIYA Pi CPU validation. Ported unchanged geometry/crop from V31.
Outputs are PRIVATE development artifacts, not verified identities.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import sys, argparse, json, math, hashlib, time
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import least_squares
import torch
import torch.nn.functional as F

ROOT=Path.home()/'HAWIYA_AI'
CKPT=ROOT/'models'/'ResNet100_154000.pt'
YOLO=ROOT/'models'/'yolo_iris_v2.pt'
REPO=ROOT/'OpenSourceIrisRecognition'/'methods'/'ArcIris'/'Python'
OUT=ROOT/'private_cpu_test'
PRED_DEVICE='cpu'

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):
            h.update(part)
    return h.hexdigest()

def load_arciris():
    assert REPO.joinpath('modules','network.py').is_file(), 'Official ArcIris network.py missing; clone CVRL repository first'
    assert sha256_file(CKPT)=='466014be7723eb0b1a741dcff5a0c8ae51d1918818a6b021952f9c4ccd443f8f', 'ArcIris SHA256 mismatch'
    sys.path.insert(0,str(REPO))
    from modules.network import iresnet100
    model=iresnet100(pretrained=False,progress=False)
    state=torch.load(str(CKPT),map_location='cpu',weights_only=True)
    assert isinstance(state,dict), 'Unexpected checkpoint structure'
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True)
    model=model.cpu().float().eval()
    print('ArcIris official ResNet100 VERIFIED strict=True | CPU FP32',flush=True)
    return model

def to_embedding(model,img_rgb):
    assert img_rgb.shape==(64,512,3)
    arr=np.ascontiguousarray(img_rgb,dtype=np.float32)/255.0
    x=torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    x=(x-0.5)/0.5
    with torch.inference_mode():
        feature=model(x)
    if isinstance(feature,(list,tuple)): feature=feature[0]
    feature=F.normalize(feature.detach().float(),dim=-1).squeeze(0).cpu().numpy()
    if not np.isfinite(feature).all(): raise ValueError('Non-finite embedding')
    return feature

def fit_circle_robust(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('Empty segmentation mask')
    contour = max(contours, key=cv2.contourArea)
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 5:
        raise ValueError('Insufficient contour points')
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x*x + y*y
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r0 = np.sqrt(max(0, c + cx*cx + cy*cy))

    def residual(p):
        return np.hypot(x - p[0], y - p[1]) - p[2]

    res = least_squares(
        residual, [cx, cy, max(r0, 1)],
        bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, np.inf]),
        loss='soft_l1', f_scale=2.0, max_nfev=200,
    )
    if not res.success:
        raise ValueError('Robust fitting failed')
    return tuple(float(v) for v in res.x)

def validate_geometry(icx, icy, ir, pcx, pcy, pr):
    vals = np.array([icx, icy, ir, pcx, pcy, pr], dtype=np.float32)
    if not np.isfinite(vals).all():
        return False
    if ir <= 5 or pr <= 2:
        return False
    ratio = pr / ir
    if not (0.12 <= ratio <= 0.75):
        return False
    if math.hypot(icx-pcx, icy-pcy) + pr > ir * 1.10:
        return False
    return True

def normalize_iris_rgb(crop_rgb, iris_mask, pupil_mask):
    icx, icy, ir = fit_circle_robust(iris_mask)
    pcx, pcy, pr = fit_circle_robust(pupil_mask)
    if not validate_geometry(icx, icy, ir, pcx, pcy, pr):
        raise ValueError('Invalid iris/pupil geometry')

    theta = np.linspace(0, 2*np.pi, 512, endpoint=False)
    rho = np.linspace(0, 1, 64)
    t, r = np.meshgrid(theta, rho)

    px = pcx + pr*np.cos(t)
    py = pcy + pr*np.sin(t)
    ix = icx + ir*np.cos(t)
    iy = icy + ir*np.sin(t)

    map_x = ((1-r) * px + r * ix).astype(np.float32)
    map_y = ((1-r) * py + r * iy).astype(np.float32)

    normalized = cv2.remap(crop_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return normalized, (icx, icy, ir), (pcx, pcy, pr)

def to_grayscale3(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

def to_gamma075(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gamma = np.clip(255.0 * ((gray.astype(np.float32) / 255.0) ** 0.75), 0, 255).astype(np.uint8)
    return cv2.cvtColor(gamma, cv2.COLOR_GRAY2RGB)

def masks_and_geometry(predict_result, shape):
    if predict_result.masks is None or predict_result.boxes is None:
        return []
    hh, ww = shape[:2]
    cls = predict_result.boxes.cls.cpu().numpy().astype(int)
    conf = predict_result.boxes.conf.cpu().numpy()
    ms = predict_result.masks.data.cpu().numpy()
    found = []
    for i, c in enumerate(cls):
        if c not in (0, 1):
            continue
        mask = cv2.resize(ms[i], (ww, hh), interpolation=cv2.INTER_NEAREST)
        mask = (mask > .5).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 15:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(contour)
        found.append(dict(cls=int(c), mask=mask, center=np.array([cx,cy]),
                          radius=float(rad), confidence=float(conf[i])))
    return found

def selected_iris_pair(image):
    pred = iris_yolo.predict(source=image, conf=.05, imgsz=640,
                             device=PRED_DEVICE, verbose=False)[0]
    detections = masks_and_geometry(pred, image.shape)
    iris_items = [d for d in detections if d['cls']==0]
    pupil_items = [d for d in detections if d['cls']==1]
    best = None
    for iris in iris_items:
        for pupil in pupil_items:
            ir, pr = iris['radius'], pupil['radius']
            if ir <= 5 or pr <= 2 or not (.12 <= pr/ir <= .75):
                continue
            distance = np.linalg.norm(iris['center']-pupil['center'])
            if distance+pr > ir*1.08:
                continue
            h,w = image.shape[:2]
            cx,cy=iris['center']
            if not (0<=cx<w and 0<=cy<h):
                continue
            quality = (iris['confidence']+pupil['confidence']-.3*distance/max(ir,1))
            if best is None or quality > best['quality']:
                best = dict(iris=iris, pupil=pupil, quality=float(quality))
    return best

def eye_box(cx, cy, radius, shape, margin=1.2):
    h,w=shape[:2]
    d=max(20., radius*(1+margin))
    x1=max(0,int(cx-d));y1=max(0,int(cy-d))
    x2=min(w,int(cx+d));y2=min(h,int(cy+d))
    return (x1,y1,x2,y2)

def hough_proposals(image, max_count=10):
    # Hough is a fallback only, not a learned iris detector.
    h,w=image.shape[:2]
    scale=min(1.,1200/max(h,w))
    small=cv2.resize(image,(int(w*scale),int(h*scale))) if scale<1 else image
    gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
    gray=cv2.medianBlur(gray,5)
    mn=max(7,int(min(gray.shape[:2])*.012))
    mx=max(mn+2,int(min(gray.shape[:2])*.38))
    circ=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,
        minDist=max(20,int(min(gray.shape[:2])*.12)),param1=90,param2=25,
        minRadius=mn,maxRadius=mx)
    proposals=[]
    if circ is not None:
        for cx,cy,r in circ[0][:max_count]:
            proposals.append((float(cx/scale),float(cy/scale),float(r/scale)))
    return proposals

def hybrid_eye_crop_live(image):
    h,w=image.shape[:2]
    if min(h,w)<100:
        raise ValueError('Image too small; retake a sharper close-up photo.')
    if max(h,w)>4000:
        ratio=4000/max(h,w)
        image=cv2.resize(image,(round(w*ratio),round(h*ratio)),
                         interpolation=cv2.INTER_AREA)
    candidate_boxes=[]
    first=selected_iris_pair(image)
    if first is not None:
        ir=first['iris']
        candidate_boxes.append(eye_box(*ir['center'],ir['radius'],image.shape,margin=1.0))
    for cx,cy,r in hough_proposals(image):
        b=eye_box(cx,cy,r,image.shape,margin=1.0)
        if b not in candidate_boxes:
            candidate_boxes.append(b)
    if not candidate_boxes:
        raise ValueError('Iris not found. Ensure the entire eye is visible and in focus.')
    best=None
    # Keep at most 12 proposals to limit GPU time, as in the V2 hybrid approach.
    for b in candidate_boxes[:12]:
        x1,y1,x2,y2=b
        crop=image[y1:y2,x1:x2]
        if min(crop.shape[:2])<50:
            continue
        ver=selected_iris_pair(crop)
        if ver is None:
            continue
        ch,cw=crop.shape[:2]
        cx,cy=ver['iris']['center']
        center_penalty=np.hypot((cx-cw/2)/max(cw,1),(cy-ch/2)/max(ch,1))
        quality=ver['quality']-.25*center_penalty
        if best is None or quality>best['quality']:
            best=dict(quality=quality,box=b,verified=ver)
    if best is None:
        raise ValueError('No valid iris/pupil pair in eye crops. Retake the photo.')
    x1,y1,_,_=best['box']
    iris=best['verified']['iris']
    c=iris['center']+np.array([x1,y1])
    final_box=eye_box(c[0],c[1],iris['radius'],image.shape,margin=1.20)
    x1,y1,x2,y2=final_box
    crop=image[y1:y2,x1:x2].copy()
    final=selected_iris_pair(crop)
    if final is None:
        raise ValueError('Segmentation failed on final crop; retake photograph.')
    # The exact selected masks are carried directly into the robust circle fitting.
    return crop, final, final_box

def first_photo():
    for f in sorted((Path.home()/'Iris_Dataset').rglob('*')):
        if f.suffix.lower() in {'.jpg','.jpeg','.png'}: return f
    return None

def run_prepare(photo):
    from ultralytics import YOLO as YOLOClass
    global iris_yolo
    iris_yolo=YOLOClass(str(YOLO))
    # Read actual camera photo, not synthetic input
    bgr=cv2.imread(str(photo),cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError('Cannot open image: '+str(photo))
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    print('Input:',photo,'resolution:',rgb.shape[1],rgb.shape[0],flush=True)
    t0=time.monotonic()
    crop,pair,box=hybrid_eye_crop_live(rgb)
    normalized,ic,pc=normalize_iris_rgb(crop,pair['iris']['mask'],pair['pupil']['mask'])
    gray=to_grayscale3(normalized)
    gamma=to_gamma075(normalized)
    assert gray.shape==(64,512,3) and gamma.shape==(64,512,3)
    OUT.mkdir(parents=True,exist_ok=True)
    OUT.chmod(0o700)
    cv2.imwrite(str(OUT/'eye_crop.png'),cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/'normalized_grayscale.png'),cv2.cvtColor(gray,cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT/'normalized_gamma075.png'),cv2.cvtColor(gamma,cv2.COLOR_RGB2BGR))
    (OUT/'source_info.json').write_text(json.dumps({'source':str(photo),'source_sha256':sha256_file(photo), 'crop_box':box,'iris_circle':ic,'pupil_circle':pc},indent=2))
    print('YOLO + HYBRID CROP + RUBBER SHEET OK',flush=True)
    print('Preparation seconds:',round(time.monotonic()-t0,2),flush=True)
    print('Normalized strips:',OUT/'normalized_grayscale.png',OUT/'normalized_gamma075.png',sep='\n',flush=True)
    print('Inspect the resulting crop/strips before trusting this embedding.',flush=True)

def run_embed():
    torch.set_num_threads(2)
    model=load_arciris()
    rng=OUT/'normalized_grayscale.png'
    rna=OUT/'normalized_gamma075.png'
    if not rng.exists() or not rna.exists():
        raise FileNotFoundError('Run prepare first; normalized strips not found')
    results={}
    for label,path in [('Grayscale3',rng),('Gamma075',rna)]:
        img=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if img is None: raise ValueError('Cannot read normalized strip '+str(path))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        t0=time.monotonic()
        emb=to_embedding(model,img)
        np.save(OUT/(label+'_embedding.npy'),emb)
        print(f'{label}: {emb.size} dimensions | norm={np.linalg.norm(emb):.6f} | CPU seconds={time.monotonic()-t0:.2f}',flush=True)
        results[label]={'dimensions':int(emb.size),'norm':float(np.linalg.norm(emb))}
    (OUT/'embedding_report.json').write_text(json.dumps(results,indent=2))
    print('REAL PHOTO EMBEDDINGS EXTRACTED | no identity matching performed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','embed','check'])
    p.add_argument('--image',type=Path,default=None)
    a=p.parse_args()
    if a.phase=='check':
        torch.set_num_threads(2)
        model=load_arciris()
        # A tensor test confirms forward inference, not genuine iris processing.
        x=torch.zeros(1,3,64,512,dtype=torch.float32)
        with torch.inference_mode(): y=model(x)
        print('Forward pass shape:',tuple(y.shape),flush=True)
        print('ARCIRIS CPU CHECK PASSED',flush=True)
    elif a.phase=='prepare':
        photo=a.image or first_photo()
        if photo is None: raise FileNotFoundError('No photos under ~/Iris_Dataset. Pass --image /path/to/photo.jpg')
        run_prepare(photo)
    else: run_embed()
