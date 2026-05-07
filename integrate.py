# =====================================================================
# integrate.py | CNN + MediaPipe + YOLO 통합 파이프라인 UI
# 자세기준서 v1.0 기반 8개 지표 판정 + 오버레이 시각화
# 측면 이미지 기준 visibility 높은 쪽 자동 선택
# 실행: python integrate.py
# =====================================================================
import os, sys, cv2, warnings
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── 경로 설정 ─────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
MP_MODEL     = os.path.join(BASE, 'MediaPipe', 'models', 'pose_landmarker_full.task')
MLP_MODEL    = os.path.join(BASE, 'MediaPipe', 'models', 'posture_mlp_final.keras')
MLP_SCALER   = os.path.join(BASE, 'MediaPipe', 'models', 'posture_scaler.pkl')
YOLO_MODEL   = os.path.join(BASE, 'YOLO', 'fit_me_up', 'combined_gpu', 'weights', 'best.pt')

# ── 자세기준서 임계값 ─────────────────────────────────────────────────
THRESHOLD_CNN   = 0.60
THRESHOLD_CVA   = 20.0
THRESHOLD_TIA   = 10.0
RANGE_ELBOW     = (90, 120)
RANGE_KNEE      = (85, 100)
THRESHOLD_WRIST = 15.0
RANGE_GAZE      = (10, 15)
THRESHOLD_DESK  = 0.10
THRESHOLD_CHAIR = 0.20
VISIBILITY_MIN  = 0.40

# ── 오버레이 색상 (BGR) ───────────────────────────────────────────────
COLOR_GOOD      = ( 40, 200,  80)   # 밝은 초록
COLOR_BAD       = ( 50,  60, 230)   # 선명한 빨강
COLOR_TARGET    = ( 50, 230, 180)   # 목표 민트 초록
COLOR_NA        = (180, 180, 180)   # 측정불가 회색
COLOR_BONE      = (180, 180, 180)   # 뼈대 연결선
COLOR_HUD_BG    = ( 20,  15,  40)   # HUD 배경
OFFSET_PX       = 40

# ── YOLO 클래스 ───────────────────────────────────────────────────────
YOLO_CLASSES = {0: 'chair', 1: 'desk', 2: 'monitor'}

# ── 피드백 메시지 ─────────────────────────────────────────────────────
FEEDBACK = {
    'cva':     {'range': '0 ~ 20°',
                'good': '머리와 경추가 수직 정렬되어 경추 추간판 하중이 최소화된 상태입니다.',
                'bad':  '모니터를 눈높이에 맞춰 올리고, 시선이 수평 하방 10~15° 범위에 오도록 조정하세요.',
                'na':   '관절 가시성이 낮아 측정할 수 없습니다. 측면 이미지를 다시 촬영하세요.'},
    'tia':     {'range': '0 ~ 10°',
                'good': '몸통이 수직에 가깝게 유지되어 요추 압박이 최소화된 최적 자세입니다.',
                'bad':  '의자 깊숙이 앉아 등받이에 허리를 완전히 기대세요. 요추 쿠션 사용을 권장합니다.',
                'na':   '관절 가시성이 낮아 측정할 수 없습니다.'},
    'elbow':   {'range': '90 ~ 120°',
                'good': '윗팔이 자연스럽게 내려뜨려져 어깨·팔꿈치 관절 부하가 최적 범위입니다.',
                'bad':  '의자 높이를 조정하여 팔꿈치가 책상면과 수평이 되도록 하세요.',
                'na':   '팔꿈치 관절이 측면에서 가려져 측정할 수 없습니다.'},
    'knee':    {'range': '85 ~ 100°',
                'good': '무릎 내각이 VDT 고시 기준을 충족하며 하지 혈액순환이 원활합니다.',
                'bad':  '의자 높이를 조절하여 무릎 내각이 90° 전후가 되도록 하세요. 발 받침대 사용을 권장합니다.',
                'na':   '무릎 관절이 측면에서 가려져 측정할 수 없습니다.'},
    'wrist':   {'range': '±15° 이내',
                'good': '아래팔과 손이 중립 자세를 유지하여 손목건초염 위험이 최소화된 상태입니다.',
                'bad':  '손목 받침대를 키보드 앞에 설치하고 키보드와 책상 사이 15cm 공간을 확보하세요.',
                'na':   '손목 관절이 측면에서 가려져 측정할 수 없습니다.'},
    'gaze':    {'range': '하방 10 ~ 15°',
                'good': '모니터 시선각이 VDT 고시 기준에 적합하여 경추부 과부하가 없는 상태입니다.',
                'bad':  '모니터 상단이 눈높이와 일치하도록 높이를 조정하세요 (VDT 고시 제6조 1항).',
                'na':   '모니터가 탐지되지 않았습니다.'},
    'desk_h':  {'range': '팔꿈치 기준 ±10%',
                'good': '책상 높이가 팔꿈치와 수평 정렬되어 어깨 부하가 최소화된 상태입니다.',
                'bad':  '팔꿈치 높이에 맞춰 65cm 전후로 조정하거나 의자 높이로 보정하세요.',
                'na':   '책상이 탐지되지 않았습니다.'},
    'chair_d': {'range': '골반너비 20% 이내',
                'good': '등받이 지지가 충분하여 요추부터 어깨까지 편안하게 지지된 상태입니다.',
                'bad':  '의자 깊숙이 앉아 등 전체가 등받이에 닿도록 하세요 (VDT 고시 제6조 4항).',
                'na':   '의자가 탐지되지 않았습니다.'},
}

INDICATOR_NAMES = {
    'cva':     'CVA 목굴곡각',
    'tia':     'TIA 몸통굴곡각',
    'elbow':   '팔꿈치 각도',
    'knee':    '무릎 각도',
    'wrist':   '손목 편차',
    'gaze':    '모니터 시선각',
    'desk_h':  '작업대 높이',
    'chair_d': '의자-등받이',
}

IND_UNITS = {
    'cva':'°', 'tia':'°', 'elbow':'°',
    'knee':'°', 'wrist':'°', 'gaze':'°',
    'desk_h':'', 'chair_d':''
}


# =====================================================================
# 유틸
# =====================================================================
def get_vis(lm, idx):
    return lm[idx].visibility if hasattr(lm[idx], 'visibility') else 0.0

def is_vis(lm, idx):
    return get_vis(lm, idx) >= VISIBILITY_MIN

def best_idx(lm, left, right):
    """좌/우 중 visibility 높은 랜드마크 인덱스 반환"""
    return left if get_vis(lm, left) >= get_vis(lm, right) else right

def best_pair(lm, left1, right1, left2, right2):
    """두 쌍 중 visibility 합이 높은 쪽 (idx1, idx2) 반환"""
    l_score = get_vis(lm, left1)  + get_vis(lm, left2)
    r_score = get_vis(lm, right1) + get_vis(lm, right2)
    return (left1, left2) if l_score >= r_score else (right1, right2)

def best_triple(lm, l1, r1, l2, r2, l3, r3):
    """세 쌍 중 visibility 합이 높은 쪽 (i1,i2,i3) 반환"""
    l_score = get_vis(lm,l1) + get_vis(lm,l2) + get_vis(lm,l3)
    r_score = get_vis(lm,r1) + get_vis(lm,r2) + get_vis(lm,r3)
    return (l1,l2,l3) if l_score >= r_score else (r1,r2,r3)

def calc_angle_3pts(A, B, C):
    v1 = np.array(A) - np.array(B)
    v2 = np.array(C) - np.array(B)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

def calc_vertical_angle(p1, p2):
    vec = (p2[0]-p1[0], p2[1]-p1[1])
    return float(np.degrees(np.arctan2(abs(vec[0]), abs(vec[1]))))

def clip_val(v, lo, hi):
    return float(np.clip(v, lo, hi)) if v is not None else None

def to_px(lm, idx, w, h):
    return (int(lm[idx].x * w), int(lm[idx].y * h))

def to_norm(lm, idx):
    return (lm[idx].x, lm[idx].y)

def target_pt(pt, ref_pt, offset=OFFSET_PX):
    if ref_pt is None:
        return (pt[0], pt[1] - offset)
    dx, dy = ref_pt[0]-pt[0], ref_pt[1]-pt[1]
    dist   = max((dx**2+dy**2)**0.5, 1e-8)
    return (int(pt[0]+dx/dist*offset), int(pt[1]+dy/dist*offset))


# =====================================================================
# Step 1. MediaPipe 관절 추출
# =====================================================================
def step1_mediapipe(image_path):
    import mediapipe as mp
    img_cv  = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    h, w    = img_cv.shape[:2]
    try:
        import mediapipe.solutions.pose as _
        det     = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        results = det.process(img_rgb)
        lm      = results.pose_landmarks.landmark if results.pose_landmarks else None
    except Exception:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MP_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1, min_pose_detection_confidence=0.5
        )
        det    = vision.PoseLandmarker.create_from_options(opts)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res    = det.detect(mp_img)
        lm     = res.pose_landmarks[0] if res.pose_landmarks else None
    return lm, h, w


# =====================================================================
# Step 2. MLP 예측 + CVA·TIA 1차 판정
# predict.py(MLP)가 관절 추출 + CVA/TIA 계산을 포함하므로
# raw_landmarks를 Step 1 결과 대신 재활용
# =====================================================================
def step2_mlp_cva_tia(image_path):
    """
    새 predict.py (귀 기준 CVA) 연동
    - 모델: posture_mlp_final.keras
    - 스케일러: posture_scaler.pkl
    - MediaPipe Tasks API로 관절 추출
    """
    import joblib, cv2 as _cv2
    import mediapipe as _mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    import tensorflow as _tf
    import numpy as _np

    # ── 모델/스케일러 로드 ────────────────────────────────────────────
    if not os.path.exists(MLP_MODEL) or not os.path.exists(MLP_SCALER):
        err = {"error": f"MLP 모델/스케일러 없음: {MLP_MODEL}, {MLP_SCALER}"}
        return err, None, None, None, None

    try:
        model  = _tf.keras.models.load_model(MLP_MODEL)
        scaler = joblib.load(MLP_SCALER)
    except Exception as e:
        return {"error": f"모델 로드 실패: {e}"}, None, None, None, None

    # ── MediaPipe Tasks API로 관절 추출 ──────────────────────────────
    if not os.path.exists(MP_MODEL):
        return {"error": f"MediaPipe 모델 없음: {MP_MODEL}"}, None, None, None, None

    try:
        base_options = _mp_python.BaseOptions(model_asset_path=MP_MODEL)
        options      = _mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=_mp_vision.RunningMode.IMAGE
        )
        detector = _mp_vision.PoseLandmarker.create_from_options(options)
        mp_image = _mp.Image.create_from_file(image_path)
        result   = detector.detect(mp_image)
        detector.close()
    except Exception as e:
        return {"error": f"MediaPipe 실패: {e}"}, None, None, None, None

    if not result.pose_landmarks:
        return {"error": "관절 탐지 실패 — 측면 전신 이미지 권장"}, None, None, None, None

    # Tasks API landmark → lm 리스트 변환
    raw_lm = result.pose_landmarks[0]

    class _LM:
        def __init__(self, x, y, vis):
            self.x, self.y, self.visibility = x, y, vis

    lm = [_LM(raw_lm[i].x, raw_lm[i].y, raw_lm[i].visibility) for i in range(33)]

    frame = _cv2.imread(image_path)
    h, w  = frame.shape[:2]

    # ── 귀 기준 CVA / 어깨-골반 TIA 계산 ─────────────────────────────
    def vis(idx): return lm[idx].visibility
    def best(a, b): return a if vis(a) >= vis(b) else b

    ear_idx = best(7, 8)
    sh_idx  = best(11, 12)
    hp_idx  = best(23, 24)

    ear = (lm[ear_idx].x, lm[ear_idx].y)
    sh  = (lm[sh_idx].x,  lm[sh_idx].y)
    hip = (lm[hp_idx].x,  lm[hp_idx].y)

    def vertical_angle(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return float(_np.degrees(_np.arctan2(abs(dx), abs(dy))))

    cva = round(vertical_angle(ear, sh), 2)
    tia = round(vertical_angle(sh, hip), 2)

    # ── MLP 예측 ─────────────────────────────────────────────────────
    try:
        features    = _np.array([[cva, tia]])
        features_sc = scaler.transform(features)
        prob_good   = float(model.predict(features_sc, verbose=0)[0][0])
        prob_bad    = 1.0 - prob_good
        label       = "bad" if prob_bad >= 0.35 else "good"
        confidence  = round(prob_bad if label == "bad" else prob_good, 4)
    except Exception as e:
        return {"error": f"예측 실패: {e}"}, None, None, None, None

    mlp_res = {
        "label":      label,
        "confidence": confidence,
        "CVA":        cva,
        "TIA":        tia,
    }

    # ── all_ind 구성 ──────────────────────────────────────────────────
    mlp_good = (label == "good")
    sh_idx2  = best(11, 12)
    hp_idx2  = best(23, 24)

    all_ind = {
        'cva': {'value': cva, 'ok': mlp_good, 'joints': (ear_idx, sh_idx)},
        'tia': {'value': tia, 'ok': mlp_good, 'joints': (sh_idx2, hp_idx2)},
    }

    return mlp_res, all_ind, lm, h, w


# =====================================================================
# Step 3. YOLO 환경 탐지
# =====================================================================
def step3_yolo(image_path):
    from ultralytics import YOLO as YOLOModel
    bboxes = {'chair': None, 'desk': None, 'monitor': None}
    if not os.path.exists(YOLO_MODEL):
        return bboxes
    res   = YOLOModel(YOLO_MODEL).predict(source=image_path, conf=0.45, iou=0.3, verbose=False)
    boxes = res[0].boxes
    if boxes and len(boxes) > 0:
        for box in boxes:
            name = YOLO_CLASSES.get(int(box.cls[0]))
            if not name: continue
            conf = float(box.conf[0])
            if bboxes[name] is None or conf > bboxes[name]['conf']:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                bboxes[name] = {'x_min':x1,'y_min':y1,'x_max':x2,'y_max':y2,'conf':conf}
    return bboxes


# =====================================================================
# Step 4. 나머지 6개 지표 (RULA/VDT 기준)
# =====================================================================
def step4_remaining(lm, h, w, bboxes, img_w, img_h):
    # 상체 기준 단위 (visibility 높은 쪽 어깨-골반)
    sh_idx, hp_idx = best_pair(lm, 11, 12, 23, 24)
    ref_unit = abs(lm[sh_idx].y - lm[hp_idx].y) + 1e-8
    ind = {}

    # 팔꿈치: 어깨-팔꿈치-손목 (좌우 중 visibility 합 높은 쪽)
    try:
        i1,i2,i3 = best_triple(lm, 11,12, 13,14, 15,16)
        if all(is_vis(lm,i) for i in [i1,i2,i3]):
            v = clip_val(calc_angle_3pts(to_px(lm,i1,w,h), to_px(lm,i2,w,h), to_px(lm,i3,w,h)), 0, 180)
            lo,hi = RANGE_ELBOW
            ind['elbow'] = {'value': round(v,1), 'ok': lo<=v<=hi, 'joints': (i1,i2,i3)}
        else:
            ind['elbow'] = {'value': None, 'ok': None, 'joints': (12,14,16)}
    except Exception:
        ind['elbow'] = {'value': None, 'ok': None, 'joints': (12,14,16)}

    # 무릎: 골반-무릎-발목
    try:
        i1,i2,i3 = best_triple(lm, 23,24, 25,26, 27,28)
        if all(is_vis(lm,i) for i in [i1,i2,i3]):
            v = clip_val(calc_angle_3pts(to_px(lm,i1,w,h), to_px(lm,i2,w,h), to_px(lm,i3,w,h)), 0, 180)
            lo,hi = RANGE_KNEE
            ind['knee'] = {'value': round(v,1), 'ok': lo<=v<=hi, 'joints': (i1,i2,i3)}
        else:
            ind['knee'] = {'value': None, 'ok': None, 'joints': (24,26,28)}
    except Exception:
        ind['knee'] = {'value': None, 'ok': None, 'joints': (24,26,28)}

    # 손목: 팔꿈치-손목-손가락MCP
    try:
        i1,i2,i3 = best_triple(lm, 13,14, 15,16, 19,20)
        if all(is_vis(lm,i) for i in [i1,i2,i3]):
            inner = calc_angle_3pts(to_px(lm,i1,w,h), to_px(lm,i2,w,h), to_px(lm,i3,w,h))
            dev   = clip_val(abs(inner-180.0), 0, 90)
            ind['wrist'] = {'value': round(dev,1), 'ok': dev<=THRESHOLD_WRIST, 'joints': (i1,i2,i3)}
        else:
            ind['wrist'] = {'value': None, 'ok': None, 'joints': (14,16,20)}
    except Exception:
        ind['wrist'] = {'value': None, 'ok': None, 'joints': (14,16,20)}

    # 모니터 시선각
    try:
        mon      = bboxes.get('monitor')
        eye_l, eye_r = 1, 4
        eye_idx  = best_idx(lm, eye_l, eye_r)
        if mon and is_vis(lm, eye_idx):
            ex = lm[eye_idx].x * img_w
            ey = lm[eye_idx].y * img_h
            mx = (mon['x_min']+mon['x_max'])/2
            my = (mon['y_min']+mon['y_max'])/2
            gaze = clip_val(float(np.degrees(np.arctan2(my-ey, abs(mx-ex)))), -30, 60)
            lo,hi = RANGE_GAZE
            ind['gaze'] = {'value': round(gaze,1), 'ok': lo<=gaze<=hi, 'joints': (eye_idx,)}
        else:
            ind['gaze'] = {'value': None, 'ok': None, 'joints': (eye_idx,)}
    except Exception:
        ind['gaze'] = {'value': None, 'ok': None, 'joints': (1,)}

    # 작업대 높이
    try:
        desk     = bboxes.get('desk')
        el_idx   = best_idx(lm, 13, 14)
        if desk and is_vis(lm, el_idx):
            diff = abs(desk['y_min']/img_h - lm[el_idx].y) / ref_unit
            ind['desk_h'] = {'value': round(diff,3), 'ok': diff<=THRESHOLD_DESK, 'joints': (el_idx,)}
        else:
            ind['desk_h'] = {'value': None, 'ok': None, 'joints': (el_idx,)}
    except Exception:
        ind['desk_h'] = {'value': None, 'ok': None, 'joints': (14,)}

    # 의자-등받이 거리
    try:
        chair   = bboxes.get('chair')
        hp_idx2 = best_idx(lm, 23, 24)
        if chair and is_vis(lm, hp_idx2):
            hip_w = max(abs(lm[23].x-lm[24].x), abs(lm[11].x-lm[12].x), 0.05)
            gap   = abs(lm[hp_idx2].x - chair['x_max']/img_w) / hip_w
            ind['chair_d'] = {'value': round(gap,3), 'ok': gap<=THRESHOLD_CHAIR, 'joints': (hp_idx2,)}
        else:
            ind['chair_d'] = {'value': None, 'ok': None, 'joints': (hp_idx2,)}
    except Exception:
        ind['chair_d'] = {'value': None, 'ok': None, 'joints': (24,)}

    return ind


# =====================================================================
# Step 5. 오버레이 시각화
# =====================================================================
def step5_overlay(image_path, lm, h, w, all_ind, early_stop, bboxes=None):
    """
    오버레이 규칙:
    - CVA/TIA 하나라도 BAD  → 현재 척추선(빨강) + 목표 척추선(민트) + 화살표만
    - CVA/TIA 모두 GOOD     → 현재 척추선(초록) + 6개 지표 현재 관절선
                               (BAD 항목은 목표 위치 점선+링 추가, 측정불가 생략)
    """
    import math
    img = cv2.imread(image_path)
    ih, iw = img.shape[:2]

    def pt(idx): return to_px(lm, idx, w, h)

    # ── 인덱스 ───────────────────────────────────────────────────────
    sh_idx, hp_idx = best_pair(lm, 11, 12, 23, 24)
    nose_idx = 0                     # 코 (촬영방향 감지용)
    ear_idx  = best_idx(lm, 7, 8)   # 귀 (CVA 기준: 자세기준서)
    el_idx   = best_idx(lm, 13, 14)
    wr_idx   = best_idx(lm, 15, 16)
    fi_idx   = best_idx(lm, 17, 18)   # 소지MCP
    kn_idx   = best_idx(lm, 25, 26)
    an_idx   = best_idx(lm, 27, 28)
    eye_idx  = best_idx(lm, 1, 4)

    sh_mid = (int((pt(11)[0]+pt(12)[0])/2), int((pt(11)[1]+pt(12)[1])/2))

    # 촬영 방향: 코 x > 무릎 x → 오른쪽이 앞
    forward = 1 if pt(nose_idx)[0] > pt(kn_idx)[0] else -1

    # ── 드로잉 헬퍼 ──────────────────────────────────────────────────
    def line(p1, p2, color, t=2):
        """실선 (흰 외곽 + 컬러)"""
        cv2.line(img, p1, p2, (255,255,255), t+2, cv2.LINE_AA)
        cv2.line(img, p1, p2, color, t, cv2.LINE_AA)

    def dot(p, color, r=8):
        """채워진 원 — 현재 관절"""
        cv2.circle(img, p, r,   color,        -1)
        cv2.circle(img, p, r+2, (255,255,255), 2)

    def goal_ring(p, r=10):
        """속 빈 링 — 목표 위치 (하늘색)"""
        GOAL = (200, 220, 255)
        cv2.circle(img, p, r,   (255,255,255), -1)
        cv2.circle(img, p, r,   GOAL,           3)
        cv2.circle(img, p, r+3, (50, 50, 50),   1)

    def dashed(p1, p2):
        """점선 — 현재→목표 연결 (하늘색)"""
        GOAL = (200, 220, 255)
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if dist < 2: return
        steps = max(int(dist/10), 1)
        for i in range(steps):
            if i % 2 == 0:
                t1=i/steps; t2=min((i+0.9)/steps, 1.0)
                x1=int(p1[0]+(p2[0]-p1[0])*t1); y1=int(p1[1]+(p2[1]-p1[1])*t1)
                x2=int(p1[0]+(p2[0]-p1[0])*t2); y2=int(p1[1]+(p2[1]-p1[1])*t2)
                cv2.line(img,(x1,y1),(x2,y2),(255,255,255),2,cv2.LINE_AA)
                cv2.line(img,(x1,y1),(x2,y2),GOAL,1,cv2.LINE_AA)

    def badge(p, num, color):
        """번호 뱃지"""
        bx, by = p[0]-14, p[1]-14
        cv2.circle(img,(bx,by), 9, (255,255,255), -1)
        cv2.circle(img,(bx,by), 10, color, 2)
        cv2.putText(img, str(num),
                    (bx-4 if num<10 else bx-6, by+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20,20,20), 1, cv2.LINE_AA)

    def arrow(src, dst, color):
        """화살표"""
        cv2.arrowedLine(img,src,dst,(255,255,255),4,tipLength=0.2,line_type=cv2.LINE_AA)
        cv2.arrowedLine(img,src,dst,color,2,tipLength=0.2,line_type=cv2.LINE_AA)

    def spine(pts, color, t=3):
        """척추선: 실선 + 중간점 + 끝점"""
        for i in range(len(pts)-1):
            cv2.line(img,pts[i],pts[i+1],(255,255,255),t+2,cv2.LINE_AA)
            cv2.line(img,pts[i],pts[i+1],color,t,cv2.LINE_AA)
        for i in range(len(pts)-1):
            p1,p2=pts[i],pts[i+1]
            for tf in [0.25,0.5,0.75]:
                mx=int(p1[0]+(p2[0]-p1[0])*tf); my=int(p1[1]+(p2[1]-p1[1])*tf)
                cv2.circle(img,(mx,my),4,color,-1)
                cv2.circle(img,(mx,my),6,(255,255,255),1)
        for p in pts:
            dot(p, color, 7)

    def inner_angle(a, b, c):
        """b를 꼭짓점으로 한 내각 (도)"""
        v1=(a[0]-b[0],a[1]-b[1]); v2=(c[0]-b[0],c[1]-b[1])
        cos_a=(v1[0]*v2[0]+v1[1]*v2[1])/(math.hypot(*v1)*math.hypot(*v2)+1e-8)
        return math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))

    def goal_end_pt(anchor, end_pt, cur_ang, goal_ang):
        """
        anchor 고정, end_pt를 goal_ang 내각 위치로 회전 이동
        이미지 좌표계(y↓) + 촬영 방향(forward) 고려
        """
        dx=end_pt[0]-anchor[0]; dy=end_pt[1]-anchor[1]
        dist=math.hypot(dx,dy)
        if dist < 1: return end_pt
        delta = math.radians(goal_ang - cur_ang) * forward * -1
        c,s = math.cos(delta), math.sin(delta)
        ndx=dx*c-dy*s; ndy=dx*s+dy*c
        return (int(anchor[0]+ndx), int(anchor[1]+ndy))

    # ── CVA/TIA 상태 확인 ────────────────────────────────────────────
    cva_ok    = all_ind.get('cva', {}).get('ok')
    tia_ok    = all_ind.get('tia', {}).get('ok')
    angle_bad = (cva_ok is False) or (tia_ok is False)

    hp_pt  = pt(hp_idx)
    sh_pt  = sh_mid
    ear_pt = pt(ear_idx)   # 귀: CVA 기준점 (자세기준서)

    # ── 현재 척추선: 골반→어깨→귀 (CVA/TIA BAD=빨강, GOOD=초록) ────
    cur_color = COLOR_BAD if angle_bad else COLOR_GOOD
    spine([hp_pt, sh_pt, ear_pt], cur_color)

    # ── CVA/TIA BAD: 목표 척추선 + 화살표 후 종료 ────────────────────
    if angle_bad:
        sh_hp_d  = math.hypot(sh_pt[0]-hp_pt[0], sh_pt[1]-hp_pt[1])
        ear_sh_d = math.hypot(ear_pt[0]-sh_pt[0], ear_pt[1]-sh_pt[1])

        # TIA 목표: Good 중앙값 5° (어깨가 골반보다 forward 방향으로 5°)
        sh_tgt = sh_pt
        if tia_ok is False:
            dx = int(forward * sh_hp_d * math.sin(math.radians(5)))
            dy = -int(sh_hp_d * math.cos(math.radians(5)))
            sh_tgt = (hp_pt[0]+dx, hp_pt[1]+dy)

        # CVA 목표: 귀-어깨 각도 Good 중앙값 10° (귀가 어깨보다 forward 방향으로 10°)
        ear_tgt = ear_pt
        if cva_ok is False:
            dx = int(forward * ear_sh_d * math.sin(math.radians(10)))
            dy = -int(ear_sh_d * math.cos(math.radians(10)))
            ear_tgt = (sh_tgt[0]+dx, sh_tgt[1]+dy)

        spine([hp_pt, sh_tgt, ear_tgt], COLOR_TARGET)

        if tia_ok is False and sh_tgt != sh_pt:
            arrow(sh_pt,  sh_tgt,  COLOR_TARGET)
        if cva_ok is False and ear_tgt != ear_pt:
            arrow(ear_pt, ear_tgt, COLOR_TARGET)

        return img   # ← 6개 지표 표시 안 함

    # ── CVA/TIA 모두 GOOD: 6개 지표 표시 ─────────────────────────────
    KEY_ORDER = ['cva','tia','elbow','knee','wrist','gaze','desk_h','chair_d']

    for key in ['elbow','knee','wrist','gaze','desk_h','chair_d']:
        info = all_ind.get(key)
        if not info: continue
        ok = info.get('ok')
        if ok is None: continue   # 측정불가 → 자동 제외

        num   = KEY_ORDER.index(key) + 1
        color = COLOR_GOOD if ok else COLOR_BAD

        # ── 3. 팔꿈치: 어깨-팔꿈치-손목 내각 90~120° (목표 105°) ────
        if key == 'elbow':
            i1,i2,i3 = info.get('joints', (sh_idx,el_idx,wr_idx))
            # 현재 관절선
            for a,b in [(i1,i2),(i2,i3)]:
                if is_vis(lm,a) and is_vis(lm,b): line(pt(a),pt(b),color)
            for i in [i1,i2,i3]:
                if is_vis(lm,i): dot(pt(i),color)
            main_pt = pt(i2)
            # BAD: 목표 위치 (팔꿈치 고정, 손목을 105° 위치로)
            if not ok and all(is_vis(lm,i) for i in [i1,i2,i3]):
                cur    = inner_angle(pt(i1),pt(i2),pt(i3))
                wr_tgt = goal_end_pt(pt(i2),pt(i3),cur,105.0)
                line(pt(i1),pt(i2),(200,220,255),1)  # 어깨→팔꿈치 목표선
                dashed(pt(i2),wr_tgt)                # 팔꿈치→손목 점선
                goal_ring(pt(i2),7)                  # 팔꿈치 기준 링
                goal_ring(wr_tgt)                    # 손목 목표 링

        # ── 4. 무릎: 골반-무릎-발목 내각 85~100° (목표 92.5°) ────────
        elif key == 'knee':
            i1,i2,i3 = info.get('joints', (hp_idx,kn_idx,an_idx))
            for a,b in [(i1,i2),(i2,i3)]:
                if is_vis(lm,a) and is_vis(lm,b): line(pt(a),pt(b),color)
            for i in [i1,i2,i3]:
                if is_vis(lm,i): dot(pt(i),color)
            main_pt = pt(i2)
            if not ok and all(is_vis(lm,i) for i in [i1,i2,i3]):
                cur    = inner_angle(pt(i1),pt(i2),pt(i3))
                an_tgt = goal_end_pt(pt(i2),pt(i3),cur,92.5)
                line(pt(i1),pt(i2),(200,220,255),1)
                dashed(pt(i2),an_tgt)
                goal_ring(pt(i2),7)
                goal_ring(an_tgt)

        # ── 5. 손목: 편차 ≤15° (목표=일직선 0°) ─────────────────────
        # 팔꿈치→손목 방향 연장선 위에 손가락 목표 위치
        elif key == 'wrist':
            i1,i2,i3 = info.get('joints', (el_idx,wr_idx,fi_idx))
            for a,b in [(i1,i2),(i2,i3)]:
                if is_vis(lm,a) and is_vis(lm,b): line(pt(a),pt(b),color)
            for i in [i1,i2,i3]:
                if is_vis(lm,i): dot(pt(i),color)
            main_pt = pt(i2)
            if not ok and all(is_vis(lm,i) for i in [i1,i2,i3]):
                dx=pt(i2)[0]-pt(i1)[0]; dy=pt(i2)[1]-pt(i1)[1]
                d=math.hypot(dx,dy)+1e-8
                fi_d=math.hypot(pt(i3)[0]-pt(i2)[0],pt(i3)[1]-pt(i2)[1])
                fi_tgt=(int(pt(i2)[0]+dx/d*fi_d),int(pt(i2)[1]+dy/d*fi_d))
                dashed(pt(i3),fi_tgt)   # 현재 손가락 → 목표 손가락
                goal_ring(pt(i2),7)     # 손목 기준 링
                goal_ring(fi_tgt)       # 손가락 목표 링

        # ── 6. 시선각: 하방 10~15° (목표 12.5°) ─────────────────────
        elif key == 'gaze':
            e = info.get('joints', (eye_idx,))[0]
            if not is_vis(lm,e): continue
            main_pt = pt(e)
            dot(main_pt, color)
            if not ok:
                g   = math.radians(12.5)
                tgt = (main_pt[0]+int(forward*80*math.cos(g)),
                       main_pt[1]+int(80*math.sin(g)))
                dashed(main_pt, tgt)
                goal_ring(tgt)

        # ── 7. 작업대 높이: 팔꿈치 y = 책상 y (±10%) ────────────────
        elif key == 'desk_h':
            e2 = info.get('joints', (el_idx,))[0]
            if not is_vis(lm,e2): continue
            main_pt = pt(e2)
            dot(main_pt, color)
            if not ok:
                if bboxes and bboxes.get('desk'):
                    desk_y = int(bboxes['desk']['y_min'])
                    tgt    = (main_pt[0], desk_y)
                else:
                    tgt = (main_pt[0], main_pt[1]-40)
                dashed(main_pt, tgt)
                goal_ring(tgt)

        # ── 8. 등받이: 골반→등받이 방향 중간값 ──────────────────────
        elif key == 'chair_d':
            h2 = info.get('joints', (hp_idx,))[0]
            if not is_vis(lm,h2): continue
            main_pt = pt(h2)
            dot(main_pt, color)
            if not ok:
                if bboxes and bboxes.get('chair'):
                    kn_x_pos = pt(kn_idx)[0]
                    hp_x_pos = main_pt[0]
                    chair_x  = int(bboxes['chair']['x_min']) if kn_x_pos > hp_x_pos                                else int(bboxes['chair']['x_max'])
                    tgt = (int((main_pt[0]+chair_x)/2), main_pt[1])
                else:
                    back = -1 if pt(kn_idx)[0] > main_pt[0] else 1
                    tgt  = (main_pt[0]+back*50, main_pt[1])
                dashed(main_pt, tgt)
                goal_ring(tgt)
        else:
            continue

        badge(main_pt, num, color)

    return img


def run_pipeline(image_path):
    img_cv       = cv2.imread(image_path)
    img_h, img_w = img_cv.shape[:2]

    # Step 1 + 2: MLP가 내부적으로 MediaPipe 포함 → 함께 처리
    mlp_res, all_ind, lm, h, w = step2_mlp_cva_tia(image_path)

    if mlp_res is None or 'error' in (mlp_res or {}):
        err = (mlp_res or {}).get('error', '관절 탐지 실패')
        return None, None, None, err

    cva_ok     = all_ind.get('cva', {}).get('ok')
    tia_ok     = all_ind.get('tia', {}).get('ok')
    early_stop = (cva_ok is False) or (tia_ok is False)

    bboxes = None
    if not early_stop:
        # Step 3
        bboxes = step3_yolo(image_path)
        # Step 4
        rest   = step4_remaining(lm, h, w, bboxes, img_w, img_h)
        all_ind.update(rest)

    # Step 5
    overlay = step5_overlay(image_path, lm, h, w, all_ind, early_stop,
                            bboxes=bboxes if not early_stop else None)

    return mlp_res, all_ind, overlay, early_stop


# =====================================================================
# Tkinter UI
# =====================================================================
class IntegrateApp:
    def __init__(self, root):
        self.root       = root
        self.root.title("Fit me up | 자세&환경 통합 분석")
        self.root.geometry("1200x820")
        self.root.configure(bg="#f4f4f4")
        self.image_path = None
        self.tk_img     = None
        self._setup_ui()

    def _setup_ui(self):
        tk.Label(self.root, text="🧘 Fit me up  |  자세&환경 통합 분석",
                 font=("Malgun Gothic", 19, "bold"),
                 bg="#1e1e2e", fg="white", pady=10
        ).pack(fill=tk.X)

        # ── 상단: 이미지 + MLP 판정 ───────────────────────────────────
        top = tk.Frame(self.root, bg="#f4f4f4")
        top.pack(pady=10, padx=18, fill=tk.X)

        # 이미지 캔버스
        left = tk.Frame(top, bg="#f4f4f4")
        left.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(left, width=480, height=480,
                                bg="white", highlightthickness=1,
                                highlightbackground="#ccc")
        self.canvas.pack()

        btn_row = tk.Frame(left, bg="#f4f4f4")
        btn_row.pack(pady=8)
        tk.Button(btn_row, text="📁 이미지 선택", command=self.select_image,
                  width=15, bg="#3498db", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="🔍 분석 시작",  command=self.run_analysis,
                  width=15, bg="#2ecc71", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="🔄 초기화",     command=self.reset,
                  width=10, bg="#95a5a6", fg="white", relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        # ── MLP 판정 패널 (이미지 오른쪽) ──────────────────────────
        right_area = tk.Frame(top, bg="#f4f4f4")
        right_area.pack(side=tk.LEFT, padx=20, anchor="n", fill=tk.Y)

        # 판정 박스: GOOD/BAD 크게 + 안내문 같은 프레임 안에
        self.res_panel = tk.Frame(right_area, width=320,
                                  bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        self.res_panel.pack(pady=(10,8), fill=tk.X)

        self.lbl_result = tk.Label(self.res_panel, text="READY",
                                   font=("Arial", 42, "bold"),
                                   bg="#ecf0f1", fg="#7f8c8d", pady=12)
        self.lbl_result.pack()

        self.lbl_guide = tk.Label(self.res_panel, text="",
                                  font=("Malgun Gothic", 10),
                                  bg="#ecf0f1", fg="#555",
                                  wraplength=290, justify=tk.CENTER, pady=8)
        self.lbl_guide.pack()

        # MLP 신뢰도
        self.lbl_conf = tk.Label(right_area, text="",
                                 font=("Malgun Gothic", 9), bg="#f4f4f4", fg="#888")
        self.lbl_conf.pack(anchor="w", pady=(0,6))

        # GOOD / BAD 2열 체크리스트 헤더
        chk_header = tk.Frame(right_area, bg="#f4f4f4")
        chk_header.pack(fill=tk.X, pady=(6,2))
        tk.Label(chk_header, text="✅ GOOD", font=("Malgun Gothic", 10, "bold"),
                 bg="#f4f4f4", fg="#1D9E75", width=18, anchor="w").pack(side=tk.LEFT)
        tk.Label(chk_header, text="❌ BAD", font=("Malgun Gothic", 10, "bold"),
                 bg="#f4f4f4", fg="#E24B4A", width=18, anchor="w").pack(side=tk.LEFT)
        tk.Frame(right_area, bg="#ccc", height=1).pack(fill=tk.X, pady=2)

        self.chk_frame = tk.Frame(right_area, bg="#f4f4f4")
        self.chk_frame.pack(fill=tk.X)

        self.ind_labels = {}

    def _build_feedback_cards(self, all_ind, early_stop):
        for w in self.fb_frame.winfo_children():
            w.destroy()

        keys = ['cva','tia'] if early_stop else list(INDICATOR_NAMES.keys())

        # 2열 그리드 컨테이너
        grid = tk.Frame(self.fb_frame, bg="#f4f4f4")
        grid.pack(fill=tk.BOTH, expand=True)
        for c in range(4):
            grid.columnconfigure(c, weight=1)

        for idx, key in enumerate(keys):
            r    = all_ind.get(key, {})
            ok   = r.get('ok')
            v    = r.get('value')
            fb   = FEEDBACK.get(key, {})
            name = INDICATOR_NAMES.get(key, key)
            unit = IND_UNITS.get(key, '')

            if ok is None:
                status, border, badge_bg, badge_fg, msg = \
                    "측정불가", "#aaa", "#eee", "#555", fb.get('na','—')
                val_color = "#aaa"
            elif ok:
                status, border, badge_bg, badge_fg, msg = \
                    "GOOD", "#1D9E75", "#d4f7e7", "#0a5e3a", fb.get('good','')
                val_color = "#1D9E75"
            else:
                status, border, badge_bg, badge_fg, msg = \
                    "BAD", "#E24B4A", "#fde8e8", "#7a1010", fb.get('bad','')
                val_color = "#E24B4A"

            col = idx % 4
            row = idx // 4

            card = tk.Frame(grid, bg="white",
                            highlightbackground=border,
                            highlightthickness=2,
                            relief=tk.FLAT)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")

            # 상단 컬러 바
            tk.Frame(card, bg=border, height=4).pack(fill=tk.X)

            body = tk.Frame(card, bg="white", padx=8, pady=7)
            body.pack(fill=tk.BOTH, expand=True)

            # 이름 + 뱃지
            hdr = tk.Frame(body, bg="white")
            hdr.pack(fill=tk.X)
            tk.Label(hdr, text=f"{idx+1}. {name}", font=("Malgun Gothic", 9, "bold"),
                     bg="white", fg="#222").pack(side=tk.LEFT)
            tk.Label(hdr, text=f" {status} ",
                     font=("Malgun Gothic", 7, "bold"),
                     bg=badge_bg, fg=badge_fg,
                     relief=tk.FLAT, padx=3).pack(side=tk.RIGHT)

            # 측정값
            val_str = f"{v}{unit}" if v is not None else "—"
            tk.Label(body, text=val_str,
                     font=("Arial", 15, "bold"),
                     bg="white", fg=val_color, anchor="w").pack(fill=tk.X, pady=(3,0))

            # 정상범위
            tk.Label(body, text=f"정상: {fb.get('range','—')}",
                     font=("Malgun Gothic", 7),
                     bg="white", fg="#888", anchor="w").pack(fill=tk.X)

            # 피드백 메시지
            tk.Label(body, text=msg,
                     font=("Malgun Gothic", 7), bg="white",
                     fg="#444", wraplength=220,
                     justify=tk.LEFT, anchor="w").pack(fill=tk.X, pady=(3,0))

    # ── GOOD/BAD 체크리스트 ──────────────────────────────────────────
    def _build_checklist(self, all_ind):
        for w in self.chk_frame.winfo_children():
            w.destroy()

        NAMES = {
            'cva':     'CVA 목굴곡각',
            'tia':     'TIA 몸통굴곡각',
            'elbow':   '팔꿈치 각도',
            'knee':    '무릎 각도',
            'wrist':   '손목 편차',
            'gaze':    '모니터 시선각',
            'desk_h':  '작업대 높이',
            'chair_d': '의자 등받이',
        }

        good_items = []  # (번호, 텍스트, ok)
        bad_items  = []

        KEY_ORDER_CHK = ['cva','tia','elbow','knee','wrist','gaze','desk_h','chair_d']
        for key, name in NAMES.items():
            num  = KEY_ORDER_CHK.index(key) + 1
            r    = all_ind.get(key)
            ok   = r.get('ok') if r else None
            v    = r.get('value') if r else None
            unit = IND_UNITS.get(key, '')
            val_str = f"{v}{unit}" if v is not None else "측정불가"

            if ok is True:
                good_items.append((num, f"✓ {num}. {name}  {val_str}", True))
            elif ok is False:
                bad_items.append((num, f"✗ {num}. {name}  {val_str}", False))
            else:
                bad_items.append((num, f"— {num}. {name}  측정불가", None))

        max_rows = max(len(good_items), len(bad_items), 1)

        for i in range(max_rows):
            row = tk.Frame(self.chk_frame, bg="#f4f4f4")
            row.pack(fill=tk.X, pady=1)

            if i < len(good_items):
                _, gtxt, _ = good_items[i]
                tk.Label(row, text=gtxt,
                         font=("Malgun Gothic", 9), bg="#f4f4f4",
                         fg="#1D9E75", width=22, anchor="w").pack(side=tk.LEFT)
            else:
                tk.Label(row, text="", width=22, bg="#f4f4f4").pack(side=tk.LEFT)

            if i < len(bad_items):
                _, btxt, bok = bad_items[i]
                fg = "#E24B4A" if bok is False else "#aaaaaa"  # 측정불가=연한색
                tk.Label(row, text=btxt,
                         font=("Malgun Gothic", 9), bg="#f4f4f4",
                         fg=fg, width=22, anchor="w").pack(side=tk.LEFT)
            else:
                tk.Label(row, text="", width=22, bg="#f4f4f4").pack(side=tk.LEFT)

    # ── 이미지 선택 ───────────────────────────────────────────────────
    def select_image(self):
        path = filedialog.askopenfilename(
            title="분석할 이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path: return
        self.image_path = path
        self._show_pil(Image.open(path))
        self.reset_labels()

    # ── 분석 실행 ─────────────────────────────────────────────────────
    def run_analysis(self):
        if not self.image_path:
            messagebox.showwarning("경고", "이미지를 먼저 선택하세요.")
            return

        self.lbl_result.config(text="분석 중...", bg="#e67e22", fg="white")
        self.res_panel.config(bg="#e67e22")
        self.root.update()

        mlp_res, all_ind, overlay, early_stop = run_pipeline(self.image_path)

        if isinstance(early_stop, str):
            messagebox.showerror("Error", early_stop)
            self.reset_labels()
            return

        # 최종 판정: CVA/TIA 기반 (둘 다 Good이면 GOOD)
        cva_ok_ui = all_ind.get('cva', {}).get('ok')
        tia_ok_ui = all_ind.get('tia', {}).get('ok')
        posture_good = (cva_ok_ui is not False) and (tia_ok_ui is not False)

        # 나머지 6개 지표 중 하나라도 Bad인지
        env_bad = any(
            all_ind.get(k, {}).get('ok') is False
            for k in all_ind if k not in ('cva', 'tia')
        )

        conf  = mlp_res.get('confidence', 0) * 100 if mlp_res else 0

        if not posture_good:
            # 자세 BAD
            color = "#e74c3c"
            self.res_panel.config(bg=color)
            self.lbl_result.config(
                text="BAD", bg=color, fg="white",
                font=("Arial", 42, "bold"))
            self.lbl_guide.config(
                text="지금 자세는 BAD입니다.\nGOOD이 될 때까지 자세를 교정하고\n다시 촬영해주세요.",
                bg=color, fg="white", font=("Malgun Gothic", 10))
        elif env_bad:
            # 자세 Good, 환경 일부 Bad
            color = "#e67e22"
            self.res_panel.config(bg=color)
            self.lbl_result.config(
                text="GOOD", bg=color, fg="white",
                font=("Arial", 42, "bold"))
            self.lbl_guide.config(
                text="자세는 좋아요! 🎉\n작업 환경을 조금 더 조정해보세요.",
                bg=color, fg="white", font=("Malgun Gothic", 10))
        else:
            # 자세 + 환경 모두 Good
            color = "#27ae60"
            self.res_panel.config(bg=color)
            self.lbl_result.config(
                text="GOOD", bg=color, fg="white",
                font=("Arial", 42, "bold"))
            self.lbl_guide.config(
                text="자세와 환경 모두 완벽해요! 👍\n지금 상태를 유지해주세요.",
                bg=color, fg="white", font=("Malgun Gothic", 10))

        self.lbl_conf.config(text=f"MLP 신뢰도: {conf:.1f}%")

        # GOOD/BAD 체크리스트
        self._build_checklist(all_ind)

        # 오버레이 이미지
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        self._show_pil(Image.fromarray(overlay_rgb))

    # ── 초기화 ────────────────────────────────────────────────────────
    def reset(self):
        self.image_path = None
        self.canvas.delete("all")
        self.reset_labels()

    def reset_labels(self):
        self.res_panel.config(bg="#ecf0f1")
        self.lbl_result.config(text="READY", bg="#ecf0f1",
                               fg="#7f8c8d", font=("Arial", 42, "bold"))
        self.lbl_guide.config(text="", bg="#ecf0f1", fg="#555")
        self.lbl_conf.config(text="")
        for w in self.chk_frame.winfo_children():
            w.destroy()

    def _show_pil(self, pil_img):
        pil_img.thumbnail((530, 530))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(265, 265, image=self.tk_img)


# =====================================================================
# 실행
# =====================================================================
if __name__ == '__main__':
    root = tk.Tk()
    IntegrateApp(root)
    root.mainloop()
