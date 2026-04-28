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
MP_MODEL   = os.path.join(BASE, 'MediaPipe', 'models', 'pose_landmarker.task')
YOLO_MODEL = os.path.join(BASE, 'YOLO', 'fit_me_up', 'combined_gpu', 'weights', 'best.pt')

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
    sys.path.insert(0, os.path.join(BASE, 'MediaPipe', 'code'))
    from predict import predict_posture

    res = predict_posture(image_path)

    if res is None or 'error' in res:
        return res, None, None, None, None

    # raw_landmarks → integrate.py 형식으로 변환
    # predict.py: {idx: {'x','y','vis'}}
    # integrate.py: lm 리스트 (lm[idx].x, lm[idx].y, lm[idx].visibility)
    raw = res.get('raw_landmarks', {})

    class _LM:
        def __init__(self, x, y, vis):
            self.x, self.y, self.visibility = x, y, vis

    lm = [_LM(raw[i]['x'], raw[i]['y'], raw[i]['vis']) for i in range(33)]
    h  = res.get('img_h', 1)
    w  = res.get('img_w', 1)

    cva = res.get('CVA')
    tia = res.get('TIA')

    # MLP label 기반 판정 (predict.py가 학습된 모델로 종합 판단)
    mlp_label = res.get('label', 'bad')   # "good" or "bad"
    mlp_good  = (mlp_label == 'good')

    all_ind = {}

    # CVA/TIA ok 여부 → MLP 판정 기준
    # 둘 다 Good이면 모델이 Good → 둘 다 ok=True
    # Bad이면 각 각도 임계값으로 어느 쪽이 문제인지 구분
    cva_bad_thresh = cva is not None and cva > THRESHOLD_CVA
    tia_bad_thresh = tia is not None and tia > THRESHOLD_TIA

    if mlp_good:
        # MLP가 Good → CVA/TIA 모두 Good
        cva_ok = True
        tia_ok = True
    else:
        # MLP가 Bad → 각도 임계값으로 어느 쪽이 Bad인지 구분
        cva_ok = not cva_bad_thresh
        tia_ok = not tia_bad_thresh
        # 둘 다 임계값 안이지만 MLP가 Bad인 경우 → 둘 다 Bad 처리
        if cva_ok and tia_ok:
            cva_ok = False
            tia_ok = False

    sh_idx_cva = best_idx(lm, 11, 12)
    sh_idx2, hp_idx = best_pair(lm, 11, 12, 23, 24)

    all_ind['cva'] = {
        'value':  round(cva, 1) if cva is not None else None,
        'ok':     cva_ok if cva is not None else None,
        'joints': (0, sh_idx_cva)
    }
    all_ind['tia'] = {
        'value':  round(tia, 1) if tia is not None else None,
        'ok':     tia_ok if tia is not None else None,
        'joints': (sh_idx2, hp_idx)
    }

    return res, all_ind, lm, h, w


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
def _draw_joint(img, pt_px, ok, num=None):
    """관절 점만 (번호 없음, 크기 2/3)"""
    color = COLOR_GOOD if ok else COLOR_BAD
    cv2.circle(img, pt_px, 7, color,        -1)
    cv2.circle(img, pt_px, 9, (255,255,255), 2)


def _draw_bad_arrow(img, joint_pt, ref_pt):
    """Bad 관절: 목표 초록 점 + 굵은 화살표"""
    tgt = target_pt(joint_pt, ref_pt)
    # 화살표 굵게
    cv2.arrowedLine(img, joint_pt, tgt, (255,255,255), 4, tipLength=0.3, line_type=cv2.LINE_AA)
    cv2.arrowedLine(img, joint_pt, tgt, COLOR_TARGET,  2, tipLength=0.3, line_type=cv2.LINE_AA)
    # 목표 초록 점 크게
    cv2.circle(img, tgt, 9,  COLOR_TARGET,   -1)
    cv2.circle(img, tgt, 11, (255,255,255),   2)


def step5_overlay(image_path, lm, h, w, all_ind, early_stop, bboxes=None):
    img = cv2.imread(image_path)

    def pt(idx): return to_px(lm, idx, w, h)

    # ── visibility 높은 쪽 선택 ──────────────────────────────────────
    sh_idx, hp_idx = best_pair(lm, 11, 12, 23, 24)
    sh2_idx        = 11 if sh_idx == 12 else 12   # 반대쪽 어깨
    hp2_idx        = 23 if hp_idx == 24 else 24   # 반대쪽 골반
    ear_idx        = best_idx(lm, 7, 8)   # 뼈대용
    nose_idx       = 0                     # CVA 기준: 코(DA팀 기준)
    el_idx         = best_idx(lm, 13, 14)
    wr_idx         = best_idx(lm, 15, 16)
    kn_idx         = best_idx(lm, 25, 26)
    an_idx         = best_idx(lm, 27, 28)

    # ── 뼈대 연결선 ───────────────────────────────────────────────────
    bones = [
        (nose_idx, sh_idx),
        (sh_idx,  hp_idx),
        (sh_idx,  el_idx),
        (el_idx,  wr_idx),
        (hp_idx,  kn_idx),
        (kn_idx,  an_idx),
    ]
    for a, b in bones:
        if is_vis(lm, a) and is_vis(lm, b):
            cv2.line(img, pt(a), pt(b), COLOR_BONE, 2, cv2.LINE_AA)

    # ── YOLO 바운딩박스 ───────────────────────────────────────────────
    YOLO_COLORS = {
        'chair':   (255, 150,  40),   # 주황
        'desk':    ( 40, 205, 255),   # 하늘
        'monitor': (185,  70, 255),   # 보라
    }
    if bboxes:
        for name, b in bboxes.items():
            if b is None:
                continue
            x1,y1,x2,y2 = int(b['x_min']),int(b['y_min']),int(b['x_max']),int(b['y_max'])
            color = YOLO_COLORS.get(name, (200,200,200))
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label_bg_x2 = min(x1 + len(name)*13 + 16, x2)
            cv2.rectangle(img, (x1, max(0,y1-26)), (label_bg_x2, y1), color, -1)
            cv2.putText(img, name, (x1+5, y1-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)

    # ── 지표별 관절 포인트 정의 ───────────────────────────────────────
    # CVA: 귀 + 어깨 (2개 점 + 연결선)
    # TIA: 어깨 중점 + 골반 중점 (2개 점 + 연결선)
    # 나머지: 핵심 관절 1~3개

    sh_mid_pt = (
        int((pt(11)[0]+pt(12)[0])/2),
        int((pt(11)[1]+pt(12)[1])/2),
    )
    hp_mid_pt = (
        int((pt(23)[0]+pt(24)[0])/2),
        int((pt(23)[1]+pt(24)[1])/2),
    )

    # {key: (관절 포인트 리스트, 대표 포인트, ref 포인트)}
    # TIA 목표: 어깨가 골반 바로 위 수직 (어깨→골반 수직)
    hp_x_ref   = pt(hp_idx)[0]
    tia_target = (hp_x_ref, sh_mid_pt[1])       # 어깨 목표: 골반 x, 어깨 y
    # CVA 목표: 코가 어깨 바로 위 수직 (코→어깨 수직, DA팀 기준)
    # TIA Bad면 어깨 목표 x 기준, TIA Good이면 현재 어깨 x 기준
    sh_x_ref   = pt(sh_idx)[0]
    cva_target = (sh_x_ref, pt(nose_idx)[1])    # 코 목표: 현재 어깨 x, 코 y

    JOINT_DEF = {
        'cva': {
            'pts':   [pt(nose_idx), pt(sh_idx)],
            'main':  pt(nose_idx),
            'ref':   cva_target,
            'line':  True,
        },
        'tia': {
            'pts':   [sh_mid_pt, hp_mid_pt],
            'main':  sh_mid_pt,
            'ref':   tia_target,
            'line':  True,
        },
        'elbow': {
            'pts':   [pt(el_idx)],
            'main':  pt(el_idx),
            'ref':   pt(sh_idx),
            'line':  False,
        },
        'knee': {
            'pts':   [pt(kn_idx)],
            'main':  pt(kn_idx),
            'ref':   pt(hp_idx),
            'line':  False,
        },
        'wrist': {
            'pts':   [pt(wr_idx)],
            'main':  pt(wr_idx),
            'ref':   pt(el_idx),
            'line':  False,
        },
        'gaze': {
            'pts':   [pt(nose_idx)],
            'main':  pt(nose_idx),
            'ref':   (pt(nose_idx)[0], pt(nose_idx)[1]-OFFSET_PX),
            'line':  False,
        },
        'desk_h': {
            'pts':   [pt(el_idx)],
            'main':  pt(el_idx),
            'ref':   (pt(el_idx)[0], pt(el_idx)[1]-OFFSET_PX),
            'line':  False,
        },
        'chair_d': {
            'pts':   [pt(hp_idx)],
            'main':  pt(hp_idx),
            'ref':   (pt(hp_idx)[0]+OFFSET_PX, pt(hp_idx)[1]),
            'line':  False,
        },
    }

    keys = ['cva','tia'] if early_stop else list(JOINT_DEF.keys())

    # ── CVA/TIA: MLP 각도값 기반 현재 자세 + 목표 자세 표시 ──────────
    cva_ok    = all_ind.get('cva', {}).get('ok')
    tia_ok    = all_ind.get('tia', {}).get('ok')
    cva_val   = all_ind.get('cva', {}).get('value')  # MLP가 계산한 실제 각도
    tia_val   = all_ind.get('tia', {}).get('value')
    angle_bad = (cva_ok is False) or (tia_ok is False)

    hp_pt   = pt(hp_idx)    # 골반 (기준, 고정)
    sh_pt   = sh_mid_pt     # 어깨 (현재 MLP 기반)
    nose_pt = pt(nose_idx)  # 코 (현재 MLP 기반)

    def draw_spine_line(pts, color, thickness=2):
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], (255,255,255), thickness+2, cv2.LINE_AA)
            cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)
        for i in range(len(pts)-1):
            p1, p2 = pts[i], pts[i+1]
            for t in [0.25, 0.5, 0.75]:
                mx = int(p1[0]+(p2[0]-p1[0])*t)
                my = int(p1[1]+(p2[1]-p1[1])*t)
                cv2.circle(img, (mx,my), 4, color,        -1)
                cv2.circle(img, (mx,my), 6, (255,255,255), 1)
        for p in pts:
            cv2.circle(img, p, 7, color,        -1)
            cv2.circle(img, p, 9, (255,255,255), 2)

    def draw_arrow(src, dst, color):
        cv2.arrowedLine(img, src, dst, (255,255,255), 4, tipLength=0.25, line_type=cv2.LINE_AA)
        cv2.arrowedLine(img, src, dst, color,         2, tipLength=0.25, line_type=cv2.LINE_AA)

    # ── 현재 자세: MLP 각도값 그대로 점 찍기 ─────────────────────────
    cur_color = COLOR_BAD if angle_bad else COLOR_GOOD
    draw_spine_line([hp_pt, sh_pt, nose_pt], cur_color)

    # ── 목표 자세: Bad일 때만 표시 ───────────────────────────────────
    if angle_bad:
        import math

        # 어깨-골반 거리 (TIA 목표 거리 기준)
        sh_hp_dist = math.hypot(sh_pt[0]-hp_pt[0], sh_pt[1]-hp_pt[1])
        # 코-어깨 거리 (CVA 목표 거리 기준)
        nose_sh_dist = math.hypot(nose_pt[0]-sh_pt[0], nose_pt[1]-sh_pt[1])

        # TIA 목표: Good 중앙값 5° 기울어진 위치
        # 측면 사진에서 앞으로 기울어짐 → x는 골반보다 앞(작은 x)으로
        tia_goal_deg = math.radians(5)
        if tia_ok is False:
            dx = -int(sh_hp_dist * math.sin(tia_goal_deg))  # 앞쪽으로
            dy = -int(sh_hp_dist * math.cos(tia_goal_deg))  # 위쪽으로
            sh_tgt = (hp_pt[0] + dx, hp_pt[1] + dy)
        else:
            sh_tgt = sh_pt

        # CVA 목표: Good 중앙값 10° 기울어진 위치
        # 코는 어깨보다 앞으로 나와있는 게 정상
        cva_goal_deg = math.radians(10)
        if cva_ok is False:
            dx = -int(nose_sh_dist * math.sin(cva_goal_deg))
            dy = -int(nose_sh_dist * math.cos(cva_goal_deg))
            nose_tgt = (sh_tgt[0] + dx, sh_tgt[1] + dy)
        else:
            nose_tgt = nose_pt

        # 목표 자세 연결선 + 점
        draw_spine_line([hp_pt, sh_tgt, nose_tgt], COLOR_TARGET)

        # 화살표: Bad인 지표만
        if tia_ok is False and sh_tgt != sh_pt:
            draw_arrow(sh_pt, sh_tgt, COLOR_TARGET)
        if cva_ok is False and nose_tgt != nose_pt:
            draw_arrow(nose_pt, nose_tgt, COLOR_TARGET)

    # ── 나머지 6개 지표: 번호 뱃지 + 라벨 + 점 ─────────────────────────
    SIX_LABELS = {
        'elbow':   'Elbow',  'knee':    'Knee',
        'wrist':   'Wrist',  'gaze':    'Gaze',
        'desk_h':  'Desk',   'chair_d': 'Chair',
    }
    KEY_ORDER = ['cva','tia','elbow','knee','wrist','gaze','desk_h','chair_d']
    for key in keys:
        if key in ('cva', 'tia'): continue
        num   = KEY_ORDER.index(key) + 1
        info  = all_ind.get(key)
        label = SIX_LABELS.get(key, key)
        if not info: continue
        ok = info.get('ok')

        defn    = JOINT_DEF[key]
        main_pt = defn['main']
        ref_pt  = defn['ref']

        if ok is None:
            # 측정불가: 회색 점 + 라벨
            color = COLOR_NA
            cv2.circle(img, main_pt, 8,  color,         -1)
            cv2.circle(img, main_pt, 10, (255,255,255),  1)
            cv2.putText(img, f"{num}.{label}?",
                        (main_pt[0]+12, main_pt[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        else:
            color = COLOR_GOOD if ok else COLOR_BAD

            # 관절 점 (크게)
            cv2.circle(img, main_pt, 10, color,         -1)
            cv2.circle(img, main_pt, 12, (255,255,255),  2)

            # 번호 뱃지
            num_pt = (main_pt[0] - 15, main_pt[1] - 15)
            cv2.circle(img, num_pt, 9,  (255,255,255),  -1)
            cv2.circle(img, num_pt, 10, color,            2)
            cv2.putText(img, str(num),
                        (num_pt[0]-4 if num<10 else num_pt[0]-6, num_pt[1]+4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (20,20,20), 1, cv2.LINE_AA)

            # 라벨 텍스트
            lbl_x, lbl_y = main_pt[0]+14, main_pt[1]+5
            cv2.putText(img, label,
                        (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255,255,255), 3, cv2.LINE_AA)  # 흰 테두리
            cv2.putText(img, label,
                        (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        color, 1, cv2.LINE_AA)

            if not ok:
                _draw_bad_arrow(img, main_pt, ref_pt)


    return img


# =====================================================================
# 전체 파이프라인
# =====================================================================
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
        self.root.title("Fit me up | 자세 통합 분석")
        self.root.geometry("1200x820")
        self.root.configure(bg="#f4f4f4")
        self.image_path = None
        self.tk_img     = None
        self._setup_ui()

    def _setup_ui(self):
        tk.Label(self.root, text="🧘 Fit me up  |  자세 통합 분석",
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
