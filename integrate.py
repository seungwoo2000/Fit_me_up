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
COLOR_GOOD      = (0, 158, 29)     # 초록
COLOR_BAD       = (50,  50, 226)   # 빨강
COLOR_TARGET    = (0, 200, 100)    # 목표 초록
COLOR_BONE      = (200, 200, 200)  # 뼈대 연결선
COLOR_HUD_BG    = (20,  15,  40)   # HUD 배경
OFFSET_PX       = 32

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

    all_ind = {}

    # CVA 판정
    if cva is not None:
        ear_idx, sh_idx = best_pair(lm, 7, 8, 11, 12)
        all_ind['cva'] = {
            'value': round(cva, 1),
            'ok':    cva <= THRESHOLD_CVA,
            'joints': (ear_idx, sh_idx)
        }
    else:
        all_ind['cva'] = {'value': None, 'ok': None, 'joints': (8, 12)}

    # TIA 판정
    if tia is not None:
        sh_idx2, hp_idx = best_pair(lm, 11, 12, 23, 24)
        all_ind['tia'] = {
            'value': round(tia, 1),
            'ok':    tia <= THRESHOLD_TIA,
            'joints': (sh_idx2, hp_idx)
        }
    else:
        all_ind['tia'] = {'value': None, 'ok': None, 'joints': (12, 24)}

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
def step5_overlay(image_path, lm, h, w, all_ind, early_stop):
    img = cv2.imread(image_path)
    ih, iw = img.shape[:2]

    def pt(idx): return to_px(lm, idx, w, h)

    # ── 뼈대 연결선 (visibility 높은 쪽만) ───────────────────────────
    sh_idx, hp_idx = best_pair(lm, 11, 12, 23, 24)
    ear_idx        = best_idx(lm, 7, 8)
    el_idx         = best_idx(lm, 13, 14)
    wr_idx         = best_idx(lm, 15, 16)
    kn_idx         = best_idx(lm, 25, 26)
    an_idx         = best_idx(lm, 27, 28)

    bones = [
        (ear_idx, sh_idx),
        (sh_idx,  hp_idx),
        (sh_idx,  el_idx),
        (el_idx,  wr_idx),
        (hp_idx,  kn_idx),
        (kn_idx,  an_idx),
    ]
    for a, b in bones:
        if is_vis(lm, a) and is_vis(lm, b):
            cv2.line(img, pt(a), pt(b), COLOR_BONE, 2, cv2.LINE_AA)

    # ── 관절 점 + Bad 목표 점 ────────────────────────────────────────
    JOINT_REF = {
        'cva':     lambda: (pt(ear_idx), pt(sh_idx)),
        'tia':     lambda: (pt(sh_idx),  pt(hp_idx)),
        'elbow':   lambda: (pt(el_idx),  pt(sh_idx)),
        'knee':    lambda: (pt(kn_idx),  pt(hp_idx)),
        'wrist':   lambda: (pt(wr_idx),  pt(el_idx)),
        'gaze':    lambda: (pt(ear_idx), (pt(ear_idx)[0], pt(ear_idx)[1]-OFFSET_PX)),
        'desk_h':  lambda: (pt(el_idx),  (pt(el_idx)[0],  pt(el_idx)[1]-OFFSET_PX)),
        'chair_d': lambda: (pt(hp_idx),  (pt(hp_idx)[0]+OFFSET_PX, pt(hp_idx)[1])),
    }

    keys = ['cva','tia'] if early_stop else list(JOINT_REF.keys())

    num = 1  # 관절 번호 (피드백 카드와 매칭)
    for key in keys:
        info = all_ind.get(key)
        if not info: continue
        ok = info.get('ok')
        if ok is None:
            num += 1
            continue

        joint_pt, ref_pt = JOINT_REF[key]()

        if ok:
            cv2.circle(img, joint_pt, 11, COLOR_GOOD,        -1)
            cv2.circle(img, joint_pt, 13, (255,255,255),       2)
        else:
            # Bad 관절 → 빨간 점
            cv2.circle(img, joint_pt, 11, COLOR_BAD,          -1)
            cv2.circle(img, joint_pt, 13, (255,255,255),       2)
            # 목표 초록 점
            tgt = target_pt(joint_pt, ref_pt)
            cv2.circle(img, tgt,       9, COLOR_TARGET,       -1)
            cv2.circle(img, tgt,       11, (255,255,255),      2)
            # 화살표
            cv2.arrowedLine(img, joint_pt, tgt, (180,180,180), 1, tipLength=0.35)
            # 수치 텍스트
            v = info.get('value')
            if v is not None:
                unit = IND_UNITS.get(key, '')
                cv2.putText(img, f"{v}{unit}",
                            (joint_pt[0]+14, joint_pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                            (80, 80, 230), 1, cv2.LINE_AA)

        # 번호 원형 배경 + 텍스트
        num_pt = (joint_pt[0] - 18, joint_pt[1] - 18)
        num_color = COLOR_GOOD if ok else COLOR_BAD
        cv2.circle(img, num_pt, 9, num_color, -1)
        cv2.circle(img, num_pt, 10, (255,255,255), 1)
        cv2.putText(img, str(num),
                    (num_pt[0] - 4 if num < 10 else num_pt[0] - 6, num_pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255,255,255), 1, cv2.LINE_AA)
        num += 1

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

    if not early_stop:
        # Step 3
        bboxes = step3_yolo(image_path)
        # Step 4
        rest   = step4_remaining(lm, h, w, bboxes, img_w, img_h)
        all_ind.update(rest)

    # Step 5
    overlay = step5_overlay(image_path, lm, h, w, all_ind, early_stop)

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

        # ── 상단: 이미지 + CNN 판정 ───────────────────────────────────
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

        # CNN 판정 (이미지 오른쪽)
        cnn_area = tk.Frame(top, bg="#f4f4f4")
        cnn_area.pack(side=tk.LEFT, padx=20, anchor="n")

        self.res_panel = tk.Frame(cnn_area, width=200, height=90,
                                  bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        self.res_panel.pack_propagate(False)
        self.res_panel.pack(pady=(10,4))
        self.lbl_result = tk.Label(self.res_panel, text="READY",
                                   font=("Arial", 28, "bold"),
                                   bg="#ecf0f1", fg="#7f8c8d")
        self.lbl_result.pack(expand=True)
        self.lbl_conf = tk.Label(cnn_area, text="신뢰도: —",
                                 font=("Malgun Gothic", 10), bg="#f4f4f4", fg="#555")
        self.lbl_conf.pack()

        # ── 하단: 피드백 카드 전체 너비 ──────────────────────────────
        tk.Label(self.root, text="피드백",
                 font=("Malgun Gothic", 12, "bold"),
                 bg="#f4f4f4").pack(anchor="w", padx=18, pady=(4,2))

        fb_outer = tk.Frame(self.root, bg="#f4f4f4")
        fb_outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0,10))

        self.fb_canvas = tk.Canvas(fb_outer, bg="#f4f4f4", highlightthickness=0)
        scrollbar      = tk.Scrollbar(fb_outer, orient="vertical", command=self.fb_canvas.yview)
        self.fb_frame  = tk.Frame(self.fb_canvas, bg="#f4f4f4")

        self.fb_frame.bind("<Configure>",
            lambda e: self.fb_canvas.configure(scrollregion=self.fb_canvas.bbox("all")))
        self.fb_canvas.create_window((0,0), window=self.fb_frame, anchor="nw")
        self.fb_canvas.configure(yscrollcommand=scrollbar.set)

        self.fb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.ind_labels = {}

    # ── 피드백 카드 생성 ──────────────────────────────────────────────
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

        cnn, all_ind, overlay, early_stop = run_pipeline(self.image_path)

        if isinstance(early_stop, str):
            messagebox.showerror("Error", early_stop)
            self.reset_labels()
            return

        # CNN 결과
        label = cnn.get('label','?').upper()
        conf  = cnn.get('confidence', 0) * 100
        color = "#27ae60" if label == "GOOD" else "#e74c3c"
        self.lbl_result.config(text=label, bg=color, fg="white")
        self.res_panel.config(bg=color)
        self.lbl_conf.config(text=f"신뢰도: {conf:.1f}%")

        # 피드백 카드
        self._build_feedback_cards(all_ind, early_stop)

        # 오버레이 이미지
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        self._show_pil(Image.fromarray(overlay_rgb))

    # ── 초기화 ────────────────────────────────────────────────────────
    def reset(self):
        self.image_path = None
        self.canvas.delete("all")
        self.reset_labels()

    def reset_labels(self):
        self.lbl_result.config(text="READY", bg="#ecf0f1", fg="#7f8c8d")
        self.res_panel.config(bg="#ecf0f1")
        self.lbl_conf.config(text="신뢰도: —")
        for w in self.fb_frame.winfo_children():
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
