# =============================================
# predict.py | 자세 분석 AI - MLP 기반 예측
# DA팀 작성 / TA팀 integrate.py 연동용
# =============================================
import os
import random
import warnings
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import tensorflow as tf
from tensorflow.keras import models

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── 시드 고정 ─────────────────────────────────────────────────────────
def set_all_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_all_seeds(42)

# ── 경로 설정 (code/ 기준 → models/ 폴더) ────────────────────────────
BASE_DIR             = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR           = os.path.join(BASE_DIR, '..', 'models')
PICKLE_PATH          = os.path.join(MODELS_DIR, 'posture_mlp_final_v2.pkl')
MODEL_PATH           = os.path.join(MODELS_DIR, 'posture_mlp_model.h5')
POSE_LANDMARKER_PATH = os.path.join(MODELS_DIR, 'pose_landmarker_full.task')

THRESHOLD            = 0.30
VISIBILITY_THRESHOLD = 0.50

# ── MediaPipe 초기화 (모듈 import 시 1회) ────────────────────────────
detector = None
if os.path.exists(POSE_LANDMARKER_PATH):
    base_options = python.BaseOptions(model_asset_path=POSE_LANDMARKER_PATH)
    options      = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.PoseLandmarker.create_from_options(options)
else:
    print(f"[WARN] MediaPipe 모델 없음: {POSE_LANDMARKER_PATH}")

# ── MLP 모델 싱글톤 ───────────────────────────────────────────────────
_meta  = None
_model = None

def _load_models():
    global _meta, _model
    if _meta is None:
        _meta  = joblib.load(PICKLE_PATH)
    if _model is None:
        _model = models.load_model(MODEL_PATH)


# =============================================
# 관절 추출
# =============================================
def extract_landmarks(image_path):
    """
    이미지에서 MediaPipe 랜드마크 추출

    Returns:
        (landmarks dict, img_w, img_h) 또는 None
        landmarks: {idx: {'x', 'y', 'vis'}}
    """
    if detector is None:
        return None

    mp_image         = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)

    if not detection_result.pose_landmarks:
        return None

    landmarks_list = detection_result.pose_landmarks[0]
    landmarks      = {
        i: {'x': lm.x, 'y': lm.y, 'vis': lm.visibility}
        for i, lm in enumerate(landmarks_list)
    }

    img_cv = cv2.imread(image_path)
    return landmarks, img_cv.shape[1], img_cv.shape[0]


# =============================================
# 피처 엔지니어링
# =============================================
def _calc_vertical_angle(p1, p2):
    dx, dy = p1[0] - p2[0], p1[1] - p2[1]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))

def _calc_horizontal_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))

def engineer_features(landmarks, feature_cols):
    """
    랜드마크 딕셔너리 → MLP 입력 피처 배열 생성

    Returns:
        (feature_array, cva, tia)
    """
    fd = {}
    for i in range(33):
        fd[f'관절{i}_x']   = landmarks[i]['x']
        fd[f'관절{i}_y']   = landmarks[i]['y']
        fd[f'관절{i}_vis'] = landmarks[i]['vis']

    def best_pt(i1, i2):
        if landmarks[i1]['vis'] >= landmarks[i2]['vis']:
            return (landmarks[i1]['x'], landmarks[i1]['y'])
        return (landmarks[i2]['x'], landmarks[i2]['y'])

    nose = (landmarks[0]['x'], landmarks[0]['y'])
    sh   = best_pt(11, 12)
    hip  = best_pt(23, 24)

    fd['CVA']           = _calc_vertical_angle(nose, sh)
    fd['TIA']           = _calc_vertical_angle(sh, hip)
    fd['Shoulder_Slope'] = _calc_horizontal_angle(
        (landmarks[11]['x'], landmarks[11]['y']),
        (landmarks[12]['x'], landmarks[12]['y'])
    )
    fd['Pelvis_Slope']  = _calc_horizontal_angle(
        (landmarks[23]['x'], landmarks[23]['y']),
        (landmarks[24]['x'], landmarks[24]['y'])
    )

    arr = np.array([fd.get(col, 0) for col in feature_cols]).reshape(1, -1)
    return arr, fd['CVA'], fd['TIA']


# =============================================
# 메인 예측 함수 (integrate.py 연동용)
# =============================================
def predict_posture(image_path):
    """
    이미지 경로를 받아 MLP 기반 자세 판정

    Returns:
        dict: {
            "label":              "good" or "bad",
            "confidence":         float,
            "threshold":          0.30,
            "CVA":                float (목굴곡각 °),
            "TIA":                float (몸통굴곡각 °),
            "landmarks_detected": bool,
            "raw_landmarks":      dict  (integrate.py Step 1 재활용용)
        }
        또는 None (실패 시)
    """
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    if not os.path.exists(PICKLE_PATH) or not os.path.exists(MODEL_PATH):
        return {"error": "MLP 모델 파일 없음 (models/ 폴더 확인)"}

    try:
        _load_models()

        # 관절 추출
        extract_res = extract_landmarks(image_path)
        if not extract_res:
            return {"error": "관절 탐지 실패 — 측면 전신 이미지 권장"}
        landmarks, w, h = extract_res

        # 피처 → 스케일링 → 예측
        features, cva, tia = engineer_features(landmarks, _meta['feature_cols'])
        features_sc        = _meta['scaler'].transform(features)
        prob_good          = float(_model.predict(features_sc, verbose=0)[0][0])
        prob_bad           = 1.0 - prob_good

        label      = "bad" if prob_bad >= THRESHOLD else "good"
        confidence = prob_bad if label == "bad" else prob_good

        return {
            "label":              label,
            "confidence":         round(confidence, 4),
            "threshold":          THRESHOLD,
            "CVA":                round(cva, 2),
            "TIA":                round(tia, 2),
            "landmarks_detected": True,
            "raw_landmarks":      landmarks,
            "img_w":              w,
            "img_h":              h,
        }

    except Exception as e:
        return {"error": f"예측 실패: {str(e)}"}


# =============================================
# 단독 실행
# =============================================
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='분석할 이미지 경로')
    args = parser.parse_args()

    print("=" * 40)
    print("자세 분석 AI (MLP v2.0)")
    print("=" * 40)

    res = predict_posture(args.image)

    if res is None or "error" in res:
        print(f"실패: {res.get('error') if res else '알 수 없는 오류'}")
    else:
        print(f"\n판정:      {res['label'].upper()}")
        print(f"신뢰도:    {res['confidence']:.1%}")
        print(f"CVA:       {res['CVA']:.1f}°")
        print(f"TIA:       {res['TIA']:.1f}°")

    if detector:
        detector.close()
    print("=" * 40)
