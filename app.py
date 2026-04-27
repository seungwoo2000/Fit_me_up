# app.py
# streamlit run app.py

import os
import sys
import math
import warnings
import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components

from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="Fit Me Up — AI 자세 분석",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 본인 경로에 맞게 수정
YOLO_MODEL = r"E:\python\FIT_ME_UP\YOLO\fit_me_up\combined_gpu\weights\best.pt"
MODEL_PATH_MP = r"E:\python\FIT_ME_UP\MediaPipe\models\pose_landmarker.task"

CLASS_NAMES = {
    0: "chair",
    1: "desk",
    2: "monitor",
}

FEEDBACK = {
    "CVA": {
        "no": "01",
        "label": "목굴곡각",
        "eng": "CVA",
        "range": "0° ~ 20°",
        "cat": "posture",
        "good": "머리·경추 수직 정렬 유지\n경추 부담 최소화 상태",
        "bad": "전방두부자세(FHP) 의심\n모니터를 눈높이로 올리세요\n1시간마다 목 스트레칭 시행",
    },
    "TIA": {
        "no": "02",
        "label": "몸통굴곡각",
        "eng": "TIA",
        "range": "0° ~ 10°",
        "cat": "posture",
        "good": "척추 수직 정렬 양호\n요추 압박 최소화 상태",
        "bad": "과도한 몸통 전굴 감지\n등받이에 허리 완전 밀착\n의자 깊숙이 앉으세요",
    },
    "팔꿈치": {
        "no": "03",
        "label": "팔꿈치 각도",
        "eng": "Elbow",
        "range": "90° ~ 120°",
        "cat": "posture",
        "good": "상지 관절 부하 최적 범위\n팔꿈치·책상면 수평 유지",
        "bad": "팔꿈치 각도 기준 이탈\n의자 높이 조정 필요\n팔꿈치·책상면 수평 유지",
    },
    "무릎": {
        "no": "04",
        "label": "무릎 각도",
        "eng": "Knee",
        "range": "85° ~ 100°",
        "cat": "posture",
        "good": "하지 혈액순환 원활\n하체 부담 최소화 상태",
        "bad": "무릎 각도 기준 이탈\n의자 높이 조절 필요\n발받침대 사용 권장",
    },
    "손목": {
        "no": "05",
        "label": "손목 각도",
        "eng": "Wrist",
        "range": "165° ~ 180°",
        "cat": "posture",
        "good": "손목 중립 자세 유지\n손목 터널 부담 최소화",
        "bad": "손목 과굴곡 감지\n손목 받침대 설치 필요\n키보드 앞 15cm 확보",
    },
    "시선각": {
        "no": "06",
        "label": "모니터 시선각",
        "eng": "Gaze",
        "range": "하방 10° ~ 15°",
        "cat": "env",
        "good": "시선각 기준 충족\n경추 부담 최소화",
        "bad": "시선각 기준 이탈\n모니터 상단을 눈높이에 맞추세요\n화면 거리 40cm 이상 권장",
    },
    "책상높이": {
        "no": "07",
        "label": "작업대 높이",
        "eng": "Desk",
        "range": "팔꿈치 수평 ±10%",
        "cat": "env",
        "good": "작업대·팔꿈치 정렬 양호\n상지 부담 최소화",
        "bad": "작업대 높이 불일치\n책상 높이 또는 의자 높이 조정 필요",
    },
    "등받이": {
        "no": "08",
        "label": "의자 등받이",
        "eng": "Chair",
        "range": "골반너비 20% 이내",
        "cat": "env",
        "good": "등받이 지지 충분\n요추 안정성 확보",
        "bad": "등받이 지지 부족\n의자 깊숙이 착석\n허리 완전 밀착 필요",
    },
}

def build_ai_correction_comment(result):
    all_data = {**result["posture"], **result["env"]}

    GUIDE = {
        "CVA": {
            "part": "목·경추",
            "bad": "고개가 앞으로 기울어진 전방두부자세 가능성이 있습니다. 모니터 상단을 눈높이에 맞추고 1시간마다 목 스트레칭을 해주세요.",
            "goal": "모니터 높이를 눈높이에 맞추기",
        },
        "TIA": {
            "part": "몸통·허리",
            "bad": "몸통이 앞으로 과도하게 굽혀져 있습니다. 의자 깊숙이 앉아 허리를 등받이에 기대세요.",
            "goal": "골반을 의자 뒤쪽까지 넣고 등받이에 허리 밀착하기",
        },
        "팔꿈치": {
            "part": "팔꿈치·어깨",
            "bad": "팔꿈치 각도가 적절하지 않습니다. 의자 높이를 조정해 팔꿈치가 책상면과 수평이 되게 하세요.",
            "goal": "팔꿈치가 책상면과 수평이 되도록 의자 높이 조정하기",
        },
        "무릎": {
            "part": "무릎·하체",
            "bad": "무릎 각도가 적절하지 않습니다. 의자 높이를 조절해 무릎이 90° 전후가 되도록 하세요.",
            "goal": "무릎이 90° 전후가 되도록 의자 높이와 발 위치 조정하기",
        },
        "손목": {
            "part": "손목",
            "bad": "손목이 과도하게 굽혀져 있습니다. 손목 받침대를 사용하고 키보드 앞 공간을 확보하세요.",
            "goal": "손목 받침대 사용하고 키보드 앞 공간 15cm 이상 확보하기",
        },
        "시선각": {
            "part": "시선·모니터",
            "bad": "모니터 위치가 적절하지 않아 목 부담이 커질 수 있습니다. 모니터 상단을 눈높이에 맞추세요.",
            "goal": "모니터 상단을 눈높이에 맞추고 화면 거리 40cm 이상 확보하기",
        },
        "책상높이": {
            "part": "작업대 높이",
            "bad": "책상 높이가 팔꿈치와 맞지 않습니다. 책상 또는 의자 높이를 조정하세요.",
            "goal": "팔꿈치와 책상면이 수평이 되도록 책상 또는 의자 높이 조정하기",
        },
        "등받이": {
            "part": "의자 등받이",
            "bad": "등받이 지지가 부족합니다. 의자 깊숙이 앉고 요추 부위를 등받이에 밀착하세요.",
            "goal": "의자 깊숙이 앉고 요추 부위를 등받이에 밀착하기",
        },
    }

    bad_items = []
    good_items = []
    goals = []

    for key, (value, is_good, raw) in all_data.items():
        if key not in GUIDE:
            continue

        if is_good:
            good_items.append(f"{GUIDE[key]['part']}({value})")
        else:
            bad_items.append((key, value, GUIDE[key]))
            goals.append(GUIDE[key]["goal"])

    if bad_items:
        first_key, first_value, first_item = bad_items[0]

        detail_html = ""
        for key, value, item in bad_items[:4]:
            detail_html += f"""
            <div style="padding:12px 0;border-top:1px solid #EEF2F6;">
                <div style="font-size:13px;font-weight:800;color:#172033;margin-bottom:4px;">
                    ⚠ {item["part"]} · 측정값 {value}
                </div>
                <div style="font-size:12.5px;line-height:1.7;color:#667085;">
                    {item["bad"]}
                </div>
            </div>
            """

        summary_html = f"""
        <div style="font-size:14px;line-height:1.85;color:#667085;margin-bottom:10px;">
            <b style="color:#172033;">가장 먼저 교정할 부위는 {first_item["part"]}입니다.</b><br>
            기준 범위를 벗어난 항목이 <b style="color:#D94A4A;">{len(bad_items)}개</b> 확인되었습니다.
        </div>
        """

    else:
        summary_html = """
        <div style="font-size:14px;line-height:1.85;color:#667085;">
            <b style="color:#172033;">전체 자세가 안정적입니다.</b><br>
            주요 자세 지표가 대부분 정상 범위에 있습니다.
        </div>
        """
        detail_html = ""
        goals = ["50분 작업 후 5분 스트레칭하기"]

    if good_items:
        good_html = f"""
        <div style="margin-top:12px;padding:12px;border-radius:12px;background:#F0FBF4;font-size:12.5px;line-height:1.7;color:#3B8C42;">
            <b>잘 유지되고 있는 항목</b><br>
            {" · ".join(good_items[:4])}
        </div>
        """
    else:
        good_html = ""

    default_goals = [
        "50분 작업 후 5분 스트레칭하기",
        "목과 어깨를 천천히 돌려 긴장 완화하기",
        "손목이 꺾이지 않도록 키보드와 마우스 위치 조정하기",
        "발바닥이 바닥에 닿는지 확인하기",
    ]

    for g in default_goals:
        if len(goals) >= 4:
            break
        if g not in goals:
            goals.append(g)

    goals_html = "".join([f"{i+1}. {goal}<br>" for i, goal in enumerate(goals[:4])])

    return f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>맞춤 교정 코멘트</span>
        <span class="fit-badge badge-blue">AI Guide</span>
    </div>

    {summary_html}
    {detail_html}
    {good_html}

    <div style="margin-top:16px;padding:14px;border-radius:14px;background:#F8FAFC;">
        <div style="font-size:13px;font-weight:800;color:#172033;margin-bottom:8px;">
            오늘의 실천 목표
        </div>
        <div style="font-size:13px;line-height:1.8;color:#667085;">
            {goals_html}
        </div>
    </div>
</div>
"""

# =========================================================
# 2. CSS — 첨부 HTML 느낌의 세련된 UI
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg: #F5F7FB;
    --panel: #FFFFFF;
    --card: #FFFFFF;
    --line: #E5EAF2;
    --text: #172033;
    --sub: #667085;
    --blue: #185FA5;
    --blue2: #0C447C;
    --teal: #00BFA6;
    --green: #3B8C42;
    --amber: #BA7517;
    --red: #D94A4A;
    --purple: #6E56CF;
    --soft-blue: #E6F1FB;
    --soft-green: #EAF6EE;
    --soft-amber: #FAEEDA;
    --soft-red: #FCEBEB;
}

html, body, [class*="css"] {
    font-family: 'Pretendard', sans-serif;
}

.stApp {
    background: var(--bg);
}

.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
    max-width: 1280px;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

div[role="radiogroup"] label {
    background: transparent;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 4px;
    transition: all .18s ease;
}

div[role="radiogroup"] label:hover {
    background: var(--soft-blue);
}

div[role="radiogroup"] label[data-checked="true"] {
    background: var(--soft-blue);
    border-left: 3px solid var(--blue);
}

.logo-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 4px 14px 4px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 14px;
}

.logo-mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, #185FA5, #00BFA6);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 900;
}

.logo-title {
    font-size: 17px;
    font-weight: 800;
    color: var(--text);
    line-height: 1.1;
}

.logo-sub {
    font-size: 11px;
    color: var(--sub);
    margin-top: 2px;
}

.page-title {
    font-size: 28px;
    font-weight: 850;
    color: var(--text);
    letter-spacing: -0.7px;
    margin-bottom: 6px;
    line-height: 1.35;
}

.page-sub {
    font-size: 14px;
    color: var(--sub);
    margin-bottom: 22px;
}

.fit-card {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 28px rgba(15, 23, 42, 0.04);
    margin-bottom: 16px;
}

.fit-card-title {
    font-size: 15px;
    font-weight: 750;
    color: var(--text);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.fit-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 9px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 700;
}

.badge-blue { background: var(--soft-blue); color: var(--blue2); }
.badge-green { background: var(--soft-green); color: var(--green); }
.badge-amber { background: var(--soft-amber); color: #854F0B; }
.badge-red { background: var(--soft-red); color: #8E2424; }
.badge-gray { background: #F2F4F7; color: var(--sub); }

.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 17px 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.035);
}

.metric-value {
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -0.8px;
}

.metric-label {
    font-size: 12px;
    color: var(--sub);
    margin-top: 4px;
}

.result-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 11px 0;
    border-bottom: 1px solid #EEF2F6;
}

.result-row:last-child {
    border-bottom: 0;
}

.result-name {
    width: 104px;
    font-size: 13px;
    color: var(--sub);
    flex-shrink: 0;
}

.result-value {
    font-size: 14px;
    font-weight: 800;
    color: var(--text);
    width: 70px;
    flex-shrink: 0;
}

.bar-wrap {
    flex: 1;
    height: 8px;
    background: #EEF2F6;
    border-radius: 999px;
    overflow: hidden;
}

.bar {
    height: 8px;
    border-radius: 999px;
}

.bar-green { background: var(--green); }
.bar-amber { background: var(--amber); }
.bar-red { background: var(--red); }
.bar-blue { background: var(--blue); }

.feedback-card {
    border-radius: 16px;
    padding: 15px 16px;
    margin-bottom: 10px;
    border: 1px solid var(--line);
}

.feedback-good {
    background: linear-gradient(135deg, #F0FBF4, #FFFFFF);
    border-left: 4px solid var(--green);
}

.feedback-bad {
    background: linear-gradient(135deg, #FFF1F1, #FFFFFF);
    border-left: 4px solid var(--red);
}

.feedback-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.feedback-name {
    font-size: 14px;
    font-weight: 800;
    color: var(--text);
}

.feedback-msg {
    font-size: 12.5px;
    line-height: 1.65;
    color: var(--sub);
    white-space: pre-line;
}

.upload-box {
    border: 1.5px dashed #CBD5E1;
    background: #F8FAFC;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    color: var(--sub);
}

.fit-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.fit-table th {
    text-align: left;
    color: var(--sub);
    font-weight: 700;
    border-bottom: 1px solid var(--line);
    padding: 10px 6px;
}

.fit-table td {
    border-bottom: 1px solid #EEF2F6;
    padding: 11px 6px;
    color: var(--text);
}

@media (max-width: 900px) {
    .metric-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.main .block-container {
    transition: opacity 0.12s ease-in-out;
}

[data-testid="stAppViewContainer"] {
    background: #F5F7FB;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# 3. 모델 로드
# =========================================================

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception as e:
        return None


@st.cache_resource(show_spinner=False)
def load_mediapipe_pose():
    try:
        import mediapipe as mp

        try:
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
            return {"mode": "legacy", "pose": pose, "mp": mp}

        except Exception:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            if not os.path.exists(MODEL_PATH_MP):
                return {
                    "mode": "error",
                    "error": f"MediaPipe task 모델 파일이 없습니다: {MODEL_PATH_MP}",
                }

            base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH_MP)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
            )
            pose = vision.PoseLandmarker.create_from_options(options)
            return {"mode": "tasks", "pose": pose, "mp": mp}

    except Exception as e:
        return {
            "mode": "error",
            "error": str(e),
        }


# =========================================================
# 4. 분석 함수
# =========================================================

def calc_angle(A, B, C_):
    v1 = np.array(A) - np.array(B)
    v2 = np.array(C_) - np.array(B)
    ct = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(ct, -1, 1)))


def calc_vert(A, B):
    v = (B[0] - A[0], B[1] - A[1])
    return np.degrees(np.arctan2(abs(v[0]), abs(v[1])))


def calc_gaze(eye, monitor_center):
    return np.degrees(np.arctan2(monitor_center[1] - eye[1], monitor_center[0] - eye[0]))


def judge(v, mn, mx):
    if v is None:
        return False
    return mn <= v <= mx


def draw_skeleton(img, lm, h, w):
    conns = [
        (8, 12),
        (12, 14),
        (14, 16),
        (16, 20),
        (12, 24),
        (24, 26),
        (26, 28),
    ]

    color = (190, 215, 0)

    for a, b in conns:
        ax, ay = int(lm[a].x * w), int(lm[a].y * h)
        bx, by = int(lm[b].x * w), int(lm[b].y * h)
        cv2.line(img, (ax, ay), (bx, by), color, 3, cv2.LINE_AA)

    for idx in [8, 12, 14, 16, 20, 24, 26, 28]:
        x, y = int(lm[idx].x * w), int(lm[idx].y * h)
        cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 7, (255, 255, 255), 2, cv2.LINE_AA)


def draw_yolo_boxes(img, bbox):
    color_map = {
        "chair": (40, 150, 255),
        "desk": (255, 205, 40),
        "monitor": (255, 70, 185),
    }

    label_map = {
        "chair": "Chair",
        "desk": "Desk",
        "monitor": "Monitor",
    }

    for name, b in bbox.items():
        if b is None:
            continue

        x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
        color = color_map.get(name, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, max(0, y1 - 28)), (x1 + 110, y1), color, -1)
        cv2.putText(
            img,
            label_map.get(name, name),
            (x1 + 8, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def draw_point(img, pt, is_good, label):
    x, y = int(pt[0]), int(pt[1])
    color = (60, 190, 70) if is_good else (70, 70, 230)

    cv2.circle(img, (x, y), 18, color, 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        img,
        label,
        (x + 20, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def analyze_image(pil_image: Image.Image):
    yolo = load_yolo_model(YOLO_MODEL)
    pose_pack = load_mediapipe_pose()

    if pose_pack is None or pose_pack.get("mode") == "error":
      return {
          "ok": False,
          "message": f"MediaPipe 로드 실패: {pose_pack.get('error', '알 수 없는 오류')}",
      }

    pose = pose_pack["pose"]
    mp = pose_pack["mp"]
    pose_mode = pose_pack["mode"]

    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    if pose_mode == "legacy":
      result = pose.process(img_rgb)

      if not result.pose_landmarks:
          return {
            "ok": False,
            "message": "사람의 자세를 인식하지 못했습니다. 측면 전신 사진을 다시 업로드해주세요.",
          }

      lm = result.pose_landmarks.landmark

    else:
        mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=img_rgb,
        )
        result = pose.detect(mp_image)

        if not result.pose_landmarks:
            return {
              "ok": False,
              "message": "사람의 자세를 인식하지 못했습니다. 측면 전신 사진을 다시 업로드해주세요.",
            }

        lm = result.pose_landmarks[0]

    def gxy(idx):
        return (lm[idx].x * w, lm[idx].y * h)

    ear_r = gxy(8)
    sh_l = gxy(11)
    sh_r = gxy(12)
    elbow_r = gxy(14)
    wrist_r = gxy(16)
    finger = gxy(20)
    hp_l = gxy(23)
    hp_r = gxy(24)
    knee_r = gxy(26)
    ankle_r = gxy(28)

    eye_c = (
        (lm[1].x + lm[4].x) / 2 * w,
        (lm[1].y + lm[4].y) / 2 * h,
    )
    sh_mid = (
        (sh_l[0] + sh_r[0]) / 2,
        (sh_l[1] + sh_r[1]) / 2,
    )
    hp_mid = (
        (hp_l[0] + hp_r[0]) / 2,
        (hp_l[1] + hp_r[1]) / 2,
    )

    bbox = {
        "chair": None,
        "desk": None,
        "monitor": None,
    }

    if yolo is not None:
        try:
            yr = yolo(img_bgr)[0]
            for box in yr.boxes:
                cls_ = int(box.cls[0])
                conf_ = float(box.conf[0])
                name = CLASS_NAMES.get(cls_)

                if name and conf_ >= 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox[name] = {
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "conf": conf_,
                    }
        except Exception:
            pass

    cva = round(calc_vert(ear_r, sh_r), 1)
    tia = round(calc_vert(sh_mid, hp_mid), 1)
    el = round(calc_angle(sh_r, elbow_r, wrist_r), 1)
    kn = round(calc_angle(hp_r, knee_r, ankle_r), 1)
    wr = round(calc_angle(elbow_r, wrist_r, finger), 1)

    gaze = None
    if bbox["monitor"]:
        mx = (bbox["monitor"]["x_min"] + bbox["monitor"]["x_max"]) / 2
        my = (bbox["monitor"]["y_min"] + bbox["monitor"]["y_max"]) / 2
        gaze = round(calc_gaze(eye_c, (mx, my)), 1)

    hd = None
    desk_center = None
    if bbox["desk"]:
        dty = bbox["desk"]["y_min"]
        ref = abs(hp_mid[1] - sh_mid[1])
        hd = round(abs(dty - elbow_r[1]) / (ref + 1e-8), 3)
        desk_center = (
            (bbox["desk"]["x_min"] + bbox["desk"]["x_max"]) // 2,
            dty,
        )

    gr = None
    if bbox["chair"]:
        cbx = bbox["chair"]["x_max"]
        hw = abs(hp_l[0] - hp_r[0]) or abs(sh_l[0] - sh_r[0])
        gr = round(abs(hp_r[0] - cbx) / (hw + 1e-8), 3)

    posture = {
        "CVA": (f"{cva}°", judge(cva, 0, 20), cva),
        "TIA": (f"{tia}°", judge(tia, 0, 10), tia),
        "팔꿈치": (f"{el}°", judge(el, 90, 120), el),
        "무릎": (f"{kn}°", judge(kn, 85, 100), kn),
        "손목": (f"{wr}°", judge(wr, 165, 180), wr),
    }

    env = {
        "시선각": (f"{gaze}°" if gaze is not None else "N/A", judge(gaze, 10, 15), gaze),
        "책상높이": (f"{hd}" if hd is not None else "N/A", judge(hd, 0, 0.10), hd),
        "등받이": (f"{gr}" if gr is not None else "N/A", judge(gr, 0, 0.20), gr),
    }

    overlay = img_bgr.copy()
    draw_skeleton(overlay, lm, h, w)
    draw_yolo_boxes(overlay, bbox)

    draw_point(overlay, ear_r, posture["CVA"][1], "CVA")
    draw_point(overlay, sh_mid, posture["TIA"][1], "TIA")
    draw_point(overlay, elbow_r, posture["팔꿈치"][1], "Elbow")
    draw_point(overlay, knee_r, posture["무릎"][1], "Knee")
    draw_point(overlay, wrist_r, posture["손목"][1], "Wrist")

    if gaze is not None:
        draw_point(overlay, eye_c, env["시선각"][1], "Gaze")

    if desk_center is not None:
        draw_point(overlay, desk_center, env["책상높이"][1], "Desk")

    if gr is not None:
        draw_point(overlay, hp_r, env["등받이"][1], "Chair")

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    all_items = {**posture, **env}

    missing_reason_map = {
        "시선각": "모니터가 인식되지 않아 시선각 평가 제외",
        "책상높이": "책상이 인식되지 않아 책상 높이 평가 제외",
        "등받이": "의자 등받이가 인식되지 않아 등받이 지지 평가 제외",
    }
    valid_items = {k: v for k, v in all_items.items() if v[0] != "N/A"}
    missing_items = [
        {
            "key": k,
            "label": FEEDBACK[k]["label"],
            "reason": missing_reason_map.get(k, "사진에서 기준점이 보이지 않아 평가 제외"),
        }
        for k, v in all_items.items()
        if v[0] == "N/A"
    ]

    good_count = sum(1 for v in valid_items.values() if v[1])
    total_count = len(valid_items)

    score = round((good_count / total_count) * 10, 1) if total_count else 0

    if score >= 7:
        risk = "양호"
    elif score >= 4:
        risk = "주의"
    else:
        risk = "위험"

    return {
        "ok": True,
        "message": "분석 완료",
        "overlay": overlay_rgb,
        "posture": posture,
        "env": env,
        "score": score,
        "risk": risk,
        "good_count": good_count,
        "total_count": total_count,
        "missing_items": missing_items,
    }


# =========================================================
# 5. UI 유틸
# =========================================================

def render_logo():
    st.sidebar.markdown(
        """
<div class="logo-box">
    <div class="logo-mark">F</div>
    <div>
        <div class="logo-title">Fit Me Up</div>
        <div class="logo-sub">AI 자세 분석 서비스</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_header(title, subtitle):
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(value, label, color="#185FA5"):
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-value" style="color:{color}">{value}</div>
    <div class="metric-label">{label}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def status_badge(is_good):
    if is_good:
        return "<span class='fit-badge badge-green'>GOOD</span>"
    return "<span class='fit-badge badge-red'>BAD</span>"


def value_to_bar_width(key, raw):
    if raw is None:
        return 12

    ranges = {
        "CVA": 40,
        "TIA": 30,
        "팔꿈치": 180,
        "무릎": 180,
        "손목": 180,
        "시선각": 45,
        "책상높이": 0.35,
        "등받이": 0.5,
    }

    max_v = ranges.get(key, 100)
    width = min(max(float(raw) / max_v * 100, 8), 100)
    return width


def render_result_rows(data):
    html = ""
    for key, (value, is_good, raw) in data.items():
        width = value_to_bar_width(key, raw)
        bar_class = "bar-green" if is_good else "bar-red"
        html += f"""
<div class="result-row">
    <div class="result-name">{FEEDBACK[key]["label"]}</div>
    <div class="result-value">{value}</div>
    <div class="bar-wrap">
        <div class="bar {bar_class}" style="width:{width}%"></div>
    </div>
    {status_badge(is_good)}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_feedback_cards(data):
    for key, (value, is_good, raw) in data.items():
        fb = FEEDBACK[key]
        msg = fb["good"] if is_good else fb["bad"]
        cls = "feedback-good" if is_good else "feedback-bad"
        badge = status_badge(is_good)

        st.markdown(
            f"""
<div class="feedback-card {cls}">
    <div class="feedback-top">
        <div class="feedback-name">{fb["no"]}. {fb["label"]} <span style="color:#667085;font-size:12px;">({fb["eng"]})</span></div>
        {badge}
    </div>
    <div style="font-size:12px;color:#98A2B3;margin-bottom:6px;">정상 범위: {fb["range"]} · 측정값: {value}</div>
    <div class="feedback-msg">{msg}</div>
</div>
""",
            unsafe_allow_html=True,
        )


def init_history():
    if "history" not in st.session_state:
        st.session_state.history = []

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None


def save_history(result):
    init_history()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state.history.insert(
        0,
        {
            "time": now,
            "score": result["score"],
            "risk": result["risk"],
            "good": result["good_count"],
            "total": result["total_count"],
            "missing_items": result.get("missing_items", []),
        },
    )


def render_measurement_coverage(result_or_history):
    missing_items = result_or_history.get("missing_items", []) or []
    total = result_or_history.get("total_count", result_or_history.get("total", 0))
    good = result_or_history.get("good_count", result_or_history.get("good", 0))

    if missing_items:
        missing_text = "<br>".join(
            [f"- {item['label']}: {item['reason']}" for item in missing_items]
        )
        badge = "일부 제외"
        badge_class = "badge-amber"
        body = (
            f"양호 지표는 <b>{good}/{total}</b>입니다.<br>"
            f"총 8개 항목 중 <b>{len(missing_items)}개 항목</b>은 사진에서 기준점이 부족해 계산에서 제외했습니다.<br><br>"
            f"<b>제외된 항목</b><br>{missing_text}"
        )
    else:
        badge = "전체 측정"
        badge_class = "badge-green"
        body = (
            f"양호 지표는 <b>{good}/{total}</b>입니다.<br>"
            f"총 8개 항목이 모두 인식되었고, 그중 <b>{good}개 항목</b>이 정상 범위로 판정되었습니다."
        )

    st.markdown(
        f"""
<div class="fit-card" style="padding:16px 18px;">
    <div class="fit-card-title" style="margin-bottom:8px;">
        <span>양호 지표 계산 기준</span>
        <span class="fit-badge {badge_class}">{badge}</span>
    </div>
    <div style="font-size:13px;line-height:1.8;color:#667085;">
{body}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =========================================================
# 6. 사이드바 탭
# =========================================================

render_logo()

menu = st.sidebar.radio(
    "나의 건강",
    [
        "🏠 대시보드",
        "📸 자세측정",
        "📈 측정이력",
        "📄 근골격계 리포트",
        "🧾 예상 영수증",
        "🎯 바른자세 챌린지",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 서비스 상태")
st.sidebar.caption("YOLO · MediaPipe 기반 자세 분석")
st.sidebar.caption("측면 사진 기준")
st.sidebar.caption("의료 진단이 아닌 자세 위험도 참고용")


# =========================================================
# 7. 페이지 함수형 렌더링 구조
# =========================================================

def risk_style(is_good):
    if is_good:
        return {
            "label": "양호",
            "color": "#3B8C42",
            "badge": "badge-green",
            "emoji": "🟢",
        }
    return {
        "label": "관리 필요",
        "color": "#D94A4A",
        "badge": "badge-red",
        "emoji": "🔴",
    }


def get_priority_items(result):
    all_data = {**result["posture"], **result["env"]}
    bad_items = []

    for key, value in all_data.items():
        measured_value, is_good, raw = value
        if not is_good:
            bad_items.append((key, measured_value, raw))

    return bad_items


def render_dashboard():
    page_header(
        "나의 자세 현황 대시보드",
        "최근 자세 분석 결과를 바탕으로 위험 부위와 교정 우선순위를 확인합니다.",
    )

    result = st.session_state.get("latest_result", None)
    history = st.session_state.get("history", [])

    if result is None:
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>아직 분석 결과가 없습니다</span>
        <span class="fit-badge badge-blue">Ready</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#667085;">
        먼저 왼쪽 메뉴에서 <b style="color:#172033;">자세측정</b>을 실행하면,
        이 대시보드에 최근 자세 점수, 위험 부위, 교정 우선순위가 자동으로 표시됩니다.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    all_data = {**result["posture"], **result["env"]}
    bad_items = get_priority_items(result)

    good_rate = round(result["good_count"] / result["total_count"] * 100) if result["total_count"] else 0
    bad_count = result["total_count"] - result["good_count"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(result["score"], "최근 자세 점수", "#185FA5")
    with c2:
        metric_card(result["risk"], "종합 위험도", "#BA7517")
    with c3:
        metric_card(f"{bad_count}개", "관리 필요 지표", "#D94A4A")
    with c4:
        metric_card(f"{good_rate}%", "정상 범위 비율", "#3B8C42")

    render_measurement_coverage(result)

    left, right = st.columns([1.15, 0.85])

    with left:
        rows = ""

        label_map = {
            "CVA": "목·경추",
            "TIA": "몸통·허리",
            "팔꿈치": "팔꿈치",
            "무릎": "무릎",
            "손목": "손목",
            "시선각": "시선·모니터",
            "책상높이": "책상 높이",
            "등받이": "의자 등받이",
        }

        for key, (value, is_good, raw) in all_data.items():
            style = risk_style(is_good)
            width = value_to_bar_width(key, raw)

            rows += f"""
<div class="result-row">
    <div class="result-name">{label_map.get(key, key)}</div>
    <div class="result-value">{value}</div>
    <div class="bar-wrap">
        <div class="bar" style="width:{width}%; background:{style["color"]};"></div>
    </div>
    <span class="fit-badge {style["badge"]}">{style["label"]}</span>
</div>
"""

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>최근 측정 기반 신체 부위별 위험 현황</span>
        <span class="fit-badge badge-blue">Live Result</span>
    </div>
    {rows}
</div>
""",
            unsafe_allow_html=True,
        )

    with right:
        if bad_items:
            priority_html = ""

            for i, (key, value, raw) in enumerate(bad_items[:3], start=1):
                fb = FEEDBACK[key]
                msg = fb["bad"].split("\n")[0]

                priority_html += f"""
<div style="padding:12px 0;border-bottom:1px solid #EEF2F6;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
        <div style="font-size:14px;font-weight:800;color:#172033;">
            {i}. {fb["label"]}
        </div>
        <span class="fit-badge badge-red">{value}</span>
    </div>
    <div style="font-size:12.5px;color:#667085;line-height:1.6;margin-top:5px;">
        {msg}
    </div>
</div>
"""

            guide_title = "오늘의 교정 우선순위"
            guide_badge = "집중관리"
        else:
            priority_html = """
<div style="font-size:14px;line-height:1.8;color:#667085;">
    현재 모든 주요 지표가 정상 범위에 있습니다.<br>
    지금 자세를 유지하면서 50분마다 가벼운 스트레칭을 해주세요.
</div>
"""
            guide_title = "오늘의 자세 상태"
            guide_badge = "양호"

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>{guide_title}</span>
        <span class="fit-badge badge-amber">{guide_badge}</span>
    </div>
    {priority_html}
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### AI 분석 요약")

    img_col, summary_col = st.columns([1, 1])

    with img_col:
        if "overlay" in result:
            st.markdown(
                """
    <div class="fit-card">
        <div class="fit-card-title">
            <span>최근 AI 오버레이</span>
            <span class="fit-badge badge-green">Analyzed</span>
        </div>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.image(result["overlay"], use_container_width=True)

    with summary_col:
        correction_html = build_ai_correction_comment(result)
        components.html(correction_html, height=650, scrolling=True)


def render_measure():
    page_header(
        "자세 측정",
        "측면 사진을 업로드하면 AI가 관절 각도와 작업환경 요소를 분석합니다.",
    )

    left, right = st.columns([0.95, 1.05])

    with left:
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>측면 사진 업로드</span>
        <span class="fit-badge badge-blue">F1</span>
    </div>
    <div style="font-size:13px;color:#667085;line-height:1.7;margin-bottom:12px;">
        의자, 책상, 모니터, 전신 측면이 최대한 함께 보이도록 촬영해주세요.
        발목·무릎·골반·어깨·귀가 보이면 분석 정확도가 좋아집니다.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "이미지 업로드",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="measure_uploader",
        )

        run_btn = st.button(
            "AI 자세 분석 실행",
            use_container_width=True,
            key="run_posture_analysis",
        )

    with right:
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="업로드된 측면 사진", use_container_width=True)
        else:
            st.markdown(
                """
<div class="upload-box">
    <div style="font-size:34px;margin-bottom:8px;">📸</div>
    <div style="font-weight:700;color:#172033;margin-bottom:4px;">측면 사진을 업로드하세요</div>
    <div style="font-size:13px;">AI 오버레이 분석 결과가 이 영역에 표시됩니다.</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if run_btn:
        if not uploaded:
            st.warning("먼저 이미지를 업로드해주세요.")
            return

        with st.spinner("AI가 자세를 분석하는 중입니다..."):
            result = analyze_image(Image.open(uploaded))

        if not result["ok"]:
            st.error(result["message"])
            return

        st.session_state.latest_result = result
        save_history(result)
        st.success("분석이 완료되었습니다.")

    result = st.session_state.get("latest_result")

    if result:
        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(result["score"], "종합 자세 점수", "#185FA5")
        with c2:
            metric_card(result["risk"], "위험도", "#BA7517")
        with c3:
            metric_card(f"{result['good_count']}/{result['total_count']}", "양호 지표", "#3B8C42")
        with c4:
            rate = round(result["good_count"] / result["total_count"] * 100)
            metric_card(f"{rate}%", "정상 범위 비율", "#6E56CF")

        render_measurement_coverage(result)

        img_col, panel_col = st.columns([1.05, 0.95])

        with img_col:
            st.markdown(
                """
<div class="fit-card">
    <div class="fit-card-title">
        <span>AI 오버레이 결과</span>
        <span class="fit-badge badge-green">F2</span>
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.image(result["overlay"], use_container_width=True)

        with panel_col:
            st.markdown(
                """
<div class="fit-card">
    <div class="fit-card-title">
        <span>7개 측정 지표</span>
        <span class="fit-badge badge-blue">F3</span>
    </div>
""",
                unsafe_allow_html=True,
            )
            render_result_rows({**result["posture"], **result["env"]})
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 맞춤 피드백")

        fb1, fb2 = st.columns(2)

        with fb1:
            st.markdown("#### 자세 지표")
            render_feedback_cards(result["posture"])

        with fb2:
            st.markdown("#### 작업환경 지표")
            render_feedback_cards(result["env"])


def render_history():
    page_header(
        "측정 이력",
        "최근 자세 분석 결과를 시간순으로 확인합니다.",
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    if not st.session_state.history:
        st.info("아직 측정 이력이 없습니다.")
        return

    latest = st.session_state.history[0]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(latest["score"], "최근 자세 점수", "#185FA5")
    with c2:
        metric_card(latest["risk"], "최근 위험도", "#BA7517")
    with c3:
        metric_card(f"{latest['good']}/{latest['total']}", "양호 지표", "#3B8C42")
    with c4:
        rate = round(latest["good"] / latest["total"] * 100)
        metric_card(f"{rate}%", "정상 비율", "#6E56CF")

    st.markdown("### 최근 측정 기록")

    for h in st.session_state.history:

        risk = h["risk"]

        if risk == "양호":
            color = "#3B8C42"
        elif risk == "주의":
            color = "#BA7517"
        else:
            color = "#D94A4A"

        rate = round(h["good"] / h["total"] * 100)
        missing_items = h.get("missing_items", []) or []
        if missing_items:
            coverage_note = f"측정 제외: {', '.join([item['label'] for item in missing_items])}"
        else:
            coverage_note = "8개 항목 전체 측정"

        st.markdown(
            f"""
<div class="fit-card" style="border-left:5px solid {color};">
<b>{h["time"]}</b><br><br>

종합 점수: <b>{h["score"]}/10</b><br>
위험도: <b>{risk}</b><br>
양호 지표: <b>{h["good"]}/{h["total"]}</b><br>
정상 비율: <b>{rate}%</b><br>
<span style="font-size:12px;color:#667085;">{coverage_note}</span>

<div style="margin-top:10px;height:8px;background:#EEF2F6;border-radius:999px;">
<div style="height:8px;width:{rate}%;background:{color};border-radius:999px;"></div>
</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if len(st.session_state.history) >= 2:
        st.markdown("### 점수 추이")
        scores = [h["score"] for h in reversed(st.session_state.history)]
        st.line_chart(scores)


# =========================================================
# 7-1. 자세 분석 결과 기반 비급여 예상 영수증
# =========================================================

NONPAY_CODES = {
    "경추": ["도수", "체외", "증식척추"],
    "요추": ["도수", "체외", "증식척추"],
    "손목": ["체외", "증식사지"],
}

NONPAY_INFO = {
    "도수": {"name": "🛏 도수치료", "avg": 107999},
    "체외": {"name": "⚡ 체외충격파", "avg": 91145},
    "증식척추": {"name": "💉 증식치료 (척추)", "avg": 93469},
    "증식사지": {"name": "💉 증식치료 (사지)", "avg": 90000},
}

PERIODS = [
    {"label": "1회 치료", "sessions": 1},
    {"label": "2주 (4회)", "sessions": 4},
    {"label": "1개월 (8회)", "sessions": 8},
    {"label": "3개월 (24회)", "sessions": 24},
    {"label": "6개월 (48회)", "sessions": 48},
]


def map_result_to_disease_locations(result):
    if result is None:
        return []
    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_keys = [key for key, (_, is_good, _) in all_data.items() if not is_good]
    active = []
    if any(key in bad_keys for key in ["CVA", "시선각"]):
        active.append("경추")
    if any(key in bad_keys for key in ["TIA", "등받이"]):
        active.append("요추")
    if any(key in bad_keys for key in ["손목", "팔꿈치", "책상높이"]):
        active.append("손목")
    return active


def build_receipt_html(active_d, period_label, t_dosu=True, t_shock=True, t_prolo=True, result=None):
    selected_period = next(p for p in PERIODS if p["label"] == period_label)
    sessions = selected_period["sessions"]

    used_np = []
    for d in active_d:
        for code in NONPAY_CODES.get(d, []):
            if code not in used_np:
                used_np.append(code)

    nonpay_total = 0
    np_rows_html = ""
    unit_html = ""
    for code in used_np:
        info = NONPAY_INFO[code]
        is_active = (code == "도수" and t_dosu) or (code == "체외" and t_shock) or (code.startswith("증식") and t_prolo)
        if not is_active:
            continue
        total = info["avg"] * sessions
        nonpay_total += total
        np_rows_html += f"""
        <div class='r-row non'><span>{info['name']}</span><span>×{sessions}회</span><b>{total:,}원</b></div>
        """
        unit_html += f"""
        <div class='r-row sub'><span>{info['name']}</span><span>1회</span><span>{info['avg']:,}원</span></div>
        """

    if not np_rows_html:
        np_rows_html = "<div class='r-row'><span>현재 자동 청구 예상 항목 없음</span><span>-</span><b>0원</b></div>"
        unit_html = "<div class='r-row sub'><span>정상 범위 유지 시 예방 관리 권장</span><span>-</span><span>0원</span></div>"

    warn_msgs = [
        (0, "⚠ 이 자세를 계속 유지하면 위 비용이 발생할 수 있습니다", "지금 자세를 교정하세요"),
        (500000, "💸 월급의 상당 부분이 병원비로 사라질 수 있습니다", "만성 통증으로 이어지기 전에 예방하세요"),
        (1500000, "🚨 해외여행 경비가 통째로 날아갈 수 있습니다", "치료보다 예방이 훨씬 저렴합니다"),
        (3000000, "🔴 분기 의료비가 차 한 대 값에 육박할 수 있습니다", "이제 자세 교정이 투자입니다"),
        (6000000, "☠️ 연봉의 상당 부분을 병원에 내야 할 수 있습니다", "지금 당장 작업환경을 바꾸세요"),
    ]
    wm = warn_msgs[0]
    for msg in reversed(warn_msgs):
        if nonpay_total >= msg[0]:
            wm = msg
            break

    now = datetime.datetime.now()
    dt_str = f"{now.year}.{now.month:02d}.{now.day:02d}  {now.hour:02d}:{now.minute:02d}"
    disease_str = " · ".join(active_d) + " 질환" if active_d else "관리 필요 질환 없음"
    barcode_num = f"FITMEUP-VDT-{str(nonpay_total).zfill(9)}"
    score_line = ""
    if result is not None:
        score_line = f"자세점수 : {result.get('score', '-')} / 10<br>위험도 : {result.get('risk', '-')}<br>"

    return f"""
<!DOCTYPE html><html lang='ko'><head><meta charset='UTF-8'>
<link href='https://fonts.googleapis.com/css2?family=Nanum+Gothic+Coding:wght@400;700&family=Pretendard:wght@300;400;500;600;700;900&display=swap' rel='stylesheet'>
<style>
:root {{ --paper:#fefcf6; --ink:#111; --red:#c0392b; --mono:'Nanum Gothic Coding',monospace; }}
body {{ margin:0; padding:10px; background:transparent; display:flex; justify-content:center; }}
.receipt-outer {{ width:100%; max-width:430px; filter:drop-shadow(0 4px 16px rgba(0,0,0,.15)); }}
.zig-top {{ height:18px; background:var(--paper); clip-path:polygon(0% 100%,4% 0%,8% 100%,12% 0%,16% 100%,20% 0%,24% 100%,28% 0%,32% 100%,36% 0%,40% 100%,44% 0%,48% 100%,52% 0%,56% 100%,60% 0%,64% 100%,68% 0%,72% 100%,76% 0%,80% 100%,84% 0%,88% 100%,92% 0%,96% 100%,100% 0%); }}
.zig-bot {{ height:18px; background:var(--paper); clip-path:polygon(0% 0%,4% 100%,8% 0%,12% 100%,16% 0%,20% 100%,24% 0%,28% 100%,32% 0%,36% 100%,40% 0%,44% 100%,48% 0%,52% 100%,56% 0%,60% 100%,64% 0%,68% 100%,72% 0%,76% 100%,80% 0%,84% 100%,88% 0%,92% 100%,96% 0%,100% 100%); }}
.body {{ background:var(--paper); padding:8px 24px 22px; font-family:var(--mono); color:var(--ink); }}
.center {{ text-align:center; }} .store {{ font-size:16px; font-weight:700; letter-spacing:3px; }} .subt {{ font-size:10px; color:#888; letter-spacing:2px; }}
.warn {{ font-size:11px; font-weight:700; color:var(--red); border:2px solid var(--red); padding:4px 8px; display:inline-block; margin:8px 0 4px; }}
.dash {{ color:#bbb; font-size:11px; white-space:nowrap; overflow:hidden; }} .meta {{ font-size:10px; color:#777; line-height:1.9; margin:8px 0; }}
.hd {{ font-size:10px; font-weight:700; color:#888; letter-spacing:1px; margin:8px 0 4px; }}
.r-row {{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; font-size:11.5px; margin-bottom:4px; }}
.r-row span:first-child {{ flex:1; color:#444; }} .r-row span:nth-child(2) {{ color:#999; font-size:10px; white-space:nowrap; }} .r-row b {{ color:var(--red); white-space:nowrap; }}
.sub {{ font-size:10px; color:#999; }} .sgl {{ border-top:1px dashed #ccc; margin:7px 0; }} .dbl {{ border-top:2px solid #111; margin:8px 0 4px; }}
.total {{ font-size:15px; font-weight:700; color:var(--red); }} .barcode {{ font-size:36px; line-height:.8; letter-spacing:-1px; opacity:.85; }} .bcnum {{ font-size:9px; letter-spacing:3px; color:#999; margin-top:4px; }}
.notice {{ font-size:9px; color:#aaa; line-height:1.8; margin-top:12px; }} .notice p {{ margin:0; }} .notice p:before {{ content:'* '; }}
</style></head><body><div class='receipt-outer'><div class='zig-top'></div><div class='body'>
<div class='center' style='padding:14px 0 8px'><div class='store'>비급여 의료비 예상 청구서</div><div class='subt'>POSTURE LINKED MEDICAL COST</div><span class='warn'>⚠ 경 고 ⚠</span><div class='dash'>────────────────────────</div><div style='font-size:11px;color:#555;margin-top:4px'>{disease_str}</div></div>
<div class='sgl'></div><div class='meta'>발행일시 : {dt_str}<br>{score_line}치료기간 : {selected_period['label']}<br>치료방식 : 주 2회 집중 치료 기준<br>자동연동 : 자세측정 BAD 항목 기반</div>
<div class='sgl'></div><div class='hd'>[ 비급여 항목 · 전액 본인부담 ]</div>{np_rows_html}
<div class='sgl'></div><div class='r-row'><span>비급여 소계</span><span></span><b>{nonpay_total:,}원</b></div><div class='dbl'></div><div class='r-row total'><span>TOTAL</span><b>{nonpay_total:,}원</b></div>
<div class='sgl'></div><div class='hd'>[ 1회 단가 참고 ]</div>{unit_html}
<div class='sgl'></div><div class='center' style='margin:10px 0'><div style='font-size:10px;color:#c0392b;font-weight:700'>{wm[1]}</div><div style='font-size:9px;color:#999;margin-top:4px'>{wm[2]}</div></div>
<div class='sgl'></div><div class='center'><div class='barcode'>▌▌ ▌▌▌ ▌ ▌▌▌▌ ▌ ▌▌ ▌▌▌ ▌▌</div><div class='bcnum'>{barcode_num}</div></div>
<div class='notice'><p>본 청구서는 예상 비용이며 실제 금액과 다를 수 있습니다</p><p>자세 분석 BAD 항목을 경추·요추·손목 질환 위치로 자동 매핑했습니다</p><p>비급여 금액은 앱 내 평균 단가 기준입니다</p><p>자세 분석은 전문 의료 진단을 대체하지 않습니다</p></div>
</div><div class='zig-bot'></div></div></body></html>
"""

def minutes_until_next_alarm(selected_times):
    now = datetime.datetime.now()
    candidates = []

    for t in selected_times:
        hour, minute = map(int, t.split(":"))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if target <= now:
            target += datetime.timedelta(days=1)

        candidates.append(target)

    if not candidates:
        return None, None

    next_time = min(candidates)
    diff_min = int((next_time - now).total_seconds() // 60)

    return next_time, diff_min

def render_alarm_effect(selected_times):
    now = datetime.datetime.now().strftime("%H:%M")

    dismissed_param = st.query_params.get("alarm_dismissed", "")
    if dismissed_param == now:
        st.session_state.alarm_dismissed_time = now
        st.query_params.clear()
        st.rerun()

    if now not in selected_times:
        return

    if st.session_state.get("alarm_dismissed_time", "") == now:
        return

    components.html(
        f"""
<script>
const alarmTime = "{now}";

function removeOldAlarm() {{
    const old = window.parent.document.getElementById("fitmeupAlarmOverlay");
    if (old) old.remove();

    const oldStyle = window.parent.document.getElementById("fitmeupAlarmStyle");
    if (oldStyle) oldStyle.remove();
}}

function closeAlarm() {{
    const overlay = window.parent.document.getElementById("fitmeupAlarmOverlay");
    if (overlay) overlay.remove();

    const params = new URLSearchParams(window.parent.location.search);
    params.set("alarm_dismissed", alarmTime);
    window.parent.location.search = params.toString();
}}

removeOldAlarm();

const style = window.parent.document.createElement("style");
style.id = "fitmeupAlarmStyle";
style.innerHTML = `
#fitmeupAlarmOverlay {{
    position: fixed;
    inset: 0;
    z-index: 2147483647;
    background: rgba(15, 23, 42, 0.38);
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fitAlarmBg 0.75s infinite alternate;
}}

#fitmeupAlarmModal {{
    position: relative;
    width: 560px;
    max-width: 84vw;
    padding: 48px 38px;
    border-radius: 30px;
    background: #FCEBEB;
    border: 4px solid #D94A4A;
    box-shadow: 0 24px 90px rgba(217, 74, 74, 0.45);
    text-align: center;
    font-family: Pretendard, sans-serif;
    animation: fitAlarmPulse 0.75s infinite alternate;
}}

#fitmeupAlarmClose {{
    position: absolute;
    top: 16px;
    right: 20px;
    border: none;
    background: transparent;
    color: #D94A4A;
    font-size: 30px;
    font-weight: 900;
    cursor: pointer;
}}

.fitmeup-alarm-icon {{
    font-size: 64px;
    margin-bottom: 14px;
}}

.fitmeup-alarm-title {{
    font-size: 36px;
    font-weight: 900;
    color: #D94A4A;
    line-height: 1.35;
}}

.fitmeup-alarm-sub {{
    margin-top: 14px;
    font-size: 18px;
    font-weight: 700;
    color: #8E2424;
}}

@keyframes fitAlarmBg {{
    from {{ background: rgba(15, 23, 42, 0.30); }}
    to {{ background: rgba(217, 74, 74, 0.42); }}
}}

@keyframes fitAlarmPulse {{
    from {{ transform: scale(1); opacity: 1; }}
    to {{ transform: scale(1.04); opacity: 0.86; }}
}}
`;
window.parent.document.head.appendChild(style);

const overlay = window.parent.document.createElement("div");
overlay.id = "fitmeupAlarmOverlay";
overlay.innerHTML = `
    <div id="fitmeupAlarmModal">
        <button id="fitmeupAlarmClose">✕</button>
        <div class="fitmeup-alarm-icon">🔔</div>
        <div class="fitmeup-alarm-title">바른자세 체크 시간입니다!</div>
        <div class="fitmeup-alarm-sub">지금 자세를 확인하고 바로 측정해보세요.</div>
    </div>
`;
window.parent.document.body.appendChild(overlay);

window.parent.document
    .getElementById("fitmeupAlarmClose")
    .addEventListener("click", closeAlarm);

window.parent.document.addEventListener("keydown", function(e) {{
    if (e.key === "Escape") {{
        closeAlarm();
    }}
}});
</script>

<audio autoplay>
    <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
</audio>
""",
        height=1,
    )

def render_posture_challenge():
    st_autorefresh(interval=30000, key="challenge_autorefresh")

    st.markdown("## 바른자세 챌린지")
    st.caption("원하는 시간 설정 → 알림 수신 → 즉시 측정 → 팀 피드 자동 공유")

    if "challenge_times" not in st.session_state:
        st.session_state.challenge_times = []

    if "challenge_members" not in st.session_state:
        st.session_state.challenge_members = []

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### 알림 시간 직접 설정")
        st.caption("원하는 알림 시간을 시/분 단위로 추가하세요.")

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            alarm_hour = st.selectbox(
            "시",
            options=list(range(0, 24)),
            format_func=lambda x: f"{x:02d}시",
            key="challenge_alarm_hour",
            )

        with c2:
            alarm_minute = st.selectbox(
            "분",
            options=list(range(0, 60)),
            format_func=lambda x: f"{x:02d}분",
            key="challenge_alarm_minute",
            )

        with c3:
            st.write("")
            st.write("")
            add_alarm = st.button(
            "알림 추가",
            use_container_width=True,
            key="challenge_add_alarm",
            )

        if add_alarm:
            new_time = f"{alarm_hour:02d}:{alarm_minute:02d}"

            if new_time not in st.session_state.challenge_times:
                st.session_state.challenge_times.append(new_time)
                st.success(f"{new_time} 알림이 추가되었습니다.")
            else:
                st.warning("이미 추가된 알림 시간입니다.")

        if st.session_state.challenge_times:
            st.markdown("#### 설정된 알림 시간")

            for t in sorted(st.session_state.challenge_times):
                c_time, c_del = st.columns([4, 1])

                with c_time:
                    st.markdown(f"**{t}**")

                with c_del:
                    if st.button("삭제", key=f"delete_alarm_{t}"):
                        st.session_state.challenge_times.remove(t)
                        st.rerun()

        if st.session_state.challenge_times:
            next_time, diff_min = minutes_until_next_alarm(
            st.session_state.challenge_times
            )

            if next_time:
                hours = diff_min // 60
                mins = diff_min % 60
                remain_text = f"{hours}시간 {mins}분 후" if hours > 0 else f"{mins}분 후"

                st.markdown(
                    f"""
    <div class="fit-card" style="background:#E6F1FB;">
        <div style="font-size:13px;color:#185FA5;font-weight:700;">다음 알림</div>
        <div style="font-size:20px;font-weight:850;color:#0C447C;margin-top:6px;">
            {next_time.strftime('%H:%M')}
        </div>
        <div style="font-size:13px;color:#185FA5;margin-top:4px;">
            현재 시간 기준 <b>{remain_text}</b>
        </div>
    </div>
    """,
                unsafe_allow_html=True,
            )
        else:
            st.info("아직 알림 시간이 설정되지 않았습니다.")

    with right:
        st.markdown("### 4팀 척추처척추")

        with st.form("challenge_member_form", clear_on_submit=True):
            name = st.text_input("이름 입력")

            score = st.number_input(
                "자세 점수",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
            )

            comment = st.text_input("응원 문구", value="좋아요!")

            submitted = st.form_submit_button("팀원 추가", use_container_width=True)

            if submitted:
                if name.strip():
                    st.session_state.challenge_members.append(
                        {
                            "name": name.strip(),
                            "score": round(score, 1),
                            "comment": comment.strip() if comment.strip() else "좋아요!",
                            "time": datetime.datetime.now().strftime("%H:%M"),
                        }
                    )
                    st.success(f"{name.strip()} 님이 추가되었습니다.")
                else:
                    st.warning("이름을 입력해주세요.")

        if not st.session_state.challenge_members:
            st.info("아직 추가된 팀원이 없습니다.")
            return

        members = sorted(
            st.session_state.challenge_members,
            key=lambda x: x["score"],
            reverse=True,
        )

        for i, member in enumerate(members, start=1):
            score = member["score"]

            if score >= 7:
                status = "양호"
            elif score >= 4:
                status = "주의"
            else:
                status = "위험"

            st.markdown(f"#### {i}. {member['name']}")
            st.caption(member["time"])

            st.progress(score / 10)

            c1, c2, c3 = st.columns([1, 1, 1])

            with c1:
                st.metric("자세 점수", f"{score}")

            with c2:
                st.metric("상태", status)

            with c3:
                st.write(f"응원 · {member['comment']}")

            st.divider()

def render_receipt_page():
    page_header("비급여 의료비 예상 영수증", "자세측정 결과에서 기준 범위를 벗어난 부위를 경추·요추·손목 항목으로 자동 연결합니다.")
    result = st.session_state.get("latest_result")
    if result is None:
        st.info("영수증을 생성하려면 먼저 왼쪽 메뉴의 자세측정에서 AI 자세 분석을 실행해주세요.")
        return

    auto_diseases = map_result_to_disease_locations(result)
    all_data = {**result["posture"], **result["env"]}
    bad_labels = [FEEDBACK[k]["label"] for k, v in all_data.items() if not v[1]]

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown(f"""
<div class='fit-card'><div class='fit-card-title'><span>자세 분석 자동 매핑</span><span class='fit-badge badge-blue'>Auto</span></div>
<div style='font-size:13px;line-height:1.8;color:#667085;'><b style='color:#172033;'>BAD 측정 항목</b><br>{' · '.join(bad_labels) if bad_labels else '현재 BAD 항목 없음'}<br><br><b style='color:#172033;'>영수증 반영 위치</b><br>{' · '.join(auto_diseases) if auto_diseases else '관리 필요 질환 없음'}</div></div>
""", unsafe_allow_html=True)
        st.subheader("📍 질환 위치")
        st.caption("자세 분석 결과에 따라 기본값이 자동 선택됩니다. 필요하면 직접 수정할 수 있습니다.")
        c1, c2, c3 = st.columns(3)
        with c1:
            d_neck = st.checkbox("경추", value=("경추" in auto_diseases), key="receipt_neck")
        with c2:
            d_waist = st.checkbox("요추", value=("요추" in auto_diseases), key="receipt_waist")
        with c3:
            d_wrist = st.checkbox("손목", value=("손목" in auto_diseases), key="receipt_wrist")
        st.divider()
        st.subheader("💉 비급여 치료 선택")
        t_dosu = st.checkbox("🛏 도수치료", value=True, key="receipt_dosu")
        t_shock = st.checkbox("⚡ 체외충격파", value=True, key="receipt_shock")
        t_prolo = st.checkbox("💉 증식치료", value=True, key="receipt_prolo")
        st.divider()
        st.subheader("⏱ 치료 기간")
        default_period = "3개월 (24회)" if result.get("risk") == "위험" else ("1회 치료" if result.get("risk") == "양호" else "1개월 (8회)")
        period_label = st.select_slider("치료 기간", options=[p["label"] for p in PERIODS], value=default_period, label_visibility="collapsed", key="receipt_period")

    active_d = []
    if d_neck:
        active_d.append("경추")
    if d_waist:
        active_d.append("요추")
    if d_wrist:
        active_d.append("손목")

    receipt_html = build_receipt_html(active_d, period_label, t_dosu, t_shock, t_prolo, result)
    with right:
        components.html(receipt_html, height=880, scrolling=True)


def render_report():
    page_header(
        "근골격계 리포트",
        "자세 측정 결과를 바탕으로 부위별 위험도와 교정 우선순위를 정리합니다.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("리포트를 생성하려면 먼저 자세 측정을 실행해주세요.")
        return

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            f"{round(result['good_count'] / result['total_count'] * 100)}%",
            "양호 판정 비율",
            "#185FA5",
        )
    with c2:
        metric_card(result["score"], "종합 점수", "#3B8C42")
    with c3:
        bad_count = result["total_count"] - result["good_count"]
        metric_card(f"{bad_count}개", "관리 필요 지표", "#BA7517")
    with c4:
        metric_card(result["risk"], "종합 위험도", "#D94A4A")

    left, right = st.columns(2)

    all_data = {**result["posture"], **result["env"]}

    with left:
        st.markdown(
            """
<div class="fit-card">
    <div class="fit-card-title">
        <span>부위별 측정 지표</span>
        <span class="fit-badge badge-blue">Report</span>
    </div>
""",
            unsafe_allow_html=True,
        )
        render_result_rows(all_data)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        bad_items = [k for k, v in all_data.items() if not v[1]]
        priority = bad_items[:3] if bad_items else ["현재 우선 교정 항목 없음"]

        st.markdown(
            f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>교정 우선순위</span>
        <span class="fit-badge badge-amber">Top Priority</span>
    </div>
    <div style="font-size:13px;line-height:1.9;color:#667085;">
        <b style="color:#172033;">1순위:</b> {priority[0] if len(priority) > 0 else "-"}<br>
        <b style="color:#172033;">2순위:</b> {priority[1] if len(priority) > 1 else "-"}<br>
        <b style="color:#172033;">3순위:</b> {priority[2] if len(priority) > 2 else "-"}<br><br>
        <b style="color:#172033;">권장 행동</b><br>
        모니터 높이, 의자 깊이, 팔꿈치 높이, 손목 중립 상태를 우선 조정하세요.
        장시간 작업자는 50분 작업 후 5~10분 스트레칭을 권장합니다.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### 상세 피드백")
    render_feedback_cards(all_data)


# =========================================================
# 8. 실제 페이지 출력부
# =========================================================

init_history()

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
page_placeholder = st.empty()

st_autorefresh(interval=10000, key="global_alarm_refresh")

if "challenge_times" not in st.session_state:
    st.session_state.challenge_times = []

render_alarm_effect(st.session_state.challenge_times)

with page_placeholder.container():
    if menu == "🏠 대시보드":
        render_dashboard()

    elif menu == "📸 자세측정":
        render_measure()

    elif menu == "📈 측정이력":
        render_history()

    elif menu == "📄 근골격계 리포트":
        render_report()

    elif menu == "🧾 예상 영수증":
        render_receipt_page()

    elif menu == "🎯 바른자세 챌린지":
        render_posture_challenge()
