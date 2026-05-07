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

import json
import hashlib
import base64

import pandas as pd
import altair as alt

from io import BytesIO

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# =========================================================
# 1. 기본 설정
# =========================================================

st.set_page_config(
    page_title="Fit Me Up — AI 자세 분석",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 경로 (분석 로직은 integrate.py에서 관리)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
        "range": "정상 0°~20° · 주의 20°초과~28° · 위험 28°초과",
        "cat": "posture",
        "good": "머리·경추 수직 정렬 유지\n경추 부담 최소화 상태",
        "bad": "전방두부자세(FHP) 의심\n모니터를 눈높이로 올리세요\n1시간마다 목 스트레칭 시행",
    },
    "TIA": {
        "no": "02",
        "label": "몸통굴곡각",
        "eng": "TIA",
        "range": "정상 0°~10° · 주의 10°초과~20° · 위험 20°초과",
        "cat": "posture",
        "good": "척추 수직 정렬 양호\n요추 압박 최소화 상태",
        "bad": "과도한 몸통 전굴 감지\n등받이에 허리 완전 밀착\n의자 깊숙이 앉으세요",
    },
    "팔꿈치": {
        "no": "03",
        "label": "팔꿈치 각도",
        "eng": "Elbow",
        "range": "정상 90°~120° · 위험 90° 미만 또는 120° 초과",
        "cat": "posture",
        "good": "상지 관절 부하 최적 범위\n팔꿈치·책상면 수평 유지",
        "bad": "팔꿈치 각도 기준 이탈\n의자 높이 조정 필요\n팔꿈치·책상면 수평 유지",
    },
    "무릎": {
        "no": "04",
        "label": "무릎 각도",
        "eng": "Knee",
        "range": "정상 85°~100° · 위험 85° 미만 또는 100° 초과",
        "cat": "posture",
        "good": "하지 혈액순환 원활\n하체 부담 최소화 상태",
        "bad": "무릎 각도 기준 이탈\n의자 높이 조절 필요\n발받침대 사용 권장",
    },
    "손목": {
        "no": "05",
        "label": "손목 각도",
        "eng": "Wrist",
        "range": "정상 165°~180° · 위험 165° 미만",
        "cat": "posture",
        "good": "손목 중립 자세 유지\n손목 터널 부담 최소화",
        "bad": "손목 과굴곡 감지\n손목 받침대 설치 필요\n키보드 앞 15cm 확보",
    },
    "시선각": {
        "no": "06",
        "label": "모니터 시선각",
        "eng": "Gaze",
        "range": "정상 하방 10°~15° · 위험 10° 미만 또는 15° 초과",
        "cat": "env",
        "good": "시선각 기준 충족\n경추 부담 최소화",
        "bad": "시선각 기준 이탈\n모니터 상단을 눈높이에 맞추세요\n화면 거리 40cm 이상 권장",
    },
    "책상높이": {
        "no": "07",
        "label": "작업대 높이",
        "eng": "Desk",
        "range": "정상 ±5% 이내 · 위험 ±5% 초과",
        "cat": "env",
        "good": "작업대·팔꿈치 정렬 양호\n상지 부담 최소화",
        "bad": "작업대 높이 불일치\n책상 높이 또는 의자 높이 조정 필요",
    },
    "등받이": {
        "no": "08",
        "label": "의자 등받이",
        "eng": "Chair",
        "range": "정상 20% 이내 · 위험 20% 초과",
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

/* Streamlit 기본 여백 조정 */
.block-container {
    padding-top: 4.5rem !important;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

/* 사이드바 라디오 */
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

/* 로고 */
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

/* 제목 */
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

/* 카드 */
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

/* 메트릭 */
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

/* 결과 행 */
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

/* 피드백 카드 */
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

/* 업로드 영역 */
.upload-box {
    border: 1.5px dashed #CBD5E1;
    background: #F8FAFC;
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    color: var(--sub);
}

/* 표 */
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

/* 모바일 */
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
# 로그인 / 회원가입 기능
# =========================================================

USER_DB_PATH = "users.json"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if not os.path.exists(USER_DB_PATH):
        return {}
    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_users(users):
    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)




def make_logo_transparent(filename="logo.png"):
    """
    app.py와 같은 폴더의 logo.png를 읽어서
    흰색/밝은 배경을 투명하게 만든 logo_transparent.png를 생성합니다.
    logo.png가 없어도 앱이 중단되지 않도록 None을 반환합니다.
    """
    try:
        base_dir = Path(__file__).resolve().parent
    except Exception:
        base_dir = Path.cwd()

    src = base_dir / filename
    out = base_dir / "logo_transparent.png"

    if not src.exists():
        return None

    try:
        img = Image.open(src).convert("RGBA")
        pixels = []

        for r, g, b, a in img.getdata():
            # 흰색 또는 거의 흰색 배경을 투명 처리
            if r >= 235 and g >= 235 and b >= 235:
                pixels.append((255, 255, 255, 0))
            else:
                pixels.append((r, g, b, a))

        img.putdata(pixels)
        img.save(out, "PNG")
        return out

    except Exception:
        return src


def render_auth_page():
    """
    로그인/회원가입 화면
    - 로고: logo.png 흰 배경 자동 투명 처리
    - 로그인/회원가입 선택 탭: 로고 좌측선에 맞춤
    - 입력창/버튼: 로고보다 살짝 크게 중앙 정렬
    - HTML img 태그를 사용하지 않아 Python 문법 오류 방지
    """

    st.markdown(
        """
<style>
/* 로그인 페이지 상단 여백 */
.block-container {
    padding-top: 2.2rem !important;
}

/* 로그인 화면 전체 폭 */
.auth-guide-text {
    text-align: left;
    color: #667085;
    font-size: 13px;
    margin-top: -4px;
    margin-bottom: 16px;
    padding-left: 0px;
}

.auth-section-title {
    text-align: center;
    font-size: 20px;
    font-weight: 850;
    color: #172033;
    margin: 10px 0 14px 0;
}

/* 로그인/회원가입 선택 영역: 로고 좌측 시작선과 맞춤 */
div[data-testid="stRadio"] {
    width: 380px !important;
    max-width: 380px !important;
    margin-left: 0 !important;
    margin-right: auto !important;
    margin-bottom: 12px !important;
}

div[data-testid="stRadio"] > div {
    justify-content: flex-start !important;
    gap: 8px !important;
}

/* 라디오 버튼 자체를 탭처럼 */
div[data-testid="stRadio"] label {
    background: #F2F4F7 !important;
    border-radius: 10px !important;
    padding: 8px 14px !important;
    margin-right: 6px !important;
}

/* 입력창과 버튼: 로고보다 살짝 큰 폭 */
div[data-testid="stTextInput"] {
    max-width: 430px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

div[data-testid="stButton"] {
    max-width: 430px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

div[data-testid="stButton"] button {
    height: 44px !important;
    border-radius: 12px !important;
    font-weight: 800 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    logo_path = make_logo_transparent("logo.png")

    # 가운데 정렬용 컬럼
    left, center, right = st.columns([1, 1.28, 1])

    with center:
        # 로고는 380px, 입력창은 CSS로 430px → 입력창이 로고보다 살짝 큼
        if logo_path is not None:
            st.image(str(logo_path), width=380)
        else:
            st.markdown(
                """
<div style="width:380px;text-align:left;font-size:34px;font-weight:900;color:#172033;margin-top:20px;">
    Fit Me Up
</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="auth-guide-text">로그인 후 AI 자세 분석 서비스를 이용하세요.</div>',
            unsafe_allow_html=True,
        )

        auth_tab = st.radio(
            "auth",
            ["로그인", "회원가입"],
            horizontal=True,
            label_visibility="collapsed",
        )

        users = load_users()

        if auth_tab == "로그인":
            st.markdown(
                '<div class="auth-section-title">로그인</div>',
                unsafe_allow_html=True,
            )

            username = st.text_input("아이디", key="login_id")
            password = st.text_input("비밀번호", type="password", key="login_pw")

            if st.button("로그인", use_container_width=True):
                if username not in users:
                    st.error("존재하지 않는 아이디입니다.")
                    return

                if users[username]["password"] != hash_password(password):
                    st.error("비밀번호가 일치하지 않습니다.")
                    return

                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"{username}님, 로그인되었습니다.")
                st.rerun()

        else:
            st.markdown(
                '<div class="auth-section-title">회원가입</div>',
                unsafe_allow_html=True,
            )

            new_username = st.text_input("아이디", key="signup_id")
            new_password = st.text_input("비밀번호", type="password", key="signup_pw")
            new_password_check = st.text_input("비밀번호 확인", type="password", key="signup_pw_check")

            if st.button("회원가입", use_container_width=True):
                if not new_username or not new_password:
                    st.error("아이디와 비밀번호를 입력해주세요.")
                    return

                if new_username in users:
                    st.error("이미 존재하는 아이디입니다.")
                    return

                if new_password != new_password_check:
                    st.error("비밀번호가 일치하지 않습니다.")
                    return

                users[new_username] = {
                    "password": hash_password(new_password),
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

                save_users(users)
                st.success("회원가입이 완료되었습니다. 로그인해주세요.")


# =========================================================
# 3. 모델 로드
# =========================================================

@st.cache_resource(show_spinner=False)

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    try:
        from ultralytics import YOLO
        return YOLO(model_path)
    except Exception:
        return None


def analyze_image(pil_image: Image.Image):
    """
    integrate.py의 run_pipeline() 기반 분석
    반환값: app.py 기존 형식과 호환
    """
    import tempfile, os as _os

    # PIL → 임시 파일
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        pil_image.save(tmp_path, quality=95)

    try:
        # integrate.py import
        sys.path.insert(0, BASE_DIR)
        from integrate import run_pipeline, INDICATOR_NAMES, IND_UNITS, FEEDBACK as INT_FEEDBACK

        mlp_res, all_ind, overlay_bgr, early_stop = run_pipeline(tmp_path)
    except Exception as e:
        return {"ok": False, "message": f"분석 오류: {str(e)}"}
    finally:
        try: _os.unlink(tmp_path)
        except: pass

    if mlp_res is None or isinstance(early_stop, str):
        return {"ok": False, "message": early_stop or "관절 탐지 실패"}

    # ── all_ind → app.py posture/env 형식 변환 ────────────────────────
    # integrate.py: {key: {'value', 'ok'}}
    # app.py:       {key: (값문자열, is_good, raw)}
    KEY_MAP = {
        'cva':     ('CVA',    '°'),
        'tia':     ('TIA',    '°'),
        'elbow':   ('팔꿈치', '°'),
        'knee':    ('무릎',   '°'),
        'wrist':   ('손목',   '°'),
        'gaze':    ('시선각', '°'),
        'desk_h':  ('책상높이', ''),
        'chair_d': ('등받이', ''),
    }
    POSTURE_KEYS = ['cva','tia','elbow','knee','wrist']
    ENV_KEYS     = ['gaze','desk_h','chair_d']

    def to_tuple(key):
        r   = all_ind.get(key, {})
        v   = r.get('value')
        ok  = r.get('ok')
        unit = KEY_MAP[key][1]
        val_str = f"{v}{unit}" if v is not None else "인식 불가"
        # is_good: ok=True→True, ok=False→False, ok=None→False(N/A)
        is_good = (ok is True)
        return (val_str, is_good, v)

    posture = {KEY_MAP[k][0]: to_tuple(k) for k in POSTURE_KEYS if k in all_ind}
    env     = {KEY_MAP[k][0]: to_tuple(k) for k in ENV_KEYS     if k in all_ind}

    # ── CVA/TIA 사전 품질검사 ────────────────────────────────
    # integrate.py의 CVA/TIA ok 값만 사용해서 업로드 사진을 통과/재촬영으로 나눕니다.
    # 둘 다 True일 때만 app.py의 기존 8개 지표 결과 화면을 보여주고,
    # 하나라도 False 또는 측정불가(None)이면 기존 결과 저장/렌더링을 막습니다.
    cva_info = all_ind.get('cva', {}) or {}
    tia_info = all_ind.get('tia', {}) or {}
    cva_gate_ok = cva_info.get('ok') is True
    tia_gate_ok = tia_info.get('ok') is True
    posture_gate_pass = cva_gate_ok and tia_gate_ok

    def _gate_value_text(info, unit='°'):
        v = info.get('value')
        return f"{v}{unit}" if v is not None else "측정불가"

    cva_gate_value = _gate_value_text(cva_info)
    tia_gate_value = _gate_value_text(tia_info)

    cva_gate_status = "GOOD" if cva_gate_ok else "BAD"
    tia_gate_status = "GOOD" if tia_gate_ok else "BAD"

    if not posture_gate_pass:
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        bad_gate_items = []
        if not cva_gate_ok:
            bad_gate_items.append(f"CVA 목굴곡각: {cva_gate_value} / {cva_gate_status}")
        if not tia_gate_ok:
            bad_gate_items.append(f"TIA 몸통굴곡각: {tia_gate_value} / {tia_gate_status}")

        return {
            "ok": True,
            "message": "CVA/TIA 자세 품질검사에서 재촬영이 필요합니다.",
            "gate_pass": False,
            "gate_reason": "CVA와 TIA가 둘 다 GOOD일 때만 종합 GOOD으로 판정됩니다.",
            "gate_bad_items": bad_gate_items,
            "gate_metrics": {
                "CVA": {"value": cva_gate_value, "status": cva_gate_status},
                "TIA": {"value": tia_gate_value, "status": tia_gate_status},
            },
            "overlay": overlay_rgb,
            "posture": posture,
            "env": env,
        }

    # classify_posture_level 적용 (app.py 기준 재분류)
    posture = {
        key: (value, classify_posture_level(key, raw) == "정상", raw)
        for key, (value, _, raw) in posture.items()
    }
    env = {
        key: (value, classify_posture_level(key, raw) == "정상", raw)
        for key, (value, _, raw) in env.items()
    }

    # overlay BGR→RGB
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # missing items
    all_items = {**posture, **env}
    missing_reason_map = {
        "시선각":   "모니터가 인식되지 않아 시선각 평가 제외",
        "책상높이": "책상이 인식되지 않아 책상 높이 평가 제외",
        "등받이":   "의자 등받이가 인식되지 않아 등받이 지지 평가 제외",
    }
    missing_items = [
        {"key": k, "label": FEEDBACK[k]["label"],
         "reason": missing_reason_map.get(k, "기준점 부족으로 평가 제외")}
        for k, v in all_items.items() if v[0] == "인식 불가"
    ]

    score, risk, level_counts = calculate_clinical_score_from_items(posture, env)
    good_count  = level_counts["정상"]
    total_count = level_counts["정상"] + level_counts["주의"] + level_counts["위험"]

    return {
        "ok":            True,
        "message":       "분석 완료",
        "overlay":       overlay_rgb,
        "posture":       posture,
        "env":           env,
        "score":         score,
        "risk":          risk,
        "good_count":    good_count,
        "total_count":   total_count,
        "missing_items": missing_items,
        "level_counts":  level_counts,
    }





def get_logo_base64(filename="logo.png"):
    logo_path = make_logo_transparent(filename)

    if logo_path is None:
        return ""

    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

def render_logo():
    logo_base64 = get_logo_base64("logo.png")

    if logo_base64:
        logo_html = (
            '<div class="sidebar-logo-wrap">'
            f'<img class="sidebar-logo-img" src="data:image/png;base64,{logo_base64}">'
            '<div class="sidebar-logo-text">AI 자세 분석 서비스</div>'
            '</div>'
        )
    else:
        logo_html = (
            '<div class="sidebar-logo-fallback-wrap">'
            '<div class="sidebar-logo-fallback">F</div>'
            '<div class="sidebar-logo-text">AI 자세 분석 서비스</div>'
            '</div>'
        )

    st.sidebar.markdown(
        f"""
<style>
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0 !important;
    margin-top: -54px !important;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 0 !important;
}}

.sidebar-header {{
    width: 100%;
    padding: 0 0 10px 0;
    border-bottom: 1px solid #E5EAF2;
    margin-bottom: 6px;
    text-align: center;
}}

.sidebar-logo-wrap {{
    position: relative;
    width: 100%;
    height: 245px;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    overflow: hidden;
}}

.sidebar-logo-img {{
    width: 255px;
    height: 255px;
    object-fit: contain;
    display: block;
}}

.sidebar-logo-text {{
    position: absolute;
    left: 50%;
    top: 74%;
    transform: translateX(-50%);
    font-size: 24px;
    font-weight: 900;
    color: #172033;
    white-space: nowrap;
    letter-spacing: -0.7px;
    text-align: center;
    text-shadow: 0 2px 7px rgba(255,255,255,0.98);
}}

.sidebar-logo-fallback-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
}}

.sidebar-logo-fallback {{
    width: 120px;
    height: 120px;
    border-radius: 28px;
    background: #185FA5;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 46px;
    font-weight: 900;
    margin-bottom: 6px;
}}
</style>

<div class="sidebar-header">
    {logo_html}
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


CLINICAL_RULES = {
    "CVA": {
        "normal": "0° ~ 20°",
        "caution": "20° 초과 ~ 28°",
        "risk": "28° 초과",
        "basis": "RULA Neck Zone 및 VDT 화면 상단-눈높이/하방 시선 기준",
    },
    "TIA": {
        "normal": "0° ~ 10°",
        "caution": "10° 초과 ~ 20°",
        "risk": "20° 초과",
        "basis": "RULA Trunk Zone 및 등받이 지지 기준",
    },
    "팔꿈치": {
        "normal": "90° ~ 120°",
        "caution": "해당 없음",
        "risk": "90° 미만 또는 120° 초과",
        "basis": "VDT 팔꿈치 내각 90° 이상 및 아래팔-손등 수평 기준",
    },
    "무릎": {
        "normal": "85° ~ 100°",
        "caution": "해당 없음",
        "risk": "85° 미만 또는 100° 초과",
        "basis": "VDT 무릎 내각 90° 전후 및 하지 지지 기준",
    },
    "손목": {
        "normal": "165° ~ 180°",
        "caution": "해당 없음",
        "risk": "165° 미만",
        "basis": "RULA Wrist Zone 및 손목 중립 ±15° 기준",
    },
    "시선각": {
        "normal": "하방 10° ~ 15°",
        "caution": "해당 없음",
        "risk": "10° 미만 또는 15° 초과",
        "basis": "VDT 수평 하방 10~15° 시선 기준",
    },
    "책상높이": {
        "normal": "팔꿈치-책상면 차이 0 ~ 0.05",
        "caution": "해당 없음",
        "risk": "0.05 초과",
        "basis": "팔꿈치와 책상면 수평 정렬, ±5%/±10% 허용 기준",
    },
    "등받이": {
        "normal": "골반너비 20% 이내",
        "caution": "해당 없음",
        "risk": "20% 초과",
        "basis": "VDT 의자 깊숙이 착석 및 RULA Trunk 지지조건 기준",
    },
}


DISPLAY_METRIC_ORDER = [
    "CVA",
    "TIA",
    "팔꿈치",
    "무릎",
    "손목",
    "시선각",
    "책상높이",
    "등받이",
]


def is_three_level_metric(key):
    """CVA, TIA만 정상/주의/위험 3분류로 판정합니다."""
    return key in ["CVA", "TIA"]


def get_range_text_html(key, line_break="<br/>"):
    """CVA/TIA는 3분류, 나머지 6개 지표는 정상/위험 2분류 기준으로 표시합니다."""
    rule = CLINICAL_RULES.get(key, {})
    if is_three_level_metric(key):
        return (
            f"정상: {rule.get('normal', '-')}"
            f"{line_break}주의: {rule.get('caution', '-')}"
            f"{line_break}위험: {rule.get('risk', '-')}"
        )
    return (
        f"정상: {rule.get('normal', '-')}"
        f"{line_break}위험: {rule.get('risk', '-')}"
    )


def classify_posture_level(key, raw):
    if raw is None:
        return "제외"

    raw = float(raw)

    # CVA, TIA만 정상/주의/위험 3분류
    if key == "CVA":
        if 0 <= raw <= 20:
            return "정상"
        if raw <= 28:
            return "주의"
        return "위험"

    if key == "TIA":
        if 0 <= raw <= 10:
            return "정상"
        if raw <= 20:
            return "주의"
        return "위험"

    # 나머지 6개 지표는 RULA/VDT 기준에 따라 정상/위험 2분류
    if key == "팔꿈치":
        return "정상" if 90 <= raw <= 120 else "위험"

    if key == "무릎":
        return "정상" if 85 <= raw <= 100 else "위험"

    if key == "손목":
        return "정상" if 165 <= raw <= 180 else "위험"

    if key == "시선각":
        return "정상" if 10 <= raw <= 15 else "위험"

    if key == "책상높이":
        return "정상" if raw <= 0.05 else "위험"

    if key == "등받이":
        return "정상" if raw <= 0.20 else "위험"

    return "정상"


def level_to_style(level):
    if level == "정상":
        return {
            "label": "정상",
            "class": "status-good",
            "color": "#45B86B",
            "desc": "양호",
            "score": 10,
            "marker": 17,
        }

    if level == "주의":
        return {
            "label": "주의",
            "class": "status-warn",
            "color": "#7467F0",
            "desc": "개선 필요",
            "score": 6,
            "marker": 50,
        }

    if level == "위험":
        return {
            "label": "위험",
            "class": "status-risk",
            "color": "#F2527D",
            "desc": "관리 필요",
            "score": 2,
            "marker": 83,
        }

    return {
        "label": "제외",
        "class": "status-none",
        "color": "#AEB6C2",
        "desc": "기준점 부족",
        "score": None,
        "marker": 50,
    }


def metric_status_for_card(key, is_good, raw):
    level = classify_posture_level(key, raw)
    return level_to_style(level)


def get_metric_range_html(key):
    """카드 안에 CVA/TIA는 3분류, 나머지는 2분류 기준을 표시합니다."""
    rule = CLINICAL_RULES.get(key, {})

    if is_three_level_metric(key):
        return f"""
        <div class="pretty-range-box">
            <div><b class="range-good">정상:</b> {rule.get("normal", "-")}</div>
            <div><b class="range-warn">주의:</b> {rule.get("caution", "-")}</div>
            <div><b class="range-risk">위험:</b> {rule.get("risk", "-")}</div>
        </div>
        """

    return f"""
    <div class="pretty-range-box">
        <div><b class="range-good">정상:</b> {rule.get("normal", "-")}</div>
        <div><b class="range-risk">위험:</b> {rule.get("risk", "-")}</div>
    </div>
    """


def gauge_percent(key, raw):
    level = classify_posture_level(key, raw)
    return level_to_style(level)["marker"]


def calculate_clinical_score_from_items(posture, env):
    all_data = {**posture, **env}
    scores = []
    level_counts = {"정상": 0, "주의": 0, "위험": 0, "제외": 0}

    for key in DISPLAY_METRIC_ORDER:
        if key not in all_data:
            continue
        _, _, raw = all_data[key]
        level = classify_posture_level(key, raw)
        level_counts[level] += 1
        score = level_to_style(level)["score"]
        if score is not None:
            scores.append(score)

    final_score = round(sum(scores) / len(scores), 1) if scores else 0

    if final_score >= 8:
        risk = "양호"
    elif final_score >= 5:
        risk = "주의"
    else:
        risk = "위험"

    return final_score, risk, level_counts




def gauge_percent(key, raw):
    level = classify_posture_level(key, raw)
    return level_to_style(level)["marker"]

def image_to_base64_src(path):
    try:
        path = Path(path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def metric_icon_svg(key, color=None):
    base_dir = Path(__file__).resolve().parent

    icon_map = {
        "CVA": base_dir / "assets" / "metric_icons" / "cva.png",
        "TIA": base_dir / "assets" / "metric_icons" / "tia.png",
        "팔꿈치": base_dir / "assets" / "metric_icons" / "elbow.png",
        "무릎": base_dir / "assets" / "metric_icons" / "knee.png",
        "손목": base_dir / "assets" / "metric_icons" / "wrist.png",
        "시선각": base_dir / "assets" / "metric_icons" / "gaze.png",
        "책상높이": base_dir / "assets" / "metric_icons" / "desk.png",
        "등받이": base_dir / "assets" / "metric_icons" / "chair.png",
    }

    img_src = image_to_base64_src(icon_map.get(key, ""))

    if img_src:
        return f'<img src="{img_src}" class="metric-img-icon">'

    return '<div class="metric-img-placeholder">이미지 없음</div>'

def image_to_base64_src(path):
    try:
        path = Path(path)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except:
        return ""

def render_pretty_8_metric_dashboard(result):
    all_data = {**result["posture"], **result["env"]}
    metric_order = DISPLAY_METRIC_ORDER

    cards_html = ""

    for idx, key in enumerate(metric_order, start=1):
        if key not in all_data:
            continue

        value, is_good, raw = all_data[key]
        fb = FEEDBACK[key]
        status = metric_status_for_card(key, is_good, raw)
        percent = status["marker"]
        icon = metric_icon_svg(key, status["color"])
        range_html = get_metric_range_html(key)

        if is_three_level_metric(key):
            gauge_html = f"""
            <div class="pretty-gauge">
                <div class="pretty-gauge-track">
                    <div class="pretty-zone zone-good"></div>
                    <div class="pretty-zone zone-warn"></div>
                    <div class="pretty-zone zone-risk"></div>
                    <div class="pretty-marker" style="left:{percent}%; background:{status["color"]};"></div>
                </div>
                <div class="pretty-range">
                    <span>정상</span>
                    <span>주의</span>
                    <span>위험</span>
                </div>
            </div>
            """
        else:
            gauge_html = f"""
            <div class="pretty-gauge">
                <div class="pretty-gauge-track">
                    <div class="pretty-zone zone-good-two"></div>
                    <div class="pretty-zone zone-risk-two"></div>
                    <div class="pretty-marker" style="left:{percent}%; background:{status["color"]};"></div>
                </div>
                <div class="pretty-range">
                    <span>정상</span>
                    <span>위험</span>
                </div>
            </div>
            """

        msg = fb["good"] if is_good else fb["bad"]
        msg_html = msg.replace("\n", "<br>")

        cards_html += f"""
        <div class="pretty-metric-card">
            <div class="pretty-card-top">
                <div class="pretty-card-title {status["class"]}">
                    {idx}. {fb["label"]}
                </div>
                <div class="pretty-card-eng">{fb["eng"]}</div>
            </div>

            <div class="pretty-icon">
                {icon}
            </div>

            <div class="pretty-value-bg" style="background:linear-gradient(180deg, {status["color"]}18, {status["color"]}08);">
                <div class="pretty-value" style="color:{status["color"]};">{value}</div>
                <div class="pretty-status {status["class"]}">{status["label"]}</div>
            </div>

            {gauge_html}

            <div class="pretty-desc">
                {range_html}
                <div class="current-status">
                    현재 상태: <b style="color:{status["color"]};">{status["desc"]}</b>
                </div>
            </div>

            <div class="pretty-feedback-box">
                <div class="pretty-feedback-title">맞춤 피드백</div>
                <div class="pretty-feedback-text">{msg_html}</div>
            </div>
        </div>
        """

    css_head = """
    <html>
    <head>
    <style>
    body {
        margin:0;
        padding:0;
        font-family:'Pretendard', Arial, sans-serif;
        background:transparent;
        color:#172033;
    }

    .pretty-dashboard {
        background:#FFFFFF;
        border:1px solid #E5EAF2;
        border-radius:24px;
        padding:28px 30px 26px 30px;
        box-shadow:0 12px 34px rgba(15,23,42,0.06);
        box-sizing:border-box;
    }

    .pretty-dashboard-title {
        text-align:center;
        font-size:30px;
        font-weight:900;
        color:#172033;
        letter-spacing:-1px;
        margin-bottom:6px;
    }

    .pretty-dashboard-sub {
        text-align:center;
        font-size:14px;
        color:#667085;
        margin-bottom:18px;
    }

    .pretty-legend {
        display:flex;
        justify-content:center;
        gap:20px;
        align-items:center;
        font-size:13px;
        color:#667085;
        margin-bottom:26px;
    }

    .legend-dot {
        width:11px;
        height:11px;
        border-radius:50%;
        display:inline-block;
        margin-right:6px;
    }

    .pretty-grid {
        display:grid;
        grid-template-columns:repeat(4, minmax(0, 1fr));
        gap:16px;
    }

    .pretty-metric-card,
    .pretty-guide {
        background:#FFFFFF;
        border:1px solid #E5EAF2;
        border-radius:20px;
        padding:18px;
        box-shadow:0 10px 26px rgba(15,23,42,0.045);
        min-height:430px;
        box-sizing:border-box;
        transition:all 0.18s ease;
    }

    .pretty-metric-card:hover {
        transform:translateY(-4px);
        box-shadow:0 16px 34px rgba(15,23,42,0.09);
    }

    .pretty-card-top {
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:8px;
        margin-bottom:10px;
    }

    .pretty-card-title {
        font-size:15px;
        font-weight:900;
        letter-spacing:-0.4px;
    }

    .pretty-card-eng {
        font-size:11px;
        color:#98A2B3;
        font-weight:700;
    }

    .status-good { color:#45B86B; }
    .status-warn { color:#7467F0; }
    .status-risk { color:#F2527D; }
    .status-none { color:#AEB6C2; }

    .pretty-icon {
        height:110px;
        display:flex;
        align-items:center;
        justify-content:center;
        margin:8px 0 14px 0;
        background:linear-gradient(180deg, #F8FAFC, #FFFFFF);
        border-radius:18px;
    }

    .metric-img-icon {
        width:132px;
        height:102px;
        object-fit:contain;
        display:block;
        filter:drop-shadow(0 8px 12px rgba(15,23,42,0.12));
    }

    .metric-img-placeholder {
        font-size:12px;
        color:#98A2B3;
        background:#F2F4F7;
        padding:10px 14px;
        border-radius:999px;
    }

    .pretty-value-bg {
        width:150px;
        height:76px;
        border-radius:90px 90px 0 0;
        margin:0 auto 12px auto;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        border-bottom:1px solid #E5EAF2;
    }

    .pretty-value {
        font-size:31px;
        font-weight:950;
        line-height:1;
    }

    .pretty-status {
        margin-top:8px;
        font-size:14px;
        font-weight:900;
    }

    .pretty-gauge-track {
        position:relative;
        height:12px;
        border-radius:999px;
        display:flex;
        background:#EEF2F6;
    }

    .pretty-zone {
        height:12px;
    }

    .zone-good {
        width:34%;
        background:#45B86B;
        border-radius:999px 0 0 999px;
    }

    .zone-warn {
        width:33%;
        background:#7467F0;
    }

    .zone-risk {
        width:33%;
        background:#F2527D;
        border-radius:0 999px 999px 0;
    }

    .zone-good-two {
        width:50%;
        background:#45B86B;
        border-radius:999px 0 0 999px;
    }

    .zone-risk-two {
        width:50%;
        background:#F2527D;
        border-radius:0 999px 999px 0;
    }

    .pretty-marker {
        position:absolute;
        top:-5px;
        width:10px;
        height:22px;
        border-radius:999px;
        transform:translateX(-50%);
        box-shadow:0 3px 8px rgba(0,0,0,0.18);
    }

    .pretty-range {
        display:flex;
        justify-content:space-between;
        font-size:10.5px;
        color:#667085;
        margin-top:7px;
    }

    .pretty-desc {
        background:#F8FAFC;
        border-radius:12px;
        padding:10px 11px;
        margin-top:12px;
        font-size:12px;
        line-height:1.6;
        color:#667085;
    }

    .pretty-range-box {
        display:flex;
        flex-direction:column;
        gap:4px;
        font-size:12px;
        line-height:1.55;
        color:#475467;
    }

    .range-good {
        color:#45B86B;
        font-weight:900;
    }

    .range-warn {
        color:#7467F0;
        font-weight:900;
    }

    .range-risk {
        color:#F2527D;
        font-weight:900;
    }

    .current-status {
        margin-top:8px;
        padding-top:8px;
        border-top:1px solid #E5EAF2;
        font-size:12px;
        color:#667085;
    }

    .pretty-feedback-box {
        margin-top:12px;
        background:#FFF7F8;
        border-left:4px solid #F2527D;
        border-radius:12px;
        padding:11px 12px;
    }

    .pretty-feedback-title {
        font-size:12px;
        font-weight:900;
        color:#172033;
        margin-bottom:6px;
    }

    .pretty-feedback-text {
        font-size:12.5px;
        line-height:1.65;
        color:#667085;
    }

    .pretty-guide {
        background:linear-gradient(135deg, #F7F4FF, #FFFFFF);
    }

    .pretty-guide-title {
        font-size:17px;
        font-weight:900;
        color:#172033;
        margin-bottom:14px;
    }

    .pretty-guide-row {
        display:flex;
        gap:10px;
        font-size:13px;
        line-height:1.7;
        color:#667085;
        margin-bottom:12px;
    }

    @media (max-width:1100px) {
        .pretty-grid {
            grid-template-columns:repeat(2, minmax(0, 1fr));
        }
    }
    </style>
    </head>

    <body>
    <div class="pretty-dashboard">
        <div class="pretty-dashboard-title">8가지 자세 측정 지표 결과</div>
        <div class="pretty-dashboard-sub">체간-책상거리 지표를 제외한 8가지 자세·작업환경 지표를 한눈에 확인하세요.</div>

        <div class="pretty-legend">
            <span><span class="legend-dot" style="background:#F2527D;"></span>위험</span>
            <span><span class="legend-dot" style="background:#7467F0;"></span>주의</span>
            <span><span class="legend-dot" style="background:#45B86B;"></span>정상</span>
            <span><span class="legend-dot" style="background:#AEB6C2;"></span>제외</span>
        </div>

        <div class="pretty-grid">
    """

    guide_html = """
            <div class="pretty-guide">
                <div class="pretty-guide-title">💡 결과 해석 가이드</div>

                <div class="pretty-guide-row">
                    <span style="color:#F2527D;">●</span>
                    <div><b>위험</b><br>정상 범위를 벗어나 자세 교정이 필요한 상태입니다.</div>
                </div>

                <div class="pretty-guide-row">
                    <span style="color:#7467F0;">●</span>
                    <div><b>주의</b><br>정상 범위에 가깝지만 지속 관찰이 필요한 상태입니다.</div>
                </div>

                <div class="pretty-guide-row">
                    <span style="color:#45B86B;">●</span>
                    <div><b>정상</b><br>현재 자세가 비교적 안정적인 상태입니다.</div>
                </div>

                <div style="margin-top:14px;font-size:12px;line-height:1.7;color:#98A2B3;">
                    ※ 측정값은 자세 참고용이며, 정확한 진단은 전문가 상담이 필요합니다.
                </div>
            </div>
        </div>
    </div>
    </body>
    </html>
    """

    html = css_head + cards_html + guide_html

    components.html(html, height=1900, scrolling=False)


def init_history():
    # =========================================================
# Session State 초기화
# =========================================================

    if "history" not in st.session_state:
        st.session_state.history = []

    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "history" not in st.session_state:
        st.session_state.history = []


HISTORY_DB_PATH = "user_history.json"
CHALLENGE_DB_PATH = "challenge_results.json"


def load_json_file(path, default):
    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_current_username():
    return st.session_state.get("username", "익명")


def save_history(result):
    username = get_current_username()
    histories = load_json_file(HISTORY_DB_PATH, {})

    if username not in histories:
        histories[username] = []

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    histories[username].insert(
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

    save_json_file(HISTORY_DB_PATH, histories)


def load_challenge_results():
    if not os.path.exists(CHALLENGE_DB_PATH):
        return []

    try:
        with open(CHALLENGE_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_challenge_results(results):
    with open(CHALLENGE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def sync_result_to_challenge(result):
    username = st.session_state.get("username", "익명")

    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_items = [
        FEEDBACK[key]["label"]
        for key, (_, is_good, _) in all_data.items()
        if key in FEEDBACK and not is_good
    ]

    new_record = {
        "name": username,
        "score": result["score"],
        "risk": result["risk"],
        "good": result["good_count"],
        "total": result["total_count"],
        "bad_items": bad_items[:3],
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    results = load_challenge_results()

    # 같은 사용자는 최신 측정 결과 1개만 유지
    results = [r for r in results if r.get("name") != username]
    results.insert(0, new_record)

    save_challenge_results(results)

def render_measurement_coverage(result_or_history):
    """양호 지표가 몇 개 기준으로 계산됐는지 설명합니다."""
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

# ===============================
# 바른자세 챌린지 JSON 구조 오류 해결
# 기존 함수와 교체해서 복붙하세요
# ===============================

def load_challenge_results():
    data = load_json_file(CHALLENGE_DB_PATH, {})

    # 예전 버전(list 구조) 자동 변환
    if isinstance(data, list):
        converted = {}

        for item in data:
            name = item.get("name", "익명")
            score = item.get("score", 0)
            point = int(round(score * 10))

            if name not in converted:
                converted[name] = {
                    "name": name,
                    "total_point": 0,
                    "count": 0,
                    "records": []
                }

            converted[name]["total_point"] += point
            converted[name]["count"] += 1
            converted[name]["records"].append(
                {
                    "score": score,
                    "point": point,
                    "risk": item.get("risk", "-"),
                    "good": item.get("good", 0),
                    "total": item.get("total", 0),
                    "bad_items": item.get("bad_items", []),
                    "time": item.get("time", "-"),
                }
            )

        save_challenge_results(converted)
        return converted

    # 새 버전(dict 구조)
    if isinstance(data, dict):
        return data

    return {}


def save_challenge_results(results):
    save_json_file(CHALLENGE_DB_PATH, results)


def sync_result_to_challenge(result):
    username = get_current_username()

    all_data = {**result.get("posture", {}), **result.get("env", {})}

    bad_items = [
        FEEDBACK[key]["label"]
        for key, (_, is_good, _) in all_data.items()
        if key in FEEDBACK and not is_good
    ]

    point = int(round(result["score"] * 10))

    challenge = load_challenge_results()

    if username not in challenge:
        challenge[username] = {
            "name": username,
            "total_point": 0,
            "count": 0,
            "records": [],
        }

    challenge[username]["total_point"] += point
    challenge[username]["count"] += 1
    challenge[username]["records"].insert(
        0,
        {
            "score": result["score"],
            "point": point,
            "risk": result["risk"],
            "good": result["good_count"],
            "total": result["total_count"],
            "bad_items": bad_items[:3],
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    )

    save_challenge_results(challenge)

# =========================================================
# 6. 사이드바 탭
# =========================================================

render_logo()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_auth_page()
    st.stop()

menu = st.sidebar.radio(
    "나의 건강",
    [
        "📸 자세측정",
        "📈 측정이력",
        "🧾 예상 영수증",
        "🎯 바른자세 챌린지",
        "📄 근골격계 리포트",
        "🛒 제품 추천",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 서비스 상태")
st.sidebar.caption("YOLO · MediaPipe 기반 자세 분석")
st.sidebar.caption("측면 사진 기준")
st.sidebar.caption("의료 진단이 아닌 자세 위험도 참고용")

st.sidebar.markdown("---")
st.sidebar.caption(f"로그인 계정: {st.session_state.username}")

if st.sidebar.button("로그아웃", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# =========================================================
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

    # 아직 측정 결과가 없는 경우
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

    # 상단 핵심 지표
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

    # 실제 신체 부위별 위험 현황
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

    # 교정 우선순위
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

    # 중단: 최근 측정 이미지 + AI 요약
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

        if not result.get("gate_pass", True):
            st.session_state.latest_result = None
            gate_metrics = result.get("gate_metrics", {})
            cva = gate_metrics.get("CVA", {})
            tia = gate_metrics.get("TIA", {})

            def _status_color(status):
                return "#3B8C42" if status == "GOOD" else "#D94A4A"

            # ── 레이아웃: 오버레이 이미지 | 판정 카드 ────────────────
            img_col, info_col = st.columns([1, 1])

            with img_col:
                overlay_img = result.get("overlay")
                if overlay_img is not None:
                    st.markdown(
                        """
<div style="border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
""", unsafe_allow_html=True)
                    st.image(overlay_img, use_container_width=True,
                             caption="현재 자세(빨강) vs 목표 자세(민트)")
                    st.markdown("</div>", unsafe_allow_html=True)

            with info_col:
                st.markdown(
                    f"""
<div class="fit-card" style="border-left:6px solid #D94A4A;
     background:linear-gradient(135deg,#FFF1F1,#FFFFFF);height:100%;">
    <div class="fit-card-title">
        <span>자세 교정이 필요합니다</span>
        <span class="fit-badge badge-red">BAD</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#172033;
         font-weight:700;margin-bottom:10px;">
        CVA 또는 TIA가 BAD 판정으로<br>
        환경 분석 결과를 제공하지 않습니다.
    </div>
    <div style="font-size:12.5px;line-height:1.8;color:#667085;margin-bottom:16px;">
        이미지의 <b style="color:#D94A4A;">빨간 선</b>이 현재 자세,
        <b style="color:#2ec4b6;">민트 선</b>이 목표 자세입니다.<br>
        목표 자세에 맞게 교정 후 다시 촬영해주세요.
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">
        <div style="padding:14px;border-radius:14px;background:#FFFFFF;
             border:2px solid {_status_color(cva.get('status','BAD'))};text-align:center;">
            <div style="font-size:11px;color:#667085;margin-bottom:4px;">CVA 목굴곡각</div>
            <div style="font-size:26px;font-weight:900;color:#172033;">
                {cva.get('value', '측정불가')}
            </div>
            <div style="font-size:13px;font-weight:900;
                 color:{_status_color(cva.get('status','BAD'))};">
                {cva.get('status', 'BAD')} · 정상 0°~20°
            </div>
        </div>
        <div style="padding:14px;border-radius:14px;background:#FFFFFF;
             border:2px solid {_status_color(tia.get('status','BAD'))};text-align:center;">
            <div style="font-size:11px;color:#667085;margin-bottom:4px;">TIA 몸통굴곡각</div>
            <div style="font-size:26px;font-weight:900;color:#172033;">
                {tia.get('value', '측정불가')}
            </div>
            <div style="font-size:13px;font-weight:900;
                 color:{_status_color(tia.get('status','BAD'))};">
                {tia.get('status', 'BAD')} · 정상 0°~10°
            </div>
        </div>
    </div>
    <div style="padding:12px;border-radius:10px;background:#FFF8E1;
         border-left:4px solid #F59E0B;font-size:12.5px;color:#92400E;line-height:1.7;">
        💡 측면에서 <b>코·어깨·골반</b>이 잘 보이도록 촬영하면<br>
        더 정확한 분석이 가능합니다.
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.warning("자세를 교정하고 다시 촬영해주세요.")
            return

        st.session_state.latest_result = result
        save_history(result)
        sync_result_to_challenge(result)
        st.success("분석이 완료되었습니다. 측정이력과 바른자세 챌린지 포인트에 반영되었습니다.")

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

        img_col, summary_col = st.columns([1.05, 0.95])

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

        with summary_col:
            correction_html = build_ai_correction_comment(result)
            components.html(correction_html, height=650, scrolling=True)

        st.markdown("### 8개 측정 지표 결과")
        render_pretty_8_metric_dashboard(result)


def render_history():
    page_header(
        "측정 이력",
        "현재 로그인한 계정의 자세 분석 결과만 확인합니다.",
    )

    username = get_current_username()
    histories = load_json_file(HISTORY_DB_PATH, {})
    user_history = histories.get(username, [])

    if not user_history:
        st.info(f"{username}님의 측정 이력이 아직 없습니다.")
        return

    latest = user_history[0]

    # 상단 핵심 지표
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

    # ==================================================
    # 1. 내 점수 추이 (최근 측정기록 위로 이동)
    # ==================================================
    if len(user_history) >= 2:
        st.markdown("### 내 점수 추이")

        chart_data = pd.DataFrame(
            [
                {
                    "회차": i + 1,
                    "점수": float(h["score"]),
                    "측정시간": h["time"],
                }
                for i, h in enumerate(reversed(user_history))
            ]
        )

        base = alt.Chart(chart_data).encode(
            x=alt.X(
                "회차:O",
                title="측정 회차",
                axis=alt.Axis(labelAngle=0, labelPadding=10, titlePadding=16),
            ),
            y=alt.Y(
                "점수:Q",
                title="자세 점수",
                scale=alt.Scale(domain=[0, 10]),
                axis=alt.Axis(values=[0, 2, 4, 6, 8, 10], titlePadding=32, labelPadding=12),
            ),
            tooltip=[
                alt.Tooltip("측정시간:N", title="측정 시간"),
                alt.Tooltip("점수:Q", title="점수", format=".1f"),
            ],
        )

        line = base.mark_line(
            strokeWidth=4,
            interpolate="monotone",
        )

        points = base.mark_circle(
            size=95,
            opacity=1,
        )

        labels = base.mark_text(
            align="center",
            baseline="bottom",
            dy=-10,
            fontSize=13,
            fontWeight="bold",
        ).encode(
            text=alt.Text("점수:Q", format=".1f")
        )

        chart = (
            (line + points + labels)
            .properties(
                height=430,
                padding={"left": 78, "right": 28, "top": 24, "bottom": 20},
            )
            .configure_view(strokeWidth=0)
            .configure_axis(
                gridColor="#E5EAF2",
                labelColor="#667085",
                titleColor="#172033",
                labelFontSize=13,
                titleFontSize=15,
                titleFontWeight="bold",
            )
        )

        st.altair_chart(chart, use_container_width=True)

    # ==================================================
    # 2. 최근 측정기록
    # ==================================================
    st.markdown(f"### {username}님의 최근 측정 기록")

    risk = latest["risk"]

    if risk == "양호":
        color = "#3B8C42"
    elif risk == "주의":
        color = "#BA7517"
    else:
        color = "#D94A4A"

    latest_rate = round(latest["good"] / latest["total"] * 100)

    missing_items = latest.get("missing_items", []) or []
    if missing_items:
        coverage_note = f"측정 제외: {', '.join([item['label'] for item in missing_items])}"
    else:
        coverage_note = "8개 항목 전체 측정"

    st.markdown(
        f"""
<div class="fit-card" style="border-left:5px solid {color};">
<b>{latest["time"]}</b><br><br>
종합 점수: <b>{latest["score"]}/10</b><br>
위험도: <b>{risk}</b><br>
양호 지표: <b>{latest["good"]}/{latest["total"]}</b><br>
정상 비율: <b>{latest_rate}%</b><br>
<span style="font-size:12px;color:#667085;">{coverage_note}</span>

<div style="margin-top:10px;height:8px;background:#EEF2F6;border-radius:999px;">
<div style="height:8px;width:{latest_rate}%;background:{color};border-radius:999px;"></div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ==================================================
    # 3. 전체 측정이력
    # ==================================================
    st.markdown("### 전체 측정이력")

    for h in user_history:
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
    st.markdown("## 바른자세 챌린지")
    st.caption("자세측정을 할 때마다 점수가 포인트로 누적됩니다.")

    if "challenge_times" not in st.session_state:
        st.session_state.challenge_times = []

    left, right = st.columns([0.9, 1.1])

    with left:
        st.markdown("### 알림 시간 설정")

        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            hour = st.selectbox("시", range(24), format_func=lambda x: "{:02d}".format(x), key="challenge_hour")

        with c2:
            minute = st.selectbox("분", range(60), format_func=lambda x: "{:02d}".format(x), key="challenge_minute")

        with c3:
            st.write("")
            st.write("")
            if st.button("추가", use_container_width=True, key="add_challenge_alarm"):
                t = "{:02d}:{:02d}".format(hour, minute)

                if t not in st.session_state.challenge_times:
                    st.session_state.challenge_times.append(t)
                    st.success("{} 알림 추가".format(t))
                else:
                    st.warning("이미 추가된 시간입니다.")

        if st.session_state.challenge_times:
            st.markdown("#### 설정된 알림")

            for t in sorted(st.session_state.challenge_times):
                c_time, c_delete = st.columns([4, 1])

                with c_time:
                    st.write("⏰ {}".format(t))

                with c_delete:
                    if st.button("삭제", key="delete_alarm_{}".format(t)):
                        st.session_state.challenge_times.remove(t)
                        st.rerun()

            render_alarm_effect(st.session_state.challenge_times)
        else:
            st.info("알림 시간이 아직 없습니다.")

    with right:
        st.markdown("### 4팀 척추처척추")
        st.caption("누적 포인트가 높을수록 결승선에 가까워집니다.")

        challenge = load_challenge_results()

        if not challenge:
            st.info("아직 자세측정을 완료한 팀원이 없습니다.")
            return

        if isinstance(challenge, list):
            converted = {}
            for item in challenge:
                name = item.get("name", "익명")
                score = item.get("score", 0)
                point = int(round(score * 10))

                if name not in converted:
                    converted[name] = {
                        "name": name,
                        "total_point": 0,
                        "count": 0,
                        "records": [],
                    }

                converted[name]["total_point"] += point
                converted[name]["count"] += 1
                converted[name]["records"].insert(0, {
                    "score": score,
                    "point": point,
                    "risk": item.get("risk", "-"),
                    "good": item.get("good", 0),
                    "total": item.get("total", 0),
                    "bad_items": item.get("bad_items", []),
                    "time": item.get("time", "-"),
                })

            challenge = converted
            save_challenge_results(challenge)

        members = sorted(
            challenge.values(),
            key=lambda x: x.get("total_point", 0),
            reverse=True,
        )

        max_point = max([m.get("total_point", 0) for m in members]) or 1
        icons = ["🐰", "🐢", "🦊", "🐻", "🐼", "🐯", "🐸", "🐹"]

        race_html = """
<div style="background:#F8FAFC;border:1px solid #E5EAF2;border-radius:22px;padding:18px;margin-bottom:18px;">
<div style="font-size:18px;font-weight:900;color:#172033;margin-bottom:14px;">🐰 자세 포인트 레이스 🐢</div>
"""

        for idx, member in enumerate(members):
            name = member.get("name", "익명")
            total_point = member.get("total_point", 0)
            count = member.get("count", 0)
            records = member.get("records", [])
            latest = records[0] if records else {}

            latest_score = latest.get("score", "-")
            latest_risk = latest.get("risk", "-")

            percent = int((total_point / max_point) * 100)
            percent = max(8, min(percent, 100))
            icon = icons[idx % len(icons)]

            race_html += """
<div style="margin-bottom:18px;">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
<div style="font-size:14px;font-weight:800;color:#172033;">{rank}. {name}</div>
<div style="font-size:13px;font-weight:800;color:#185FA5;">{point}P · {count}회</div>
</div>
<div style="position:relative;height:38px;background:#EAF0F7;border-radius:999px;overflow:hidden;">
<div style="position:absolute;left:0;top:0;height:38px;width:{percent}%;background:linear-gradient(90deg,#DFF3FF,#B9E6FF);border-radius:999px;"></div>
<div style="position:absolute;left:calc({percent}% - 22px);top:2px;font-size:28px;">{icon}</div>
<div style="position:absolute;right:10px;top:8px;font-size:18px;">🏁</div>
</div>
<div style="font-size:12px;color:#667085;margin-top:5px;">최근 점수 {score}/10 · 상태 {risk}</div>
</div>
""".format(
                rank=idx + 1,
                name=name,
                point=total_point,
                count=count,
                percent=percent,
                icon=icon,
                score=latest_score,
                risk=latest_risk,
            )

        race_html += "</div>"
        st.markdown(race_html, unsafe_allow_html=True)

        st.markdown("### 누적 포인트 순위")

        for i, member in enumerate(members, start=1):
            name = member.get("name", "익명")
            total_point = member.get("total_point", 0)
            count = member.get("count", 0)
            records = member.get("records", [])
            latest = records[0] if records else {}

            latest_time = latest.get("time", "-")
            latest_score = latest.get("score", "-")
            latest_risk = latest.get("risk", "-")
            bad_items = latest.get("bad_items", [])

            if latest_risk == "양호":
                badge = "badge-green"
            elif latest_risk == "주의":
                badge = "badge-amber"
            else:
                badge = "badge-red"

            bad_text = " · ".join(bad_items) if bad_items else "관리 필요 항목 없음"

            card_html = """
<div class="fit-card">
<div class="fit-card-title">
<span>{rank}. {name}</span>
<span class="fit-badge {badge}">{risk}</span>
</div>
<div style="font-size:28px;font-weight:900;color:#185FA5;margin-bottom:8px;">{point}P</div>
<div style="font-size:14px;line-height:1.8;color:#172033;">
측정 횟수: <b>{count}회</b><br>
최근 점수: <b>{score}/10</b><br>
최근 측정: <b>{time}</b><br>
교정 필요: <b>{bad_text}</b>
</div>
</div>
""".format(
                rank=i,
                name=name,
                badge=badge,
                risk=latest_risk,
                point=total_point,
                count=count,
                score=latest_score,
                time=latest_time,
                bad_text=bad_text,
            )

            st.markdown(card_html, unsafe_allow_html=True)


def calculate_receipt_total(active_d, period_label, t_dosu=True, t_shock=True, t_prolo=True):
    """영수증에 표시될 비급여 TOTAL 금액을 계산합니다."""
    selected_period = next(p for p in PERIODS if p["label"] == period_label)
    sessions = selected_period["sessions"]

    used_np = []
    for d in active_d:
        for code in NONPAY_CODES.get(d, []):
            if code not in used_np:
                used_np.append(code)

    total_cost = 0
    for code in used_np:
        info = NONPAY_INFO[code]
        is_active = (
            (code == "도수" and t_dosu)
            or (code == "체외" and t_shock)
            or (code.startswith("증식") and t_prolo)
        )
        if is_active:
            total_cost += info["avg"] * sessions

    return total_cost

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
    total_cost = calculate_receipt_total(active_d, period_label, t_dosu, t_shock, t_prolo)

    with right:
        st.markdown(
            f"""
<div style="
    width:100%;
    max-width:430px;
    margin:0 auto 14px auto;
    text-align:left;
">
    <div style="
        font-size:14px;
        font-weight:800;
        color:#667085;
        margin-bottom:6px;
        letter-spacing:-0.3px;
    ">
        예상 총 진료비
    </div>
    <div style="
        font-size:42px;
        font-weight:950;
        color:#D94A4A;
        line-height:1.05;
        letter-spacing:-1.2px;
    ">
        {total_cost:,}원
    </div>
    <div style="
        margin-top:8px;
        font-size:12.5px;
        color:#98A2B3;
        line-height:1.5;
    ">
        선택한 질환 위치 · 치료 방식 · 치료 기간 기준
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

        components.html(receipt_html, height=880, scrolling=True)

# =========================================================
# 제품 추천 탭
# =========================================================

PRODUCT_RECOMMENDATIONS = {
    "경추": [
        {
            "name": "목 지지대 + 높낮이 조절 의자",
            "reason": "목·경추 부담이 크거나 등받이 지지가 부족할 때 추천",
            "url": "https://www.coupang.com/vp/products/7830801595?itemId=21297117616&vendorItemId=88356855899&q=시디즈+의자",
        },
        {
            "name": "모니터 받침대",
            "reason": "시선각이 맞지 않거나 모니터가 낮아 목이 앞으로 숙여질 때 추천",
            "url": "https://www.coupang.com/vp/products/8641572134?itemId=25078452009&vendorItemId=92082407026",
        },
    ],
    "요추": [
        {
            "name": "등받이 요추 쿠션",
            "reason": "몸통이 앞으로 굽거나 허리 지지가 부족할 때 추천",
            "url": "https://drohbros.com/product/바른자세-허리쿠션-룸바/29/",
        },
        {
            "name": "허리 보호대",
            "reason": "허리 부담이 크고 장시간 앉아 있는 경우 보조용으로 추천",
            "url": "https://www.coupang.com/vp/products/8525671118?itemId=24684684526&vendorItemId=91509747922",
        },
    ],
    "팔꿈치": [
        {
            "name": "팔받침대",
            "reason": "팔꿈치 각도가 맞지 않거나 책상과 팔 높이가 맞지 않을 때 추천",
            "url": "https://www.coupang.com/vp/products/9061406350?itemId=26604067534&vendorItemId=93577201462",
        },
    ],
    "손목": [
        {
            "name": "손목 받침대",
            "reason": "손목이 꺾이거나 키보드 사용 시 손목 부담이 클 때 추천",
            "url": "https://www.coupang.com/vp/products/8604154504?itemId=24950395992&vendorItemId=91962355876",
        },
        {
            "name": "버티컬 마우스",
            "reason": "마우스 사용 시 손목 회전 부담이 크거나 손목 통증 예방이 필요할 때 추천",
            "url": "https://www.coupang.com/vp/products/7295558262?itemId=20340965740&vendorItemId=86330525952&q=버티컬+마우스",
        },
    ],
    "무릎": [
        {
            "name": "사무실 발받침대",
            "reason": "무릎 각도가 맞지 않거나 발이 바닥에 안정적으로 닿지 않을 때 추천",
            "url": "https://www.coupang.com/vp/products/5227999402?itemId=23156160593&vendorItemId=74642538655&q=사무실+발받침대",
        },
    ],
}


def map_result_to_product_categories(result):
    all_data = {**result.get("posture", {}), **result.get("env", {})}
    categories = []

    # 경추: 목굴곡각, 시선각 문제
    if not all_data.get("CVA", ("", True, None))[1] or not all_data.get("시선각", ("", True, None))[1]:
        categories.append("경추")

    # 요추: 몸통굴곡각, 등받이 문제
    if not all_data.get("TIA", ("", True, None))[1] or not all_data.get("등받이", ("", True, None))[1]:
        categories.append("요추")

    # 팔꿈치: 팔꿈치 각도, 책상높이 문제
    if not all_data.get("팔꿈치", ("", True, None))[1] or not all_data.get("책상높이", ("", True, None))[1]:
        categories.append("팔꿈치")

    # 손목
    if not all_data.get("손목", ("", True, None))[1]:
        categories.append("손목")

    # 무릎
    if not all_data.get("무릎", ("", True, None))[1]:
        categories.append("무릎")

    return list(dict.fromkeys(categories))


def render_product_recommendation_page():
    page_header(
        "제품 추천",
        "자세측정 결과에서 기준 범위를 벗어난 부위에 맞춰 필요한 제품을 추천합니다.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("제품 추천을 보려면 먼저 왼쪽 메뉴의 자세측정에서 AI 자세 분석을 실행해주세요.")
        return

    categories = map_result_to_product_categories(result)

    if not categories:
        st.success("현재 자세측정 결과상 필수 추천 제품은 없습니다. 현재 작업환경을 잘 유지해주세요.")
        return

    all_data = {**result.get("posture", {}), **result.get("env", {})}
    bad_labels = [
        FEEDBACK[k]["label"]
        for k, v in all_data.items()
        if k in FEEDBACK and not v[1]
    ]

    st.markdown(
        f"""
<div class="fit-card">
    <div class="fit-card-title">
        <span>추천 기준</span>
        <span class="fit-badge badge-blue">AI Product Match</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#667085;">
        기준 범위를 벗어난 항목:
        <b style="color:#172033;">{" · ".join(bad_labels) if bad_labels else "없음"}</b><br>
        추천 카테고리:
        <b style="color:#185FA5;">{" · ".join(categories)}</b>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>
.product-grid {
    display:grid;
    grid-template-columns:repeat(3, minmax(0, 1fr));
    gap:16px;
    margin-top:10px;
}
.product-card {
    background:#FFFFFF;
    border:1px solid #E5EAF2;
    border-radius:18px;
    padding:18px;
    box-shadow:0 8px 28px rgba(15,23,42,0.04);
    min-height:210px;
}
.product-category {
    display:inline-flex;
    padding:5px 10px;
    border-radius:999px;
    background:#E6F1FB;
    color:#0C447C;
    font-size:12px;
    font-weight:800;
    margin-bottom:12px;
}
.product-name {
    font-size:18px;
    font-weight:900;
    color:#172033;
    margin-bottom:8px;
    letter-spacing:-0.4px;
}
.product-reason {
    font-size:13px;
    line-height:1.7;
    color:#667085;
    min-height:68px;
    margin-bottom:14px;
}
.product-link {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:100%;
    padding:10px 12px;
    border-radius:12px;
    background:#185FA5;
    color:white !important;
    text-decoration:none !important;
    font-size:13px;
    font-weight:850;
}
@media (max-width: 1000px) {
    .product-grid {
        grid-template-columns:repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 700px) {
    .product-grid {
        grid-template-columns:1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )

    html = '<div class="product-grid">'

    for category in categories:
        for product in PRODUCT_RECOMMENDATIONS.get(category, []):
            html += f"""
<div class="product-card">
    <div class="product-category">{category}</div>
    <div class="product-name">{product["name"]}</div>
    <div class="product-reason">{product["reason"]}</div>
    <a class="product-link" href="{product["url"]}" target="_blank">
        제품 보러가기
    </a>
</div>
"""

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)

    st.caption("※ 제품 추천은 자세측정 결과 기반의 작업환경 개선 참고용이며, 의료적 진단이나 치료 목적이 아닙니다.")    

# =========================================================
# 근골격계 리포트 전용 UI — 직관형 그래프 + 3단계 상세 피드백
# =========================================================

def _report_level_style(level):
    if level == "정상":
        return {"color": "#45B86B", "soft": "#F0FBF4", "badge": "정상", "desc": "양호", "marker": 17}
    if level == "주의":
        return {"color": "#7467F0", "soft": "#F4F1FF", "badge": "주의", "desc": "개선 필요", "marker": 50}
    if level == "위험":
        return {"color": "#F2527D", "soft": "#FFF1F5", "badge": "위험", "desc": "관리 필요", "marker": 83}
    return {"color": "#AEB6C2", "soft": "#F2F4F7", "badge": "제외", "desc": "기준점 부족", "marker": 50}


def _report_feedback_text(key, is_normal):
    fb = FEEDBACK.get(key, {})
    text = fb.get("good", "") if is_normal else fb.get("bad", "")
    return text.replace("\n", "<br>")


def _report_metric_icon(key):
    base_dir = Path(__file__).resolve().parent

    icon_map = {
        "CVA": base_dir / "assets" / "metric_icons" / "cva.png",
        "TIA": base_dir / "assets" / "metric_icons" / "tia.png",
        "팔꿈치": base_dir / "assets" / "metric_icons" / "elbow.png",
        "무릎": base_dir / "assets" / "metric_icons" / "knee.png",
        "손목": base_dir / "assets" / "metric_icons" / "wrist.png",
        "시선각": base_dir / "assets" / "metric_icons" / "gaze.png",
        "책상높이": base_dir / "assets" / "metric_icons" / "desk.png",
        "등받이": base_dir / "assets" / "metric_icons" / "chair.png",
    }

    img_src = image_to_base64_src(icon_map.get(key, ""))
    if img_src:
        return f"<img class='ms-metric-img' src='{img_src}' alt='{key}'>"

    return "📍"


def _report_range_html(key):
    rule = CLINICAL_RULES.get(key, {})
    if is_three_level_metric(key):
        return f'''
        <div class="report-range-box">
            <div><b class="report-range-good">정상:</b> {rule.get("normal", "-")}</div>
            <div><b class="report-range-warn">주의:</b> {rule.get("caution", "-")}</div>
            <div><b class="report-range-risk">위험:</b> {rule.get("risk", "-")}</div>
        </div>
        '''
    return f'''
    <div class="report-range-box">
        <div><b class="report-range-good">정상:</b> {rule.get("normal", "-")}</div>
        <div><b class="report-range-risk">위험:</b> {rule.get("risk", "-")}</div>
    </div>
    '''


def generate_legal_pdf(is_vdt_over_4h: bool):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

    buffer = BytesIO()
    font_name = get_korean_pdf_font()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=8 * mm,
        bottomMargin=8 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="LegalTitle",
        fontName=font_name,
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        name="LegalSub",
        fontName=font_name,
        fontSize=12,
        leading=16,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        name="LegalCell",
        fontName=font_name,
        fontSize=6.2,
        leading=8,
        alignment=1,
        textColor=colors.black,
    ))
    styles.add(ParagraphStyle(
        name="LegalSmall",
        fontName=font_name,
        fontSize=7,
        leading=9,
        alignment=1,
        textColor=colors.black,
    ))

    def P(text, style="LegalCell"):
        return Paragraph(str(text).replace("\n", "<br/>"), styles[style])

    story = []

    story.append(Paragraph("근골격계부담작업 체크리스트", styles["LegalTitle"]))
    story.append(Spacer(1, 4))

    top_table = Table(
        [
            ["사업장명", "아시아경제교육센터", "조사 일자", "2026년 4월 30일", "조사자", "김OO"],
            ["부서명", "4팀", "작업 내용", "사무작업", "", ""],
        ],
        colWidths=[24*mm, 55*mm, 24*mm, 50*mm, 24*mm, 45*mm],
        rowHeights=[9*mm, 9*mm],
    )
    top_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.black),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EDEDED")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EDEDED")),
        ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#EDEDED")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (4, 1), (5, 1)),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 4))

    headers = ["구분"] + [f"{i})" for i in range(1, 12)]

    exposure_time = [
        "노출 시간",
        "하루에\n총 4시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "-",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
        "하루에\n총 2시간 이상",
    ]

    body_part = [
        "신체 부위",
        "손, 손가락,\n팔, 어깨",
        "목, 어깨,\n손목, 손,\n팔꿈치",
        "어깨, 팔",
        "목, 허리",
        "다리, 무릎",
        "손가락",
        "손",
        "허리",
        "손, 무릎",
        "허리",
        "목, 무릎,\n팔꿈치",
    ]

    work_posture = [
        "작업 자세\n및\n내용",
        "집중적인\n입력 작업\n(마우스·키보드 사용)",
        "같은 동작\n반복 작업",
        "머리 위 또는\n팔꿈치가 몸통\n뒤쪽에 위치",
        "구부리거나\n비트는 자세",
        "쪼그리고 앉거나\n무릎을 굽힘",
        "한 손가락 집어\n올리거나 쥐는 작업",
        "물건을 한손으로\n들거나 잡는 작업",
        "물건을 드는 작업",
        "어깨 위에서\n팔을 드는 작업",
        "물건을 드는 작업",
        "반복적인 충격",
    ]

    weight = [
        "무게",
        "-",
        "-",
        "-",
        "-",
        "-",
        "1kg 이상 물건\n또는 2kg 이상\n상응하는 힘",
        "4.5kg 이상\n물건 들기",
        "25kg 이상",
        "10kg 이상",
        "4.5kg 이상",
        "-",
    ]

    mark_office = "O" if is_vdt_over_4h else "X"

    table_data = [
        [P(x, "LegalSmall") for x in headers],
        [P(x) for x in exposure_time],
        [P(x) for x in body_part],
        [P(x) for x in work_posture],
        [P(x) for x in weight],
        [P("단위작업명", "LegalSmall"), P("설계작업", "LegalSmall")] + [P("X", "LegalSmall") for _ in range(11)],
        ["", P("사무작업", "LegalSmall")] + [P(mark_office if i == 1 else "X", "LegalSmall") for i in range(1, 12)],
        ["", P("현장작업", "LegalSmall")] + [P("X", "LegalSmall") for _ in range(11)],
        ["", P("시운전", "LegalSmall")] + [P("X", "LegalSmall") for _ in range(11)],
    ]

    col_widths = [21*mm, 18*mm] + [20*mm] * 11

    main_table = Table(table_data, colWidths=col_widths, repeatRows=1)
    main_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E6E6E6")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (0, 5), (0, 8)),
        ("FONTSIZE", (0, 0), (-1, -1), 6.2),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    story.append(main_table)
    story.append(Spacer(1, 6))

    note = (
        "※ 본 체크리스트는 근골격계부담작업 제1호 해당 여부를 중심으로 자동 작성되었습니다. "    
    )
    story.append(Paragraph(note, styles["LegalSub"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

def render_legal_checklist_page():
    page_header(
        "근골격계부담작업 체크리스트",
        "제1호 VDT 작업 해당 여부만 확인하고 법정 PDF를 생성합니다.",
    )

    st.markdown("""
<div class="fit-card">
    <div class="fit-card-title">
        <span>제1호 VDT 작업 확인</span>
        <span class="fit-badge badge-blue">Legal Checklist</span>
    </div>
    <div style="font-size:14px;line-height:1.8;color:#667085;">
        하루에 4시간 이상 집중적으로 자료입력 등을 위해 키보드 또는 마우스를 조작하는 작업에 해당하는지 확인합니다.<br>
        화면에는 제1호 기준만 표시되며, PDF에는 법정 체크리스트 양식에 따라 제1호부터 제11호까지 포함됩니다.
    </div>
</div>
""", unsafe_allow_html=True)

    is_vdt_over_4h = st.radio(
        "하루에 4시간 이상 집중적으로 키보드 또는 마우스를 조작했나요?",
        ["네", "아니오"],
        horizontal=True,
        key="legal_vdt_over_4h",
    ) == "네"

    pdf_buffer = generate_legal_pdf(is_vdt_over_4h)

    st.download_button(
        label="📋 근골격계부담작업 체크리스트 PDF 다운로드",
        data=pdf_buffer,
        file_name="legal_musculoskeletal_checklist_20260430.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

def render_report():
    page_header(
        "근골격계 리포트",
        "8가지 항목의 자세 및 환경을 종합적으로 분석했습니다.",
    )

    result = st.session_state.get("latest_result")

    if result is None:
        st.info("리포트를 생성하려면 먼저 자세 측정을 실행해주세요.")
        return

    pdf_buffer = make_musculoskeletal_report_pdf(result)
    st.download_button(
        label="📄 근골격계 리포트 PDF 저장",
        data=pdf_buffer,
        file_name=f"fit_me_up_musculoskeletal_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    all_data = {**result["posture"], **result["env"]}
    
    st.markdown("---")
    st.subheader(":clipboard: 법정 문서(근골격계부담작업 체크리스트) 자동 생성")
    st.caption("법정 유해요인 조사 결과(제1호)를 PDF로 출력합니다")

    is_vdt_over_4h_report = st.radio(
        "하루에 4시간 이상 집중적으로 자료입력 등을 위해 키보드 또는 마우스를 조작했나요?",
        ["네", "아니오"],
        horizontal=True,
        key="report_legal_vdt_over_4h",
    ) == "네"

    legal_pdf_buffer = generate_legal_pdf(is_vdt_over_4h_report)

    st.download_button(
        label="📋 법정 유해요인 조사 PDF 다운로드",
        data=legal_pdf_buffer,
        file_name="legal_musculoskeletal_checklist_20260430.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    def _level_info(level):
        if level == "정상":
            return {"label": "GOOD", "kr": "양호", "color": "#2FB35A", "soft": "#EAF8EF", "class": "good"}
        if level == "주의":
            return {"label": "CAUTION", "kr": "주의", "color": "#F2A900", "soft": "#FFF5D8", "class": "caution"}
        if level == "위험":
            return {"label": "BAD", "kr": "위험", "color": "#F43F5E", "soft": "#FFE8EE", "class": "bad"}
        return {"label": "EXCLUDED", "kr": "제외", "color": "#98A2B3", "soft": "#F2F4F7", "class": "none"}

    def _fmt_value(key, value, raw):
        if raw is None:
            return "인식 불가"
        if key in ["책상높이"]:
            return f"{float(raw):.3f}"
        if key in ["등받이"]:
            return f"{float(raw) * 100:.1f}%"
        return value

    def _subtitle(key):
        return {
            "CVA": "목의 전방 기울기 각도",
            "TIA": "몸통의 전방 굴곡 각도",
            "팔꿈치": "팔꿈치 굴곡 각도",
            "무릎": "무릎 굴곡 각도",
            "손목": "손목 굴곡 각도",
            "시선각": "수평선 대비 시선 각도",
            "책상높이": "팔꿈치 대비 작업대 높이",
            "등받이": "등받이 지지 비율",
        }.get(key, "측정 지표")

    def _icon(key):
        base_dir = Path(__file__).resolve().parent

        icon_map = {
            "CVA": base_dir / "assets" / "metric_icons" / "cva.png",
            "TIA": base_dir / "assets" / "metric_icons" / "tia.png",
            "팔꿈치": base_dir / "assets" / "metric_icons" / "elbow.png",
            "무릎": base_dir / "assets" / "metric_icons" / "knee.png",
            "손목": base_dir / "assets" / "metric_icons" / "wrist.png",
            "시선각": base_dir / "assets" / "metric_icons" / "gaze.png",
            "책상높이": base_dir / "assets" / "metric_icons" / "desk.png",
            "등받이": base_dir / "assets" / "metric_icons" / "chair.png",
        }

        img_src = image_to_base64_src(icon_map.get(key, ""))
        if img_src:
            return f"<img class='ms-metric-img' src='{img_src}' alt='{key}'>"

        return "📍"

    def _bar_meta(key):
        # min/max는 그래프 표시용 범위입니다. 실제 판정은 classify_posture_level() 기준을 사용합니다.
        return {
            "CVA": {"min": 0, "max": 40, "unit": "°", "segments": [(0, 20, "good"), (20, 28, "caution"), (28, 40, "bad")], "ticks": [0, 20, 28, 40]},
            "TIA": {"min": 0, "max": 45, "unit": "°", "segments": [(0, 10, "good"), (10, 20, "caution"), (20, 45, "bad")], "ticks": [0, 10, 20, 45]},
            "팔꿈치": {"min": 70, "max": 140, "unit": "°", "segments": [(70, 90, "bad"), (90, 120, "good"), (120, 140, "bad")], "ticks": [90, 120]},
            "무릎": {"min": 65, "max": 125, "unit": "°", "segments": [(65, 85, "bad"), (85, 100, "good"), (100, 125, "bad")], "ticks": [85, 100]},
            "손목": {"min": 140, "max": 180, "unit": "°", "segments": [(140, 165, "bad"), (165, 180, "good")], "ticks": [165, 180]},
            "시선각": {"min": -10, "max": 45, "unit": "°", "segments": [(-10, 10, "bad"), (10, 15, "good"), (15, 45, "bad")], "ticks": [10, 15]},
            "책상높이": {"min": -0.05, "max": 0.15, "unit": "", "segments": [(-0.05, 0.05, "good"), (0.05, 0.15, "bad")], "ticks": [0, 0.05]},
            "등받이": {"min": 0, "max": 0.50, "unit": "%", "segments": [(0, 0.20, "good"), (0.20, 0.50, "bad")], "ticks": [0, 0.20]},
        }.get(key)

    def _pct(value, min_v, max_v):
        if max_v == min_v:
            return 0
        return max(0, min(100, (float(value) - min_v) / (max_v - min_v) * 100))

    def _tick_label(key, v, unit):
        if key == "등받이":
            return f"{int(round(v * 100))}%"
        if key == "책상높이":
            return "0" if abs(v) < 1e-9 else f"{v:.2f}"
        return f"{int(v) if float(v).is_integer() else v}{unit}"

    def _range_lines(key):
        rule = CLINICAL_RULES.get(key, {})
        if is_three_level_metric(key):
            return f"""
            <div><b class='range-good'>정상</b><span>{rule.get('normal', '-')}</span></div>
            <div><b class='range-caution'>주의</b><span>{rule.get('caution', '-')}</span></div>
            <div><b class='range-bad'>위험</b><span>{rule.get('risk', '-')}</span></div>
            """
        return f"""
        <div><b class='range-good'>정상</b><span>{rule.get('normal', '-')}</span></div>
        <div><b class='range-bad'>위험</b><span>{rule.get('risk', '-')}</span></div>
        """


    def _bar_html(key, raw, level):
        meta = _bar_meta(key)
        if meta is None:
            return ""
        min_v, max_v = meta["min"], meta["max"]
        seg_html = ""
        for s, e, cls in meta["segments"]:
            left = _pct(s, min_v, max_v)
            width = max(0, _pct(e, min_v, max_v) - left)
            seg_html += f"<div class='ms-seg seg-{cls}' style='left:{left:.4f}%;width:{width:.4f}%;'></div>"

        tick_html = ""
        for t in meta["ticks"]:
            left = _pct(t, min_v, max_v)
            tick_html += f"<div class='ms-tick' style='left:{left:.4f}%;'></div><div class='ms-tick-label' style='left:{left:.4f}%;'>{_tick_label(key, t, meta['unit'])}</div>"

        if raw is None:
            marker_html = "<div class='ms-missing-marker'></div>"
        else:
            marker_left = _pct(raw, min_v, max_v)
            marker_html = f"<div class='ms-marker marker-{_level_info(level)['class']}' style='left:{marker_left:.4f}%;'></div>"

        return f"""
        <div class='ms-bar-wrap'>
            <div class='ms-bar'>
                {seg_html}
                {tick_html}
                {marker_html}
            </div>
        </div>
        """

    rows_html = ""
    counts = {"정상": 0, "주의": 0, "위험": 0, "제외": 0}

    for key in DISPLAY_METRIC_ORDER:
        if key not in all_data:
            continue
        value, is_good, raw = all_data[key]
        level = classify_posture_level(key, raw)
        counts[level] = counts.get(level, 0) + 1
        info = _level_info(level)
        display_value = _fmt_value(key, value, raw)
        standard_text = CLINICAL_RULES.get(key, {}).get("normal", "-")

        rows_html += f"""
        <div class='ms-row'>
            <div class='ms-item'>
                <div class='ms-icon icon-{info['class']}'>{_icon(key)}</div>
                <div>
                    <div class='ms-name'>{FEEDBACK[key]['label']} <span>({FEEDBACK[key]['eng']})</span></div>
                    <div class='ms-sub'>{_subtitle(key)}</div>
                </div>
            </div>
            <div class='ms-value-box'>
                <div class='ms-value' style='color:{info['color']};'>{display_value}</div>
                <div class='ms-standard'>기준 {standard_text}</div>
            </div>
            <div class='ms-status-graph'>
                {_bar_html(key, raw, level)}
            </div>
            <div class='ms-status-box'>
                <div class='ms-badge badge-{info['class']}'>{info['label']}</div>
                <div class='ms-status-kr'>{info['kr']}</div>
            </div>
            <div class='ms-range-box'>
                {_range_lines(key)}
            </div>
        </div>
        """

    good_count = counts.get("정상", 0)
    caution_count = counts.get("주의", 0)
    bad_count = counts.get("위험", 0)
    excluded_count = counts.get("제외", 0)

    logo_src = image_to_base64_src(Path(__file__).resolve().parent / "logo.png")
    if logo_src:
        logo_html = f"<img class='ms-main-logo' src='{logo_src}'>"
    else:
        logo_html = "<div class='ms-main-icon'>🧬</div>"

    html = f"""
    <html>
    <head>
    <style>
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0;
        padding: 0;
        font-family: Pretendard, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: transparent;
        color: #0F1F3A;
    }}
    .ms-report {{
        width: 100%;
        background: linear-gradient(180deg, #F7FAFF 0%, #FFFFFF 100%);
        border: 1px solid #E3EAF5;
        border-radius: 24px;
        padding: 22px 24px 14px 24px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    }}
    .ms-top {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        padding: 4px 0 22px 0;
    }}
    .ms-title-wrap {{
        display: flex;
        align-items: center;
        gap: 18px;
    }}
    .ms-main-icon {{
        width: 60px;
        height: 60px;
        border-radius: 18px;
        background: #EEF5FF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 34px;
        color: #1C64F2;
        box-shadow: inset 0 0 0 1px #D9E8FF;
    }}

    .ms-main-logo {{
        width: 60px;
        height: 60px;
        object-fit: contain;
        border-radius: 18px;
        background: #EEF5FF;
        padding: 8px;
        box-shadow: inset 0 0 0 1px #D9E8FF;
        box-sizing: border-box;
    }}
    .ms-title {{
        font-size: 30px;
        font-weight: 950;
        letter-spacing: -1.2px;
        color: #0B1B38;
        line-height: 1.1;
    }}
    .ms-desc {{
        margin-top: 8px;
        font-size: 15px;
        color: #58708F;
        font-weight: 500;
    }}
    .ms-summary {{
        min-width: 430px;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid #E4ECF7;
        border-radius: 18px;
        padding: 14px 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    }}
    .ms-sum-item {{
        display: flex;
        align-items: center;
        gap: 11px;
    }}
    .ms-sum-dot {{
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 20px;
        font-weight: 950;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.12);
    }}
    .sum-good {{ background: linear-gradient(135deg, #21A84C, #61D07E); }}
    .sum-caution {{ background: linear-gradient(135deg, #F3A600, #FFC247); }}
    .sum-bad {{ background: linear-gradient(135deg, #F43F5E, #FB7185); }}
    .ms-sum-num {{ font-size: 18px; font-weight: 950; color: #17233F; line-height: 1; }}
    .ms-sum-label {{ font-size: 12px; font-weight: 900; color: #304766; margin-top: 4px; }}
    .ms-sum-kr {{ font-size: 12px; color: #667A99; margin-top: 3px; }}
    .ms-table {{
        background: #FFFFFF;
        border: 1px solid #E4ECF7;
        border-radius: 18px;
        overflow: hidden;
    }}
    .ms-head, .ms-row {{
        display: grid;
        grid-template-columns: 1.35fr 0.72fr 1.75fr 0.42fr 0.92fr;
        align-items: center;
        gap: 18px;
    }}
    .ms-head {{
        height: 54px;
        padding: 0 22px;
        background: #FFFFFF;
        border-bottom: 1px solid #E6EDF7;
        color: #445D7F;
        font-size: 13px;
        font-weight: 900;
        text-align: center;
    }}
    .ms-head div:first-child {{ text-align: center; }}
    .ms-row {{
        min-height: 92px;
        padding: 12px 14px 12px 22px;
        border-bottom: 1px solid #EAF0F8;
        background: rgba(255,255,255,0.98);
    }}
    .ms-row:last-child {{ border-bottom: 0; }}
    .ms-item {{
        display: flex;
        align-items: center;
        gap: 16px;
        min-width: 0;
    }}
    .ms-icon {{
        width: 52px;
        height: 52px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        flex-shrink: 0;
    }}
    .ms-metric-img {{
        width: 38px;
        height: 38px;
        object-fit: contain;
        display: block;
        filter: drop-shadow(0 4px 7px rgba(15, 23, 42, 0.10));
    }}
    .icon-good {{ background: #E8F8EF; color: #2FB35A; }}
    .icon-caution {{ background: #FFF4D8; color: #F2A900; }}
    .icon-bad {{ background: #FFE4EA; color: #F43F5E; }}
    .icon-none {{ background: #F2F4F7; color: #98A2B3; }}
    .ms-name {{
        font-size: 17px;
        font-weight: 950;
        color: #0F1F3A;
        letter-spacing: -0.4px;
        white-space: nowrap;
    }}
    .ms-name span {{ color: #405877; font-size: 14px; font-weight: 850; }}
    .ms-sub {{ margin-top: 8px; font-size: 13px; color: #526986; font-weight: 600; }}
    .ms-value-box {{ text-align: center; }}
    .ms-value {{ font-size: 26px; font-weight: 950; letter-spacing: -0.6px; line-height: 1; }}
    .ms-standard {{ margin-top: 10px; color: #526986; font-size: 13px; font-weight: 700; }}
    .ms-status-graph {{ padding: 0 2px; }}
    .ms-bar-wrap {{ position: relative; height: 48px; }}
    .ms-bar {{
        position: relative;
        height: 8px;
        top: 15px;
        border-radius: 999px;
        background: #E7ECF5;
    }}
    .ms-seg {{ position: absolute; height: 8px; top: 0; }}
    .seg-good {{ background: #2FB35A; }}
    .seg-caution {{ background: #F2A900; }}
    .seg-bad {{ background: #F43F5E; }}
    .ms-seg:first-child {{ border-radius: 999px 0 0 999px; }}
    .ms-seg:last-of-type {{ border-radius: 0 999px 999px 0; }}
    .ms-marker {{
        position: absolute;
        top: -5px;
        width: 18px;
        height: 18px;
        border-radius: 50%;
        transform: translateX(-50%);
        border: 3px solid #FFFFFF;
        box-shadow: 0 4px 10px rgba(15,23,42,0.18);
        z-index: 5;
    }}
    .marker-good {{ background: #2FB35A; }}
    .marker-caution {{ background: #F2A900; }}
    .marker-bad {{ background: #F43F5E; }}
    .marker-none {{ background: #98A2B3; }}
    .ms-missing-marker {{
        position: absolute;
        left: 0;
        top: -1px;
        height: 10px;
        width: 56px;
        border-radius: 999px;
        background: #F43F5E;
        box-shadow: 0 3px 8px rgba(244,63,94,0.24);
    }}
    .ms-tick {{
        position: absolute;
        top: 15px;
        width: 1px;
        height: 13px;
        background: #B8C5D8;
        transform: translateX(-50%);
    }}
    .ms-tick-label {{
        position: absolute;
        top: 27px;
        transform: translateX(-50%);
        color: #546B8D;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }}
    .ms-status-box {{ text-align: center; }}
    .ms-badge {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 64px;
        height: 30px;
        border-radius: 10px;
        color: #FFFFFF;
        font-size: 14px;
        font-weight: 950;
        letter-spacing: -0.2px;
        box-shadow: 0 8px 16px rgba(15,23,42,0.10);
    }}
    .badge-good {{ background: linear-gradient(135deg, #24A84F, #53C874); }}
    .badge-caution {{ background: linear-gradient(135deg, #F2A900, #FFC247); }}
    .badge-bad {{ background: linear-gradient(135deg, #F43F5E, #FB7185); }}
    .badge-none {{ background: linear-gradient(135deg, #98A2B3, #CBD5E1); }}
    .ms-status-kr {{ margin-top: 7px; font-size: 13px; color: #526986; font-weight: 800; }}
    .ms-range-box {{
        border: 1px solid #E2EAF5;
        border-radius: 12px;
        padding: 10px 14px;
        background: #FFFFFF;
        font-size: 12.5px;
        line-height: 1.65;
        color: #4B6382;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.65);
    }}
    .ms-range-box div {{ display: grid; grid-template-columns: 42px 1fr; gap: 8px; }}
    .ms-range-box b {{ font-weight: 950; }}
    .range-good {{ color: #2FB35A; }}
    .range-caution {{ color: #F2A900; }}
    .range-bad {{ color: #F43F5E; }}
    .ms-tip {{
        margin-top: 12px;
        border: 1px solid #CFE1FF;
        background: linear-gradient(135deg, #F2F7FF, #FFFFFF);
        border-radius: 12px;
        padding: 13px 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        color: #20569B;
        font-size: 14px;
        font-weight: 800;
    }}
    .ms-legend {{ display: flex; align-items: center; gap: 16px; color: #445D7F; font-size: 13px; font-weight: 800; white-space: nowrap; }}
    .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: -1px; }}
    @media (max-width: 980px) {{
        .ms-top {{ flex-direction: column; align-items: stretch; }}
        .ms-summary {{ min-width: 0; }}
        .ms-head {{ display: none; }}
        .ms-row {{ grid-template-columns: 1fr; gap: 12px; }}
        .ms-range-box {{ max-width: none; }}
        .ms-tip {{ flex-direction: column; align-items: flex-start; }}
    }}
    </style>
    </head>
    <body>
        <div class='ms-report'>
            <div class='ms-top'>
                <div class='ms-title-wrap'>
                    {logo_html}
                    <div>
                        <div class='ms-title'>근골격계 측정 리포트</div>
                        <div class='ms-desc'>8가지 항목의 자세 및 환경을 종합적으로 분석했습니다.</div>
                    </div>
                </div>
                <div class='ms-summary'>
                    <div class='ms-sum-item'>
                        <div class='ms-sum-dot sum-good'>{good_count}</div>
                        <div><div class='ms-sum-num'>{good_count}</div><div class='ms-sum-label'>GOOD</div><div class='ms-sum-kr'>양호</div></div>
                    </div>
                    <div class='ms-sum-item'>
                        <div class='ms-sum-dot sum-caution'>{caution_count}</div>
                        <div><div class='ms-sum-num'>{caution_count}</div><div class='ms-sum-label'>CAUTION</div><div class='ms-sum-kr'>주의</div></div>
                    </div>
                    <div class='ms-sum-item'>
                        <div class='ms-sum-dot sum-bad'>{bad_count}</div>
                        <div><div class='ms-sum-num'>{bad_count}</div><div class='ms-sum-label'>BAD</div><div class='ms-sum-kr'>위험</div></div>
                    </div>
                </div>
            </div>

            <div class='ms-table'>
                <div class='ms-head'>
                    <div>항목</div>
                    <div>측정값</div>
                    <div>상태</div>
                    <div></div>
                    <div>기준 범위</div>
                </div>
                {rows_html}
            </div>

            <div class='ms-tip'>
                <div>💡 <b>TIP</b>&nbsp;&nbsp; 빨간색 항목부터 우선적으로 교정하는 것이 자세 개선에 효과적입니다.</div>
                <div class='ms-legend'>
                    <span><span class='legend-dot' style='background:#2FB35A;'></span>GOOD(양호)</span>
                    <span><span class='legend-dot' style='background:#F2A900;'></span>CAUTION(주의)</span>
                    <span><span class='legend-dot' style='background:#F43F5E;'></span>BAD(위험)</span>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=980, scrolling=True)

    if excluded_count > 0:
        st.caption(f"※ 기준점 또는 사물 인식이 부족한 {excluded_count}개 항목은 그래프에서 제외로 표시됩니다.")


def get_korean_pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", font_path))
                return "KoreanFont"
            except Exception:
                pass

    return "Helvetica"


def clean_pdf_text(text):
    if text is None:
        return "-"
    return str(text).replace("<br>", "\n").replace("·", "-").replace("→", "->")


def make_musculoskeletal_report_pdf(result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    buffer = BytesIO()
    font_name = get_korean_pdf_font()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="KTitle",
            fontName=font_name,
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#172033"),
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSub",
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            textColor=colors.HexColor("#667085"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="KBody",
            fontName=font_name,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#172033"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSmall",
            fontName=font_name,
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor("#667085"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="KSection",
            fontName=font_name,
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#172033"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    story = []

    all_data = {**result.get("posture", {}), **result.get("env", {})}
    level_counts = result.get("level_counts", {})
    username = st.session_state.get("username", "익명")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    story.append(Paragraph("Fit Me Up 근골격계 리포트", styles["KTitle"]))
    story.append(
        Paragraph(
            f"사용자: {username}  |  생성일시: {now}<br/>"
            "본 리포트는 AI 자세 분석 기반 참고 자료이며, 의료 진단을 대체하지 않습니다.",
            styles["KSub"],
        )
    )

    summary_data = [
        ["종합 점수", "종합 위험도", "정상 지표", "관리 필요 지표"],
        [
            f"{result.get('score', 0)}/10",
            result.get("risk", "-"),
            f"{result.get('good_count', 0)}개",
            f"{result.get('total_count', 0) - result.get('good_count', 0)}개",
        ],
    ]

    summary_table = Table(summary_data, colWidths=[40 * mm, 40 * mm, 40 * mm, 40 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#667085")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#172033")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, 1), 15),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E5EAF2")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. 부위별 측정 결과", styles["KSection"]))

    metric_rows = [["번호", "지표", "측정값", "판정", "판정 기준"]]

    for idx, key in enumerate(DISPLAY_METRIC_ORDER, start=1):
        if key not in all_data:
            continue

        value, _, raw = all_data[key]
        level = classify_posture_level(key, raw)
        rule = CLINICAL_RULES.get(key, {})

        range_text = get_range_text_html(key, line_break="<br/>")

        metric_rows.append(
            [
                str(idx),
                FEEDBACK[key]["label"],
                value,
                level,
                Paragraph(range_text, styles["KSmall"]),
            ]
        )

    metric_table = Table(
        metric_rows,
        colWidths=[12 * mm, 30 * mm, 25 * mm, 22 * mm, 78 * mm],
        repeatRows=1,
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172033")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (3, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5EAF2")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(metric_table)

    story.append(PageBreak())
    story.append(Paragraph("2. 상세 피드백", styles["KSection"]))

    for key in DISPLAY_METRIC_ORDER:
        if key not in all_data:
            continue

        value, is_good, raw = all_data[key]
        level = classify_posture_level(key, raw)
        fb = FEEDBACK[key]
        msg = fb["good"] if level == "정상" else fb["bad"]
        rule = CLINICAL_RULES.get(key, {})

        color = "#45B86B" if level == "정상" else "#7467F0" if level == "주의" else "#F2527D" if level == "위험" else "#AEB6C2"

        block = Table(
            [
                [
                    Paragraph(
                        f"<b>{fb['no']}. {fb['label']} ({fb['eng']})</b>",
                        styles["KBody"],
                    ),
                    Paragraph(f"<b>{level}</b>", styles["KBody"]),
                ],
                [
                    Paragraph(
                        f"측정값: {value}<br/>"
                        f"{get_range_text_html(key, line_break='<br/>')}",
                        styles["KSmall"],
                    ),
                    "",
                ],
                [
                    Paragraph(clean_pdf_text(msg).replace("\n", "<br/>"), styles["KBody"]),
                    "",
                ],
            ],
            colWidths=[135 * mm, 28 * mm],
        )

        block.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("SPAN", (0, 1), (1, 1)),
                    ("SPAN", (0, 2), (1, 2)),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7F8") if level != "정상" else colors.HexColor("#F0FBF4")),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(color)),
                    ("LINEBEFORE", (0, 0), (0, -1), 4, colors.HexColor(color)),
                    ("ALIGN", (1, 0), (1, 0), "CENTER"),
                    ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor(color)),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )

        story.append(block)
        story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer

# =========================================================
# 8. 실제 페이지 출력부 — 잔상 방지 핵심
# =========================================================

init_history()

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
page_placeholder = st.empty()

if "challenge_times" not in st.session_state:
    st.session_state.challenge_times = []

render_alarm_effect(st.session_state.challenge_times)

with page_placeholder.container():

    if menu == "📸 자세측정":
        render_measure()

    elif menu == "📈 측정이력":
        render_history()

    elif menu == "📄 근골격계 리포트":
        render_report()

    elif menu == "🧾 예상 영수증":
        render_receipt_page()

    elif menu == "🎯 바른자세 챌린지":
        render_posture_challenge()
    elif menu == "🛒 제품 추천":
        render_product_recommendation_page()
