<!--
████████████████████████████████████████████████████████████████████
  seungwoo2000 · Fit_me_up — K-디지털 트레이닝 포트폴리오 README
████████████████████████████████████████████████████████████████████
-->

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e1b4b,50:4f46e5,100:818cf8&height=220&section=header&text=🧘%20자세히봐&fontSize=60&fontColor=ffffff&fontAlignY=40&desc=AI로%20자세와%20작업환경을%20분석해%20개선%20가이드를%20제공합니다&descAlignY=62&descColor=c7d2fe&animation=fadeIn" alt="header" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-환경탐지-00CFDD?style=for-the-badge)
![MediaPipe](https://img.shields.io/badge/MediaPipe-자세분석-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_2.5.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA_12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

<br/>

> **앉은 자세 + 작업 환경을 이미지로 분석해 교정 및 개선 가이드를 제공하는 AI 시스템**  
> CNN · MediaPipe · YOLOv8 3개 모델을 통합한 컴퓨터 비전 프로젝트

</div>

---

## 📋 훈련 과정 정보

| 항목 | 내용 |
|:---|:---|
| 🏫 **훈련기관** | 아시아경제 교육센터 |
| 📚 **훈련과정명** | 융합\_데이터 기반 차세대 디지털 헬스케어 AI 솔루션 5회차 |
| 🏷️ **훈련유형** | K-디지털 트레이닝 (고용노동부) |
| 📅 **훈련기간** | 2026-02-03 ~ 2026-07-30 (6개월) |
| 💡 **프로젝트 분류** | 컴퓨터 비전 · 헬스케어 AI · 자세·작업환경 분석 |

---

## 🧘 프로젝트 소개

```
"장시간 앉아서 일하는 현대인의 잘못된 자세와 작업 환경, AI가 한 번에 분석해줍니다."

카메라 하나로 자세를 분석하고, 책상·의자·모니터 위치까지 파악해
RULA / VDT 기준으로 근골격계 위험도를 점수화합니다.
```

**자세히봐**는 사용자의 **앉은 자세**와 **주변 작업 환경**(책상·의자·모니터)을  
동시에 분석하여 **자세 교정 + 작업환경 개선** 가이드를 제공하는 컴퓨터 비전 헬스케어 프로젝트입니다.

| 🔑 키워드 | 설명 |
|:---|:---|
| **RULA** | 작업자의 자세 위험도를 평가하는 국제 표준 인간공학 지표 |
| **VDT 증후군** | 컴퓨터 장시간 사용으로 인한 눈·목·어깨 통증 증후군 |
| **컴퓨터 비전** | 카메라 영상을 AI가 분석해 의미 있는 정보를 추출하는 기술 |
| **실시간 추론** | 영상을 저장하지 않고 즉시 분석해 피드백을 주는 방식 |

---

## 🗂️ 목차

1. [분석 파이프라인](#-분석-파이프라인)
2. [기술 스택](#-기술-스택)
3. [주요 기능](#-주요-기능)
4. [파일 구조](#-파일-구조)
5. [배운 점 · 성장 포인트](#-배운-점--성장-포인트)
6. [실행 방법](#-실행-방법)

---

## 🔄 분석 파이프라인

```
이미지 입력 (사진 업로드 / 촬영)
        ↓
┌───────────────────────────────────┐
│  ① CNN (MobileNetV2)             │  자세 Good / Bad 1차 판정
│  ② MediaPipe Pose                │  신체 관절 17개 좌표 추출 → 각도 분석
│  ③ YOLOv8                        │  책상·의자·모니터 위치 탐지
└───────────────────────────────────┘
        ↓
④ 통합 분석   →  관절 좌표 + 객체 위치 결합
        ↓
⑤ RULA / VDT 점수 산출
        ↓
⑥ 화면 오버레이 시각화 + 교정 피드백 출력
```

---

## 🛠️ 기술 스택

> 비전공자도 이해할 수 있도록, 각 기술이 **어떤 역할**을 하는지 함께 설명합니다.

### 🤖 AI · 컴퓨터 비전

| 기술 | 역할 | 핵심 키워드 |
|:---:|:---|:---|
| **MediaPipe Pose** | 사람의 관절 17개 좌표를 실시간으로 추출 | `Landmarks` `관절 각도` `실시간` |
| **YOLOv8** | 카메라 화면에서 책상·의자·모니터를 탐지 | `객체 탐지` `바운딩 박스` `환경 분석` |
| **CNN (MobileNetV2)** | 이미지를 보고 자세가 Good인지 Bad인지 판정 | `이미지 분류` `전이학습` |
| **MLP** | 관절 좌표 데이터로 자세를 추가 분류 | `다층 퍼셉트론` `좌표 기반 분류` |

### ⚙️ 개발 환경

| 항목 | 내용 |
|:---:|:---|
| **언어** | Python 3.11 |
| **딥러닝** | PyTorch 2.5.1 · TensorFlow |
| **GPU** | NVIDIA GeForce RTX 4060 (CUDA 12.1) |
| **OS** | Windows 10 / 11 |

---

## ✨ 주요 기능

```
👁️  이미지 자세 분석      →  업로드한 이미지에서 앉은 자세를 Good / Bad 판정
🦴  관절 각도 분석        →  목·어깨·허리·무릎 각도를 수치로 측정
🪑  작업 환경 탐지        →  책상·의자·모니터 위치를 YOLOv8으로 인식
📊  RULA / VDT 점수       →  국제 표준 지표로 근골격계 위험도를 점수화
💬  교정·환경 피드백      →  자세 교정 + 작업환경(책상·의자·모니터) 개선 방향 안내
📱  Streamlit 앱          →  app.py 실행으로 웹 UI에서 바로 사용
🏆  챌린지 기록           →  challenge_results.json으로 기록 저장
```

---

## 📁 파일 구조

```
Fit_me_up/
│
├── 📂 MediaPipe/              # 자세 분석 모듈 (Postural Analysis)
│   ├── code/                  # 추론 API · 테스트 UI · 학습 스크립트
│   ├── models/                # CNN(MobileNetV2) · MLP 모델 파일
│   └── data/                  # 학습용 관절 좌표 데이터셋
│
├── 📂 YOLO/                   # 환경 탐지 모듈 (Environment Detection)
│   ├── fit_me_up/
│   │   └── combined_gpu/
│   │       └── weights/
│   │           └── best.pt    # ✅ 학습 완료 모델 (클론 후 즉시 테스트 가능)
│   ├── run_all.py             # 자동 학습 및 전략 비교 스크립트
│   └── yolo_test.py           # YOLO 추론 테스트 UI
│
├── app.py                     # 🌐 Streamlit 웹 대시보드
├── integrate.py               # 🔗 통합 파이프라인
├── challenge_results.json     # 챌린지 기록 저장
├── user_history.json          # 사용자 분석 이력
├── users.json                 # 사용자 정보
├── logo.png                   # 서비스 로고
└── requirements.txt           # 패키지 의존성 목록
```

---

## 📈 배운 점 · 성장 포인트

| 분야 | 배운 것 | 이걸 배워서 뭘 할 수 있게 됐나? |
|:---|:---|:---|
| 👁️ **컴퓨터 비전** | MediaPipe · YOLOv8 · CNN 통합 | 단일 모델이 아닌 여러 AI를 결합해 자세·환경을 함께 분석 |
| 🦴 **자세 분석** | 관절 좌표 → 각도 계산 → 위험도 판정 | 숫자 좌표를 인체공학 지표로 변환하는 파이프라인 설계 |
| 🎯 **객체 탐지** | YOLOv8 커스텀 학습 (best.pt) | 원하는 객체(책상·의자·모니터)를 직접 탐지하도록 훈련 |
| 🔗 **모델 통합** | integrate.py 파이프라인 설계 | 독립적인 모듈 3개를 하나의 흐름으로 연결 |
| 🖥️ **GPU 활용** | CUDA 12.1 · RTX 4060 환경 구성 | 딥러닝 학습 속도를 GPU로 가속하는 환경 세팅 |
| 🏥 **헬스케어 연계** | RULA · VDT 지표 적용 | 국제 표준 인간공학 지표를 AI 출력과 연결 |

---

## ⚙️ 실행 방법

**1️⃣ 가상환경 세팅 및 패키지 설치**
```bash
# 가상환경 생성 및 활성화 (Windows)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 필수 패키지 설치
pip install -r requirements.txt
```

**2️⃣ 모듈별 독립 테스트**
```bash
# YOLO 환경 탐지 테스트 (학습된 best.pt 즉시 사용 가능)
python YOLO/yolo_test.py

# 자세 분석 (CNN / MediaPipe) 테스트
python MediaPipe/code/posture_test.py
```

**3️⃣ Streamlit 웹 앱 실행**
```bash
streamlit run app.py
```

> ✅ Python 3.11 · CUDA 12.1 환경 권장  
> ✅ `YOLO/fit_me_up/combined_gpu/weights/best.pt` 포함 — 별도 학습 없이 즉시 테스트 가능

---

<div align="center">

<br/>

*"이미지 하나로 자세와 작업환경을 동시에 분석하고, 구체적인 개선 방향을 제시합니다."* 🧘

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:818cf8,50:4f46e5,100:1e1b4b&height=130&section=footer&text=K-디지털%20트레이닝%20|%20아시아경제%20교육센터&fontSize=15&fontColor=ffffff&fontAlignY=65" width="100%"/>

**📅 2026.02 ~ 2026.07** &nbsp;|&nbsp; Made with 🧘 during K-Digital Training

</div>
