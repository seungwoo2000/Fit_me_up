# 딥러닝 기반 측면 이미지를 활용한 VDT 취급 근로자의 앉은 자세와 작업 환경 교정 및 지속 관리 서비스
## Fit Me Up

> **근거 법령**: VDT 취급근로자 작업관리지침 제2020-17호 · RULA · 산업안전보건법

---

## 📌 프로젝트 개요

측면 이미지 한 장으로 VDT 근로자의 **8개 자세 지표**를 자동 분석하고 교정 가이드를 제공하는 AI 서비스입니다.

- **개발 기간**: 2026.04.02 ~ 2026.04.30
- **기술 스택**: YOLOv8 · MediaPipe · MobileNetV2 · Streamlit · Python

---

## 👥 팀원

| 역할 | 이름 |
|------|------|
| AA | 신나은 |
| DA | 김아름, 조예원 |
| TA | 안승우, 홍지연 |

---

## 📁 프로젝트 구조

```
FIT_ME_UP/
├── README.md
├── .gitignore
│
├── YOLO/                                        # Step 1: 객체 탐지 (의자·책상·모니터)
│   ├── run_all.py                               # CPU+GPU 학습 자동 실행 & 비교표 출력
│   ├── split_images.py                          # 데이터셋 분할 (train 80 / val 10 / test 10)
│   ├── train_full_body.py                       # full_body 데이터 학습
│   ├── train_ankle_visible.py                   # ankle_visible 데이터 학습
│   ├── train_combined.py                        # 통합 데이터 학습 (Best 모델)
│   ├── check_labels.py                          # 라벨 시각화 확인
│   ├── data_full_body.yaml                      # full_body 데이터셋 설정
│   ├── data_ankle_visible.yaml                  # ankle_visible 데이터셋 설정
│   ├── data_combined.yaml                       # 통합 데이터셋 설정
│   ├── yolov8n.pt                               # 사전학습 모델 ← 별도 다운로드 필요
│   ├── YOLO_full_body_Labeling/
│   │   ├── images/                              # 원본 이미지 ← gitignore
│   │   ├── labels/                              # 라벨 파일
│   │   └── split_data/                          # 분할 결과 ← gitignore
│   ├── YOLO_ankle_visible_Labeling/
│   │   ├── images/                              # 원본 이미지 ← gitignore
│   │   ├── labels/
│   │   └── split_data/                          # 분할 결과 ← gitignore
│   └── fit_me_up/                               # 학습 결과 폴더
│       ├── full_body_only_cpu/
│       ├── full_body_only_gpu/
│       ├── ankle_visible_only_cpu/
│       ├── ankle_visible_only_gpu/
│       ├── combined_cpu/
│       └── combined_gpu/
│           └── weights/best.pt                  # ← Streamlit 실행에 필요
│
├── MediaPipe/                                   # Step 2: 자세 Good/Bad 분류
│   ├── code/
│   │   ├── predict.py                           # YOLO 연동용 함수형 API
│   │   ├── posture_test.py                      # GUI 단독 테스트 앱
│   │   ├── train_mobilenetv2.py                 # MobileNetV2 학습 파이프라인
│   │   └── requirements.txt
│   ├── data/
│   │   ├── images/                              # 학습 데이터 ← gitignore
│   │   ├── labeled_dataset_v3.csv
│   │   └── models/                              # MediaPipe .task ← gitignore (자동 다운로드)
│   ├── models/
│   │   └── posture_mobilenetv2_finetuned.h5     # ← Streamlit 실행에 필요
│   └── reports/                                 # 성능 시각화 자료
│
├── integrate.py                                 # Step 3: YOLO + MediaPipe 결합 분석
└── app.py                                       # Step 4: Streamlit 서비스
```

---

## 🚀 빠른 시작 (git clone 후 순서)

### 1단계 — 환경 설정

```bash
cd Fit_me_up
py -3.11 -m venv .venv
.venv\Scripts\activate

pip install ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install mediapipe tensorflow opencv-python pillow split-folders streamlit streamlit-autorefresh
```

### 2단계 — YOLO 사전학습 모델 다운로드

[yolov8n.pt 다운로드](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) → `YOLO/` 폴더에 저장

### 3단계 — 데이터 분할

```bash
cd YOLO
python split_images.py
```

### 4단계 — YOLO 학습 (CPU + GPU 자동 비교)

```bash
python run_all.py
```

> GPU(RTX 4060) 기준 약 10분, CPU 기준 약 1시간 40분

### 5단계 — MediaPipe 단독 테스트

```bash
cd ..\MediaPipe\code
python posture_test.py
```

> 최초 실행 시 인터넷 연결 필요 (MediaPipe .task 모델 자동 다운로드)

### 6단계 — YOLO + MediaPipe 통합 분석

```bash
cd ..\..
python integrate.py
```

### 7단계 — Streamlit 실행

```bash
streamlit run app.py
```

---

## 🤖 Step 1 — YOLO 객체 탐지

의자(chair) · 책상(desk) · 모니터(monitor) 3개 클래스를 탐지합니다.

### 데이터셋

| 구분 | 이미지 수 | train | val | test |
|------|-----------|-------|-----|------|
| full_body | 244장 | 195 | 24 | 25 |
| ankle_visible | 105장 | 84 | 10 | 11 |
| **combined** | **349장** | **279** | **34** | **36** |

### CPU vs GPU 학습 결과

| 모델 | CPU 소요시간 | GPU 소요시간 | 단축률 |
|------|------------|------------|-------|
| full_body_only | 0h 30m 3s | - | - |
| ankle_visible_only | 0h 7m 36s | - | - |
| combined | 1h 5m 41s | - | - |

### 성능 지표 (CPU 기준)

| 모델 | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| full_body_only | 0.9776 | 0.8436 | 0.9736 | 0.9453 |
| ankle_visible_only | 0.8824 | 0.6699 | 0.9625 | 0.8039 |
| **combined** ✅ | **0.9815** | **0.8306** | **0.9646** | **0.9148** |

**Best 모델**: `combined` (mAP50: 0.9815) — 파인튜닝 불필요

### 하이퍼파라미터 (YOLOv8 공식 권장값)

| epochs | patience | batch | imgsz | optimizer | lr0 | lrf |
|--------|----------|-------|-------|-----------|-----|-----|
| 100 | 20 | 16 | 640 | SGD | 0.01 | 0.001 |

---

## 🦴 Step 2 — MediaPipe 자세 분류

MobileNetV2 전이학습으로 측면 이미지를 **Good / Bad** 이진 분류합니다.

| 지표 | 값 |
|------|----|
| Accuracy | 95.00% |
| Bad Recall | 95.0% |
| 판정 임계값 | 0.60 |
| 모델 | MobileNetV2 + 상위 30레이어 Fine-tuning |

```python
from predict import predict_posture

result = predict_posture("side_view.jpg")
print(result["label"])       # 'good' or 'bad'
print(result["confidence"])  # 0.0 ~ 1.0
```

---

## 🔗 Step 3 — integrate.py (YOLO + MediaPipe 결합)

```bash
python integrate.py
```

- YOLO로 의자·책상·모니터 탐지
- MediaPipe로 33개 관절 랜드마크 추출
- **8개 자세 지표** 자동 산출 (VDT 고시 + RULA 기준)
- 결과 이미지 저장 (`result.jpg`, 2560×1440)

### 8개 자세 지표

| No | 지표 | 정상 범위 | 법령 근거 |
|----|------|-----------|-----------|
| 01 | 목굴곡각 (CVA) | 0° ~ 20° | VDT 고시 제6조 1항 |
| 02 | 몸통굴곡각 (TIA) | 0° ~ 10° | VDT 고시 제6조 4항 |
| 03 | 팔꿈치 각도 | 90° ~ 120° | VDT 고시 제6조 2항 |
| 04 | 무릎 각도 | 85° ~ 100° | VDT 고시 제6조 6항 |
| 05 | 손목 편위각 | ±15° 이내 | VDT 고시 제6조 7항 |
| 06 | 모니터 시선각 | 하방 10° ~ 15° | VDT 고시 제6조 1항 |
| 07 | 작업대 높이 | 팔꿈치 수평 ±10% | VDT 고시 제5조 3항 |
| 08 | 의자 등받이 | 골반너비 20% 이내 | VDT 고시 제5조 4항 |

---

## 🖥️ Step 4 — Streamlit 서비스

```bash
streamlit run app.py
```

| 메뉴 | 기능 |
|------|------|
| 대시보드 | 최근 분석 결과 요약 |
| 자세측정 | 측면 사진 업로드 → AI 분석 |
| 측정이력 | 과거 측정 결과 확인 |
| 근골격계 리포트 | 부위별 위험도 분석 |
| 예상 영수증 | 비급여 의료비 예상 비용 |
| 바른자세 챌린지 | 팀 알림 & 랭킹 |
