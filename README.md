# 딥러닝 기반 측면 이미지를 활용한 영상표시단말기(VDT) 취급 근로자의 앉은 자세와 작업 환경 교정 및 지속 관리 서비스 개발
## Fit Me Up

---

## 📌 프로젝트 개요

VDT(영상표시단말기) 취급 근로자의 측면 이미지를 딥러닝으로 분석하여 앉은 자세와 작업 환경을 교정하고 지속 관리하는 서비스입니다.

- **개발 기간**: 2026.04.02 ~ 2026.04.30
- **기술 스택**: YOLOv8, MediaPipe, Python

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
├── YOLO/                                       # 객체 탐지 (의자, 책상, 모니터)
│   ├── run_all.py                              # 학습 전체 실행 + 결과 비교표 출력
│   ├── split_images.py                         # 데이터셋 분할 (80/10/10)
│   ├── train_full_body.py                      # full_body 데이터 학습
│   ├── train_ankle_visible.py                  # ankle_visible 데이터 학습
│   ├── train_combined.py                       # 통합 데이터 학습
│   ├── check_labels.py                         # 라벨 시각화 확인
│   ├── data_full_body.yaml                     # full_body 데이터셋 설정
│   ├── data_ankle_visible.yaml                 # ankle_visible 데이터셋 설정
│   ├── data_combined.yaml                      # 통합 데이터셋 설정
│   ├── yolov8n.pt                              # 사전학습 모델 (gitignore)
│   ├── YOLO_full_body_Labeling/
│   │   ├── images/
│   │   ├── labels/
│   │   └── split_data/                         # (gitignore)
│   ├── YOLO_ankle_visible_Labeling/
│   │   ├── images/
│   │   ├── labels/
│   │   └── split_data/                         # (gitignore)
│   └── fit_me_up/                              # 학습 결과 (gitignore)
│       ├── full_body_only_gpu/
│       ├── ankle_visible_only_gpu/
│       └── combined_gpu/
└── MediaPipe/                                  # 자세 분석 (추가 예정)
```

---

## 🤖 YOLO - 객체 탐지

작업 환경 내 **의자(chair), 책상(desk), 모니터(monitor)** 를 탐지합니다.

### 탐지 클래스

| ID | 클래스 |
|----|--------|
| 0 | chair |
| 1 | desk |
| 2 | monitor |

### 데이터셋 구성

| 구분 | 설명 |
|------|------|
| full_body | 전신이 보이는 이미지 |
| ankle_visible | 발목까지 보이는 이미지 |
| combined | 두 데이터셋 통합 |

### 실행 순서

**1. 사전학습 모델 다운로드**

[YOLOv8n 다운로드](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) 후 `YOLO/` 폴더에 저장

**2. 데이터 분할**

```bash
cd YOLO
python split_images.py
```

**3. 학습 실행 + 결과 비교**

```bash
python run_all.py
```

### 학습 결과

> GPU(RTX 4060) 기준 총 소요시간: 약 10분

| 모델 | mAP50 | Precision | Recall | 소요시간 |
|------|-------|-----------|--------|----------|
| full_body_only | 0.9658 | 0.9696 | 0.9558 | 3분 50초 |
| ankle_visible_only | 0.9246 | 0.8450 | 0.8667 | 1분 22초 |
| **combined** ✅ | **0.9792** | **0.9751** | **0.9191** | 5분 15초 |

**Best 모델**: `combined_gpu` (mAP50: 0.9792)

### 하이퍼파라미터

| 파라미터 | 값 |
|----------|----|
| epochs | 100 |
| patience | 20 |
| batch | 16 |
| imgsz | 640 |
| optimizer | SGD |
| lr0 | 0.01 |
| lrf | 0.001 |

---

## 🦴 MediaPipe - 자세 분석

*(추가 예정)*

---

## ⚙️ 환경 설정

```bash
# 가상환경 생성
python -m venv .venv
.venv\Scripts\activate

# 패키지 설치
pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python
```
