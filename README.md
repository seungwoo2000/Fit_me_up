# Fit_me_up 🧘

앉은 자세를 분석하여 **Good / Bad** 판정과 개선 피드백을 제공하는 컴퓨터 비전 프로젝트입니다.

---

## 프로젝트 구조

```
Fit_me_up/
├── MediaPipe/                        # 자세 분석 모듈
│   ├── code/
│   │   ├── predict.py                # CNN 추론 (YOLO 연동용 API)
│   │   ├── posture_test.py           # CNN + MediaPipe UI 테스트 앱
│   │   ├── train_mobilenetv2.py      # CNN(MobileNetV2) 학습
│   │   ├── train_mlp.py              # MLP 학습 (관절 좌표 기반)
│   │   └── requirements.txt
│   ├── data/
│   │   └── labeled_dataset_v3.csv    # MLP 학습용 관절 좌표 데이터
│   ├── models/
│   │   ├── posture_mobilenetv2_finetuned.h5   # 학습된 CNN 모델
│   │   ├── posture_mlp_final.pkl              # 학습된 MLP 모델
│   │   └── pose_landmarker.task               # MediaPipe 모델
│   └── reports/                      # 학습 결과 그래프 및 리포트
│
├── YOLO/                             # 환경 탐지 모듈
│   ├── YOLO_full_body_Labeling/      # 전신 이미지 데이터셋
│   │   ├── images/
│   │   ├── labels/
│   │   └── split_data/               # split_images.py 실행 후 생성
│   ├── YOLO_ankle_visible_Labeling/  # 발목 보이는 이미지 데이터셋
│   │   ├── images/
│   │   ├── labels/
│   │   └── split_data/               # split_images.py 실행 후 생성
│   ├── fit_me_up/                    # 학습 결과 저장 (run_all.py 실행 후 생성)
│   │   ├── full_body_only_gpu/
│   │   │   └── weights/best.pt
│   │   ├── ankle_visible_only_gpu/
│   │   │   └── weights/best.pt
│   │   └── combined_gpu/
│   │       └── weights/best.pt       # ← 통합 연동에 사용할 모델
│   ├── data_full_body.yaml
│   ├── data_ankle_visible.yaml
│   ├── data_combined.yaml
│   ├── split_images.py               # 데이터 분할 (train 80/val 10/test 10)
│   ├── train_full_body.py            # full_body 단독 학습
│   ├── train_ankle_visible.py        # ankle_visible 단독 학습
│   ├── train_combined.py             # 통합 학습 (권장)
│   ├── run_all.py                    # 3가지 전략 자동 실행 & 비교표 출력
│   └── yolov8n.pt                    # YOLOv8 사전학습 가중치
│
└── integrate/                        # 통합 파이프라인 (추후 작성)
    └── (작성 예정)
```

---

## 분석 파이프라인

```
입력 이미지
    │
    ├── ① CNN (MobileNetV2)     → 자세 Good/Bad 1차 분류
    │        └── CVA·TIA 2개 지표 중 하나라도 Bad → 조기 피드백 출력
    │            둘 다 Good → 8개 지표 전체 판정
    │
    ├── ② MediaPipe             → 앉은 자세 관절 좌표 추출
    │
    ├── ③ YOLO                  → 환경 객체 탐지 (의자·책상·모니터)
    │
    ├── ④ 자세 + 환경 통합       → MediaPipe × YOLO 결합 분석
    │
    ├── ⑤ RULA + VDT 고시       → 관절각·거리 기반 점수 계산
    │
    └── ⑥ 시각화                → Good/Bad 포인트 오버레이 + 피드백 출력
```

---

## 모듈별 설명

### MediaPipe 모듈
| 파일 | 설명 |
|------|------|
| `predict.py` | CNN 추론 API. `predict_posture(image_path)` 호출 시 `{"label": "good/bad", "confidence": 0.87}` 반환. YOLO 크롭 이미지 입력 가능 |
| `posture_test.py` | Tkinter 기반 UI 테스트 앱. 이미지 선택 → CNN 판정 + MediaPipe TIA 각도 시각화 |
| `train_mobilenetv2.py` | MobileNetV2 전이학습. Stage1(헤드 학습) → Stage2(파인튜닝) 2단계 학습 |
| `train_mlp.py` | 관절 좌표 기반 MLP 분류기 학습. `labeled_dataset_v3.csv` 사용 |

### YOLO 모듈
| 파일 | 설명 |
|------|------|
| `split_images.py` | 이미지·라벨을 train/val/test(8:1:1)로 분할 |
| `run_all.py` | 3가지 학습 전략을 순차 실행하고 평가지표 비교표 출력 |
| `train_full_body.py` | 전신 이미지만으로 학습 |
| `train_ankle_visible.py` | 발목 보이는 이미지만으로 학습 |
| `train_combined.py` | 두 데이터셋 통합 학습 (권장) |

**탐지 클래스:** `chair` / `desk` / `monitor`

---

## 실행 방법

### 1. 환경 설정

```bash
cd MediaPipe/code
pip install -r requirements.txt
```

### 2. YOLO 학습

```bash
cd YOLO

# 데이터 분할
python split_images.py

# 3가지 전략 자동 학습 + 비교표 출력 (GPU 자동 감지)
python run_all.py

# 단일 전략만 학습할 경우
python train_combined.py
```

학습 완료 후 `YOLO/fit_me_up/combined_gpu/weights/best.pt` 생성됨

### 3. MediaPipe 테스트

```bash
cd MediaPipe/code
python posture_test.py
```

### 4. CNN 추론 (단독 실행)

```bash
cd MediaPipe/code
python predict.py <이미지경로>
```

---

## 평가지표

| 지표 | 설명 |
|------|------|
| mAP50 ↑ | IoU 0.5 기준 평균 정밀도 (높을수록 좋음) |
| mAP50-95 ↑ | IoU 0.5~0.95 평균 정밀도 (엄격한 기준) |
| Precision ↑ | 탐지한 것 중 실제 정답 비율 |
| Recall ↑ | 실제 정답 중 탐지한 비율 |
| Box Loss ↓ | 바운딩박스 위치 손실 (낮을수록 좋음) |
| Cls Loss ↓ | 클래스 분류 손실 (낮을수록 좋음) |

---

## 개발 환경

- Python 3.11
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics YOLOv8
- TensorFlow / Keras
- MediaPipe
- GPU: NVIDIA GeForce RTX 4060

---

## TODO

- [ ] YOLO 학습 완료 → `best.pt` 생성
- [ ] MediaPipe 모듈 완성
- [ ] YOLO + MediaPipe + CNN 통합 파이프라인 작성 (`integrate/`)
- [ ] RULA + VDT 고시 기반 점수 계산 구현
- [ ] 결과 시각화 (Good/Bad 포인트 오버레이)
