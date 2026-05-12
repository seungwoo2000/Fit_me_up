# 🧘 자세히봐

사용자의 앉은 자세와 주변 작업 환경(책상, 의자, 모니터)을 통합적으로 분석하여 실시간 교정 피드백을 제공하는 컴퓨터 비전 프로젝트입니다.

---

## 📂 프로젝트 구조

```text
Fit_me_up/
├── MediaPipe/                # 자세 분석 모듈 (Postural Analysis)
│   ├── code/                 # 추론 API, 테스트 UI, 학습 스크립트
│   ├── models/               # CNN(MobileNetV2), MLP 모델 파일
│   └── data/                 # 학습용 관절 좌표 데이터셋
├── YOLO/                     # 환경 탐지 모듈 (Environment Detection)
│   ├── fit_me_up/            # 학습 결과 저장소
│   │   └── combined_gpu/
│   │       └── weights/
│   │           └── best.pt   # ← [공유 모델] 클론 후 즉시 테스트 가능
│   ├── run_all.py            # 자동 학습 및 전략 비교 스크립트
│   └── yolo_test.py          # YOLO 추론 테스트 UI (best.pt 사용)
├── integrate/                # 통합 파이프라인 (작성 예정)
└── requirements.txt          # 프로젝트 공통 의존성 관리
🚀 빠른 시작 (Quick Start)
팀원들은 별도의 YOLO 학습 과정 없이, 포함된 best.pt 모델을 사용하여 즉시 기능을 테스트할 수 있습니다.

1. 가상환경 세팅 및 라이브러리 설치
Bash
# 가상환경 생성 및 활성화 (Windows)
py -3.11 -m venv .venv
.venv\Scripts\activate

# 필수 패키지 설치 (GPU 환경 대응 버전)
pip install -r requirements.txt
2. 모듈별 독립 테스트
YOLO 환경 탐지: python YOLO/yolo_test.py 실행 후 이미지 선택

자세 분석(CNN/MP): python MediaPipe/code/posture_test.py 실행

⚙️ 분석 파이프라인 (Pipeline)
CNN (MobileNetV2): 입력 이미지를 기반으로 자세의 Good/Bad를 1차 판정합니다.

MediaPipe: 신체 관절 좌표(Landmarks)를 추출하여 세부 각도를 분석합니다.

YOLOv8: 환경 객체(의자, 책상, 모니터)를 탐지하여 위치 정보를 확보합니다.

통합 분석: 추출된 관절 좌표와 객체 위치를 결합하여 RULA/VDT 지표 점수를 산출합니다.

시각화: 분석 결과 및 개선 피드백을 화면에 오버레이하여 출력합니다.

💻 개발 환경 (Environment)
Language: Python 3.11

Framework: PyTorch 2.5.1 (CUDA 12.1), Ultralytics YOLOv8, TensorFlow, MediaPipe

Hardware: NVIDIA GeForce RTX 4060

OS: Windows 10/11