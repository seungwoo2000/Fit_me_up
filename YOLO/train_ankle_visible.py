from ultralytics import YOLO
import os
import time

def main():
    # ── 시간 측정 시작 ─────────────────────────────────────────
    start = time.time()

    # ── 경로 ──────────────────────────────────────────────────
    yaml_path   = r"E:\python\FIT_ME_UP\YOLO\data_ankle_visible.yaml"
    test_source = r"E:\python\FIT_ME_UP\YOLO\YOLO_ankle_visible_Labeling\split_data\test\images"

    # ── 모델 로드 ──────────────────────────────────────────────
    model = YOLO(r'E:\python\FIT_ME_UP\YOLO\yolov8n.pt')

    # ── 하이퍼파라미터 ─────────────────────────────────────────
    params = {
        'data':         yaml_path,
        'epochs':       100,         # 반복 학습 횟수
        'patience':     20,         # 조기 종료
        'batch':        16,        # 한 번에 처리할 이미지 수
        'imgsz':        640,       # 입력 이미지 크기
        'lr0':          0.01,      # 초기 학습 속도
        'lrf':          0.01,      # 최종 학습 속도
        'momentum':     0.937,     # 학습 방향 유지 관성
        'weight_decay': 0.0005,    # 과적합 방지
        'optimizer':    'SGD',     # 최적화 방식
        'device':       'cpu',     # GPU: '0' / CPU: 'cpu'
        'project':      r'E:\python\FIT_ME_UP\YOLO\fit_me_up',
        'name':         'ankle_visible_only',
        'exist_ok':     True
    }

    # ── 학습 ───────────────────────────────────────────────────
    print(">>> [Step 1] 학습 시작...")
    model.train(**params)

    # ── 검증 & 성능 지표 ───────────────────────────────────────
    print(">>> [Step 2] 검증 시작...")
    metrics = model.val()
    print(f"\n📊 [ankle_visible_only] 성능 지표")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    # ── 추론 ───────────────────────────────────────────────────
    print(">>> [Step 3] 추론 시작...")
    if os.path.exists(test_source):
        model.predict(source=test_source, save=True, conf=0.45)
        print("완료! YOLO/fit_me_up/ankle_visible_only 폴더 확인하세요.")
    else:
        print(f"경로 없음: {test_source}")

    # ── 시간 측정 종료 ─────────────────────────────────────────
    end = time.time()
    elapsed = end - start
    print(f"\n⏱ ankle_visible_only 총 소요시간: {int(elapsed//3600)}시간 {int((elapsed%3600)//60)}분 {int(elapsed%60)}초")

if __name__ == '__main__':
    main()