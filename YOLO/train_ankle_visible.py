# =====================================================================
# train_ankle_visible.py | ankle_visible 데이터셋 단독 학습
# 실행: python train_ankle_visible.py
# =====================================================================
from ultralytics import YOLO
import torch, os, time

BASE = os.path.dirname(os.path.abspath(__file__))


def fmt_time(sec):
    """초를 h m s 형식으로 변환"""
    return f"{int(sec//3600)}h {int((sec%3600)//60)}m {int(sec%60)}s"


def train(device):
    start      = time.time()
    device_tag = 'gpu' if device == '0' else 'cpu'
    print(f"\n>>> 디바이스: {device_tag.upper()}  ({torch.cuda.get_device_name(0) if device == '0' else 'CPU'})")

    model = YOLO(os.path.join(BASE, 'yolov8n.pt'))

    # ── 하이퍼파라미터 ────────────────────────────────────────────────
    params = {
        'data':         os.path.join(BASE, 'data_ankle_visible.yaml'),
        'epochs':       100,
        'patience':     20,        # 20 epoch 연속 개선 없으면 조기 종료
        'batch':        16,
        'imgsz':        640,
        'lr0':          0.01,      # 초기 학습률
        'lrf':          0.001,     # 최종 학습률
        'momentum':     0.937,
        'weight_decay': 0.0005,
        'optimizer':    'SGD',
        'augment':      True,      # 데이터 증강
        'cache':        True,      # 이미지 캐싱 (학습 속도 향상)
        'device':       device,
        'project':      os.path.join(BASE, 'fit_me_up'),
        'name':         f'ankle_visible_only_{device_tag}',
        'exist_ok':     True,
    }

    # ── Step 1: 학습 ──────────────────────────────────────────────────
    print(">>> [Step 1] 학습 시작...")
    t0 = time.time()
    model.train(**params)
    t0_elapsed = time.time() - t0
    print(f"  학습 완료: {fmt_time(t0_elapsed)}")

    # ── Step 2: 검증 ──────────────────────────────────────────────────
    print(">>> [Step 2] 검증 중...")
    t1 = time.time()
    m = model.val()
    t1_elapsed = time.time() - t1
    print(f"  검증 완료: {fmt_time(t1_elapsed)}")
    print(f"  mAP50={m.box.map50:.4f} | mAP50-95={m.box.map:.4f} | Precision={m.box.mp:.4f} | Recall={m.box.mr:.4f}")

    # ── Step 3: 추론 ──────────────────────────────────────────────────
    print(">>> [Step 3] 테스트 추론 중...")
    src = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'split_data', 'test', 'images')
    t2  = time.time()
    if os.path.exists(src):
        model.predict(source=src, save=True, conf=0.25)
        t2_elapsed = time.time() - t2
        print(f"  추론 완료: {fmt_time(t2_elapsed)}")
    else:
        print(f"  [WARN] 테스트 경로 없음: {src}")

    elapsed = time.time() - start
    print(f"\n  ankle_visible_only_{device_tag} 총 소요시간: {fmt_time(elapsed)}")
    return elapsed


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Windows 멀티프로세싱 필수

    # ── GPU 학습 (기본) ───────────────────────────────────────────────
    device = '0' if torch.cuda.is_available() else 'cpu'
    train(device)

    # ── CPU 학습 (필요시 주석 해제) ───────────────────────────────────
    # train('cpu')
