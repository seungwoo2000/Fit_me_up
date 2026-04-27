# =====================================================================
# train_ankle_visible.py | ankle_visible 데이터셋 단독 학습
# 실행: python train_ankle_visible.py
# =====================================================================
from ultralytics import YOLO
import torch, os, time

BASE = os.path.dirname(os.path.abspath(__file__))  # 경로 자동 감지


def train(device):
    start      = time.time()
    device_tag = 'gpu' if device == '0' else 'cpu'
    print(f"\n>>> 디바이스: {device_tag.upper()}")

    model = YOLO(os.path.join(BASE, 'yolov8n.pt'))

    # ── 하이퍼파라미터 (YOLOv8 공식 권장값) ──────────────────
    params = {
        'data':         os.path.join(BASE, 'data_ankle_visible.yaml'),
        'epochs':       100,
        'patience':     20,
        'batch':        16,
        'imgsz':        640,
        'lr0':          0.01,
        'lrf':          0.001,
        'momentum':     0.937,
        'weight_decay': 0.0005,
        'optimizer':    'SGD',
        'device':       device,
        'project':      os.path.join(BASE, 'fit_me_up'),
        'name':         f'ankle_visible_only_{device_tag}',
        'exist_ok':     True,
    }

    # Step 1: 학습
    print(">>> [Step 1] 학습...")
    t0 = time.time()
    model.train(**params)
    print(f"⏱ {int((time.time()-t0)//60)}분 {int((time.time()-t0)%60)}초")

    # Step 2: 검증
    print(">>> [Step 2] 검증...")
    t1 = time.time()
    m = model.val()
    print(f"⏱ {int((time.time()-t1)//60)}분 {int((time.time()-t1)%60)}초")
    print(f"  mAP50={m.box.map50:.4f} | Precision={m.box.mp:.4f} | Recall={m.box.mr:.4f}")

    # Step 3: 추론
    print(">>> [Step 3] 추론...")
    src = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'split_data', 'test', 'images')
    t2  = time.time()
    if os.path.exists(src):
        model.predict(source=src, save=True, conf=0.25)
        print(f"⏱ {int((time.time()-t2)//60)}분 {int((time.time()-t2)%60)}초")
    else:
        print(f"[WARN] 경로 없음: {src}")

    elapsed = time.time() - start
    print(f"\n⏱ ankle_visible_only_{device_tag} 총 소요: "
          f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m {int(elapsed%60)}s")
    return elapsed


if __name__ == '__main__':
    train('0' if torch.cuda.is_available() else 'cpu')
