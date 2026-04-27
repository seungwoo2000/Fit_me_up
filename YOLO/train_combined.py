from ultralytics import YOLO
import torch
import os
import time

# ── 현재 파일 기준 경로 자동 감지 ─────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

def train(device):
    start      = time.time()
    device_tag = 'gpu' if device == '0' else 'cpu'
    print(f">>> 사용 디바이스: {device_tag.upper()}")

    yaml_path    = os.path.join(BASE, 'data_combined.yaml')
    test_sources = [
        os.path.join(BASE, 'YOLO_full_body_Labeling',    'split_data', 'test', 'images'),
        os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'split_data', 'test', 'images'),
    ]
    model = YOLO(os.path.join(BASE, 'yolov8n.pt'))

    params = {
        'data':         yaml_path,
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
        'name':         f'combined_{device_tag}',
        'exist_ok':     True,
    }

    print(">>> [Step 1] 학습 시작...")
    t0 = time.time()
    model.train(**params)
    print(f"⏱ 학습: {int((time.time()-t0)//60)}분 {int((time.time()-t0)%60)}초")

    print(">>> [Step 2] 검증 시작...")
    t1 = time.time()
    metrics = model.val()
    print(f"⏱ 검증: {int((time.time()-t1)//60)}분 {int((time.time()-t1)%60)}초")
    print(f"mAP50: {metrics.box.map50:.4f} | Precision: {metrics.box.mp:.4f} | Recall: {metrics.box.mr:.4f}")

    print(">>> [Step 3] 추론 시작...")
    t2 = time.time()
    for source in test_sources:
        if os.path.exists(source):
            model.predict(source=source, save=True, conf=0.25)
            print(f"완료: {source}")
        else:
            print(f"경로 없음: {source}")
    print(f"⏱ 추론: {int((time.time()-t2)//60)}분 {int((time.time()-t2)%60)}초")

    elapsed = time.time() - start
    print(f"\n⏱ combined_{device_tag} 총 소요시간: {int(elapsed//3600)}시간 {int((elapsed%3600)//60)}분 {int(elapsed%60)}초")
    return elapsed

if __name__ == '__main__':
    device = '0' if torch.cuda.is_available() else 'cpu'
    train(device)
