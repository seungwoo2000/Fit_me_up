import os
import csv
import time
import subprocess
import sys
import torch

# ── 디바이스 태그 ──────────────────────────────────────────
device_tag = 'gpu' if torch.cuda.is_available() else 'cpu'

scripts = [
    ('train_full_body.py',      f'full_body_only_{device_tag}'),
    ('train_ankle_visible.py',  f'ankle_visible_only_{device_tag}'),
    ('train_combined.py',       f'combined_{device_tag}'),
]

# ── 학습 실행 ──────────────────────────────────────────────
elapsed_times = []
total_start = time.time()

for script, run_name in scripts:
    print(f"\n{'='*55}")
    print(f">>> 실행 중: {script}")
    print(f"{'='*55}")
    start = time.time()
    subprocess.run([sys.executable, script])
    elapsed_times.append(time.time() - start)

total_elapsed = time.time() - total_start

# ── 결과 읽기 ──────────────────────────────────────────────
def read_best_metrics(run_name, project=r'E:\python\FIT_ME_UP\YOLO\fit_me_up'):
    csv_path = os.path.join(project, run_name, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"[WARN] 파일 없음: {csv_path}")
        return None

    with open(csv_path, newline='') as f:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]

    if not rows:
        return None

    best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0)))
    return {
        'total_epochs': len(rows),
        'best_epoch':   int(float(best.get('epoch', 0))) + 1,
        'mAP50':        float(best.get('metrics/mAP50(B)', 0)),
        'mAP50_95':     float(best.get('metrics/mAP50-95(B)', 0)),
        'precision':    float(best.get('metrics/precision(B)', 0)),
        'recall':       float(best.get('metrics/recall(B)', 0)),
        'box_loss':     float(best.get('val/box_loss', 0)),
        'cls_loss':     float(best.get('val/cls_loss', 0)),
    }

metrics_list = [read_best_metrics(run_name) for _, run_name in scripts]

# ── 결과표 출력 ────────────────────────────────────────────
def fmt(val, dec):
    return f"{val:.{dec}f}" if val is not None else "N/A"

def fmt_time(sec):
    return f"{int(sec//3600)}시{int((sec%3600)//60)}분{int(sec%60)}초"

print(f"\n{'='*62}")
print(f"{'📊 학습 결과 비교':^62}")
print(f"{'='*62}")
print(f"{'항목':<22} {'full_body':>12} {'ankle_vis':>12} {'combined':>12}")
print(f"{'-'*62}")

# 소요시간
print(f"{'소요시간':<22}", end="")
for e in elapsed_times:
    print(f" {fmt_time(e):>12}", end="")
print()

# 성능 지표
metric_labels = [
    ('총 epoch 수',  'total_epochs', 0),
    ('Best epoch',   'best_epoch',   0),
    ('mAP50',        'mAP50',        4),
    ('mAP50-95',     'mAP50_95',     4),
    ('Precision',    'precision',    4),
    ('Recall',       'recall',       4),
    ('Box Loss',     'box_loss',     4),
    ('Cls Loss',     'cls_loss',     4),
]

for label, key, dec in metric_labels:
    print(f"{label:<22}", end="")
    for m in metrics_list:
        val = m[key] if m else None
        print(f" {fmt(val, dec):>12}", end="")
    print()

print(f"{'-'*62}")
print(f"{'총 합계 소요시간':<22} {fmt_time(total_elapsed)}")
print(f"{'='*62}")

# ── Best 모델 추천 ─────────────────────────────────────────
valid = [(i, m) for i, m in enumerate(metrics_list) if m]
if valid:
    best_idx, best_m = max(valid, key=lambda x: x[1]['mAP50'])
    best_name = scripts[best_idx][1]
    print(f"\n🏆 Best 모델: {best_name}  (mAP50: {best_m['mAP50']:.4f})")
print(f"{'='*62}\n")