# python run_all.py
import os
import csv
import time
import sys
import torch

# ── 현재 파일 기준 경로 자동 감지 ─────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from train_full_body    import train as train_full
from train_ankle_visible import train as train_ankle
from train_combined     import train as train_combined

HAS_GPU   = torch.cuda.is_available()
GPU_NAME  = torch.cuda.get_device_name(0) if HAS_GPU else 'N/A'

scripts = [
    ('full_body_only',    train_full),
    ('ankle_visible_only', train_ankle),
    ('combined',           train_combined),
]

# =========================================================
# 학습 실행 함수
# =========================================================
def run_all(device):
    device_tag = 'gpu' if device == '0' else 'cpu'
    elapsed_times = []
    total_start   = time.time()

    for name, fn in scripts:
        print(f"\n{'='*55}")
        print(f">>> [{device_tag.upper()}] {name} 학습 시작")
        print(f"{'='*55}")
        elapsed_times.append(fn(device))

    total_elapsed = time.time() - total_start
    return elapsed_times, total_elapsed

# =========================================================
# 결과 CSV 읽기
# =========================================================
def read_best_metrics(run_name):
    project  = os.path.join(BASE, 'fit_me_up')
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

# =========================================================
# 결과표 출력 함수
# =========================================================
def fmt(val, dec):
    return f"{val:.{dec}f}" if val is not None else "N/A"

def fmt_time(sec):
    return f"{int(sec//3600)}시{int((sec%3600)//60)}분{int(sec%60)}초"

def print_table(title, elapsed_times, total_elapsed, device_tag):
    run_names = [f'{n}_{device_tag}' for n, _ in scripts]
    metrics   = [read_best_metrics(r) for r in run_names]

    print(f"\n{'='*66}")
    print(f"  {title}")
    print(f"{'='*66}")
    print(f"{'항목':<22} {'full_body':>13} {'ankle_vis':>13} {'combined':>13}")
    print(f"{'-'*66}")

    print(f"{'소요시간':<22}", end="")
    for e in elapsed_times:
        print(f" {fmt_time(e):>13}", end="")
    print()

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
        for m in metrics:
            val = m[key] if m else None
            print(f" {fmt(val, dec):>13}", end="")
        print()

    print(f"{'-'*66}")
    print(f"{'총 합계 소요시간':<22} {fmt_time(total_elapsed)}")
    print(f"{'='*66}")

    valid = [(i, m) for i, m in enumerate(metrics) if m]
    if valid:
        best_idx, best_m = max(valid, key=lambda x: x[1]['mAP50'])
        print(f"\n  Best 모델: {run_names[best_idx]}  (mAP50: {best_m['mAP50']:.4f})")
    print(f"{'='*66}\n")

# =========================================================
# 메인 실행
# =========================================================
print(f"\nGPU 감지: {'있음 (' + GPU_NAME + ')' if HAS_GPU else '없음 (CPU만 사용)'}")

# ── CPU 학습 ───────────────────────────────────────────────
print("\n" + "="*66)
print("  [1/2] CPU 학습 시작")
print("="*66)
cpu_times, cpu_total = run_all('cpu')
print_table("CPU 학습 결과", cpu_times, cpu_total, 'cpu')

# ── GPU 학습 (있을 때만) ───────────────────────────────────
if HAS_GPU:
    print("\n" + "="*66)
    print(f"  [2/2] GPU 학습 시작 ({GPU_NAME})")
    print("="*66)
    gpu_times, gpu_total = run_all('0')
    print_table(f"GPU 학습 결과 ({GPU_NAME})", gpu_times, gpu_total, 'gpu')

    # ── CPU vs GPU 비교표 ──────────────────────────────────
    print(f"\n{'='*66}")
    print(f"  CPU vs GPU 학습 시간 비교")
    print(f"{'='*66}")
    print(f"{'모델':<24} {'CPU':>13} {'GPU':>13} {'단축률':>10}")
    print(f"{'-'*66}")
    labels = ['full_body_only', 'ankle_visible_only', 'combined']
    for i, label in enumerate(labels):
        c = cpu_times[i]
        g = gpu_times[i]
        ratio = (c - g) / c * 100 if c > 0 else 0
        print(f"  {label:<22} {fmt_time(c):>13} {fmt_time(g):>13} {ratio:>9.1f}%")
    print(f"{'-'*66}")
    cr = (cpu_total - gpu_total) / cpu_total * 100 if cpu_total > 0 else 0
    print(f"  {'총 합계':<22} {fmt_time(cpu_total):>13} {fmt_time(gpu_total):>13} {cr:>9.1f}%")
    print(f"{'='*66}\n")
else:
    print("\nGPU가 없어 CPU 학습만 수행했습니다.")
