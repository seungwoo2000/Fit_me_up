# =====================================================================
# run_all.py | CPU + GPU 학습 자동 실행 & 결과 비교표 출력
# 실행: python run_all.py
# =====================================================================
import os, csv, time, sys, torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from train_full_body     import train as train_full
from train_ankle_visible import train as train_ankle
from train_combined      import train as train_combined

# GPU 감지
HAS_GPU  = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else 'N/A'

SCRIPTS = [
    ('full_body_only',     train_full),
    ('ankle_visible_only', train_ankle),
    ('combined',           train_combined),
]


# ── 학습 실행 ──────────────────────────────────────────────
def run_all(device):
    device_tag    = 'gpu' if device == '0' else 'cpu'
    elapsed_times = []
    total_start   = time.time()

    for name, fn in SCRIPTS:
        print(f"\n{'='*58}")
        print(f"  [{device_tag.upper()}] {name} 학습 시작")
        print(f"{'='*58}")
        elapsed_times.append(fn(device))

    return elapsed_times, time.time() - total_start


# ── results.csv 읽기 ───────────────────────────────────────
def read_metrics(run_name):
    path = os.path.join(BASE, 'fit_me_up', run_name, 'results.csv')
    if not os.path.exists(path):
        print(f"[WARN] 파일 없음: {path}")
        return None
    with open(path, newline='') as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0)))
    return {
        'total_epochs': len(rows),
        'best_epoch':   int(float(best.get('epoch', 0))) + 1,
        'mAP50':        float(best.get('metrics/mAP50(B)',    0)),
        'mAP50_95':     float(best.get('metrics/mAP50-95(B)', 0)),
        'precision':    float(best.get('metrics/precision(B)', 0)),
        'recall':       float(best.get('metrics/recall(B)',    0)),
        'box_loss':     float(best.get('val/box_loss', 0)),
        'cls_loss':     float(best.get('val/cls_loss', 0)),
    }


# ── 결과표 출력 ────────────────────────────────────────────
def fmt(v, d):     return f"{v:.{d}f}" if v is not None else "N/A"
def fmt_t(sec):    return f"{int(sec//3600)}h {int((sec%3600)//60)}m {int(sec%60)}s"

def print_table(title, elapsed, total, device_tag):
    names   = [f'{n}_{device_tag}' for n, _ in SCRIPTS]
    metrics = [read_metrics(n) for n in names]

    W = 68
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"{'항목':<22} {'full_body':>14} {'ankle_vis':>14} {'combined':>14}")
    print(f"{'-'*W}")

    # 소요시간
    print(f"{'소요시간':<22}", end="")
    for e in elapsed:
        print(f" {fmt_t(e):>14}", end="")
    print()

    # 성능 지표
    for label, key, dec in [
        ('총 epoch 수',  'total_epochs', 0),
        ('Best epoch',   'best_epoch',   0),
        ('mAP50',        'mAP50',        4),
        ('mAP50-95',     'mAP50_95',     4),
        ('Precision',    'precision',    4),
        ('Recall',       'recall',       4),
        ('Box Loss',     'box_loss',     4),
        ('Cls Loss',     'cls_loss',     4),
    ]:
        print(f"{label:<22}", end="")
        for m in metrics:
            print(f" {fmt(m[key] if m else None, dec):>14}", end="")
        print()

    print(f"{'-'*W}")
    print(f"{'총 합계 소요시간':<22} {fmt_t(total)}")
    print(f"{'='*W}")

    valid = [(i, m) for i, m in enumerate(metrics) if m]
    if valid:
        bi, bm = max(valid, key=lambda x: x[1]['mAP50'])
        print(f"\n  Best 모델: {names[bi]}  (mAP50: {bm['mAP50']:.4f})")
    print(f"{'='*W}\n")


# ── 메인 실행 ──────────────────────────────────────────────
print(f"\nGPU: {'있음 (' + GPU_NAME + ')' if HAS_GPU else '없음'}")
print(f"{'='*68}")

# CPU 학습
print("  [1/2] CPU 학습")
cpu_times, cpu_total = run_all('cpu')
print_table("CPU 학습 결과", cpu_times, cpu_total, 'cpu')

# GPU 학습 (GPU 있을 때만)
if HAS_GPU:
    print(f"  [2/2] GPU 학습 ({GPU_NAME})")
    gpu_times, gpu_total = run_all('0')
    print_table(f"GPU 학습 결과 ({GPU_NAME})", gpu_times, gpu_total, 'gpu')

    # CPU vs GPU 비교표
    W = 68
    print(f"\n{'='*W}")
    print(f"  CPU vs GPU 학습 시간 비교")
    print(f"{'='*W}")
    print(f"  {'모델':<24} {'CPU':>12} {'GPU':>12} {'단축률':>10}")
    print(f"  {'-'*60}")
    for i, (name, _) in enumerate(SCRIPTS):
        c, g = cpu_times[i], gpu_times[i]
        r    = (c - g) / c * 100 if c > 0 else 0
        print(f"  {name:<24} {fmt_t(c):>12} {fmt_t(g):>12} {r:>9.1f}%")
    print(f"  {'-'*60}")
    cr = (cpu_total - gpu_total) / cpu_total * 100 if cpu_total > 0 else 0
    print(f"  {'총 합계':<24} {fmt_t(cpu_total):>12} {fmt_t(gpu_total):>12} {cr:>9.1f}%")
    print(f"{'='*W}\n")

else:
    print("\nGPU 없음 → CPU 학습만 수행했습니다.")
