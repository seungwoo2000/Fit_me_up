# =====================================================================
# run_all.py | 3가지 학습 전략 자동 실행 & 평가지표 비교표 출력
# 학습 전략: full_body_only / ankle_visible_only / combined
# 실행: python run_all.py
# =====================================================================
import os, csv, time, sys, torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from train_full_body     import train as train_full
from train_ankle_visible import train as train_ankle
from train_combined      import train as train_combined

# ── 디바이스 감지 ─────────────────────────────────────────────────────
HAS_GPU  = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else 'N/A'

STRATEGIES = [
    ('full_body_only',     train_full),
    ('ankle_visible_only', train_ankle),
    ('combined',           train_combined),
]


# ── 유틸 ──────────────────────────────────────────────────────────────
def fmt_time(sec):
    """초를 h m s 형식으로 변환"""
    return f"{int(sec//3600)}h {int((sec%3600)//60)}m {int(sec%60)}s"

def fmt(v, d):
    """숫자 포맷 (None이면 N/A)"""
    return f"{v:.{d}f}" if v is not None else "N/A"


# ── 학습 실행 ─────────────────────────────────────────────────────────
def run_all(device):
    device_tag    = 'gpu' if device == '0' else 'cpu'
    elapsed_times = []
    total_start   = time.time()

    for idx, (name, fn) in enumerate(STRATEGIES, 1):
        print(f"\n{'='*62}")
        print(f"  [{idx}/3] [{device_tag.upper()}] {name} 학습 시작")
        print(f"{'='*62}")
        elapsed_times.append(fn(device))

    return elapsed_times, time.time() - total_start


# ── results.csv 읽기 ──────────────────────────────────────────────────
def read_metrics(run_name):
    """학습 결과 CSV에서 best epoch 기준 평가지표 추출"""
    path = os.path.join(BASE, 'fit_me_up', run_name, 'results.csv')
    if not os.path.exists(path):
        print(f"  [WARN] 결과 파일 없음: {path}")
        return None
    with open(path, newline='') as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0)))
    return {
        'total_epochs': len(rows),
        'best_epoch':   int(float(best.get('epoch', 0))) + 1,
        'mAP50':        float(best.get('metrics/mAP50(B)',     0)),
        'mAP50_95':     float(best.get('metrics/mAP50-95(B)', 0)),
        'precision':    float(best.get('metrics/precision(B)', 0)),
        'recall':       float(best.get('metrics/recall(B)',    0)),
        'box_loss':     float(best.get('val/box_loss',         0)),
        'cls_loss':     float(best.get('val/cls_loss',         0)),
    }


# ── 평가지표 비교표 출력 ──────────────────────────────────────────────
def print_table(title, elapsed, total, device_tag):
    names   = [f'{n}_{device_tag}' for n, _ in STRATEGIES]
    metrics = [read_metrics(n) for n in names]

    W = 72
    COL = 15

    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")
    print(f"  {'항목':<20} {'full_body':>{COL}} {'ankle_vis':>{COL}} {'combined':>{COL}}")
    print(f"  {'-'*(W-2)}")

    # 소요시간
    print(f"  {'소요시간':<20}", end="")
    for e in elapsed:
        print(f" {fmt_time(e):>{COL}}", end="")
    print()

    print(f"  {'-'*(W-2)}")

    # 평가지표
    indicators = [
        ('총 epoch 수',  'total_epochs', 0),
        ('Best epoch',   'best_epoch',   0),
        ('mAP50  ↑',     'mAP50',        4),
        ('mAP50-95  ↑',  'mAP50_95',     4),
        ('Precision  ↑', 'precision',    4),
        ('Recall  ↑',    'recall',       4),
        ('Box Loss  ↓',  'box_loss',     4),
        ('Cls Loss  ↓',  'cls_loss',     4),
    ]

    for label, key, dec in indicators:
        print(f"  {label:<20}", end="")
        values = [m[key] if m else None for m in metrics]

        # mAP50 기준 Best 강조
        best_val = None
        if key in ('mAP50', 'mAP50_95', 'precision', 'recall'):
            valid = [v for v in values if v is not None]
            best_val = max(valid) if valid else None
        elif key in ('box_loss', 'cls_loss'):
            valid = [v for v in values if v is not None]
            best_val = min(valid) if valid else None

        for v in values:
            cell = fmt(v, dec)
            marker = " ★" if (v is not None and v == best_val) else "  "
            print(f" {(cell + marker):>{COL}}", end="")
        print()

    print(f"  {'-'*(W-2)}")
    print(f"  {'총 소요시간':<20} {fmt_time(total)}")
    print(f"{'='*W}")

    # Best 모델 요약
    valid = [(i, m) for i, m in enumerate(metrics) if m]
    if valid:
        bi, bm = max(valid, key=lambda x: x[1]['mAP50'])
        print(f"\n  ★ Best 모델: {names[bi]}")
        print(f"    mAP50={bm['mAP50']:.4f} | mAP50-95={bm['mAP50_95']:.4f} | "
              f"Precision={bm['precision']:.4f} | Recall={bm['recall']:.4f}")
    print(f"{'='*W}\n")


# ── 메인 실행 ─────────────────────────────────────────────────────────
# Windows 멀티프로세싱 필수 가드 (없으면 RuntimeError 발생)
if __name__ == '__main__':
    print(f"\n{'='*62}")
    print(f"  YOLO 학습 자동 실행 (3가지 전략 비교)")
    print(f"  GPU: {'있음 → ' + GPU_NAME if HAS_GPU else '없음 → CPU로 실행'}")
    print(f"{'='*62}")

    if not HAS_GPU:
        print("\n  [경고] GPU가 감지되지 않아 CPU로 실행합니다.")
        print("  학습 시간이 매우 길어질 수 있습니다.\n")

    # ── GPU 학습 (기본) ───────────────────────────────────────────────
    device     = '0' if HAS_GPU else 'cpu'
    device_tag = 'gpu' if HAS_GPU else 'cpu'

    print(f"\n  [{device_tag.upper()}] 3가지 전략 학습 시작")
    times, total = run_all(device)
    print_table(
        f"학습 결과 비교표 [{device_tag.upper()} / {GPU_NAME if HAS_GPU else 'CPU'}]",
        times, total, device_tag
    )

    # ── CPU 학습 (필요시 주석 해제) ───────────────────────────────────
    # print("\n  [CPU] 3가지 전략 학습 시작")
    # cpu_times, cpu_total = run_all('cpu')
    # print_table("학습 결과 비교표 [CPU]", cpu_times, cpu_total, 'cpu')
