# =====================================================================
# split_images.py | 이미지 데이터셋 분할 (train 80 / val 10 / test 10)
# 실행: python split_images.py
# =====================================================================
import os, shutil, random

BASE = os.path.dirname(os.path.abspath(__file__))

random.seed(1337)  # 재현 가능한 셔플 (함수 밖에서 한 번만 설정)


def split_dataset(images_dir, labels_dir, output_dir):
    """이미지와 라벨을 train / val / test 폴더로 분할"""

    if not os.path.exists(images_dir):
        print(f"[ERROR] 이미지 폴더 없음: {images_dir}")
        return

    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    if not image_files:
        print(f"[WARN] 이미지 없음: {images_dir}")
        return

    random.shuffle(image_files)

    total  = len(image_files)
    splits = {
        'train': image_files[:int(total * 0.8)],
        'val':   image_files[int(total * 0.8):int(total * 0.9)],
        'test':  image_files[int(total * 0.9):]
    }

    missing_labels = 0

    for split, files in splits.items():
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

        for fname in files:
            # 이미지 복사
            shutil.copy(
                os.path.join(images_dir, fname),
                os.path.join(output_dir, split, 'images', fname)
            )
            # 라벨 복사
            label     = os.path.splitext(fname)[0] + '.txt'
            label_src = os.path.join(labels_dir, label)
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(output_dir, split, 'labels', label))
            else:
                missing_labels += 1
                print(f"  [WARN] 라벨 없음: {label}")

    print(f"  완료 → train:{len(splits['train'])} / val:{len(splits['val'])} / test:{len(splits['test'])}"
          f"  (라벨 누락: {missing_labels}개)")


# ── full_body ──────────────────────────────────────────────────────────
print("\n>>> [1/2] full_body 분할 중...")
split_dataset(
    images_dir = os.path.join(BASE, 'YOLO_full_body_Labeling',     'images'),
    labels_dir = os.path.join(BASE, 'YOLO_full_body_Labeling',     'labels'),
    output_dir = os.path.join(BASE, 'YOLO_full_body_Labeling',     'split_data'),
)

# ── ankle_visible ──────────────────────────────────────────────────────
print("\n>>> [2/2] ankle_visible 분할 중...")
split_dataset(
    images_dir = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'images'),
    labels_dir = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'labels'),
    output_dir = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'split_data'),
)

print("\n>>> 데이터 분할 완료!\n")