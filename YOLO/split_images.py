# =====================================================================
# split_images.py | 이미지 데이터셋 분할 (train 80 / val 10 / test 10)
# 실행: python split_images.py
# =====================================================================
import os, shutil, random

BASE = os.path.dirname(os.path.abspath(__file__))  # 경로 자동 감지


def split_dataset(images_dir, labels_dir, output_dir):
    """이미지와 라벨을 train/val/test 폴더로 분할"""
    random.seed(1337)  # 재현 가능한 셔플

    image_files = [f for f in os.listdir(images_dir)
                   if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    total  = len(image_files)
    splits = {
        'train': image_files[:int(total * 0.8)],
        'val':   image_files[int(total * 0.8):int(total * 0.9)],
        'test':  image_files[int(total * 0.9):]
    }

    for split, files in splits.items():
        # 폴더 생성
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)

        # 파일 복사
        for fname in files:
            shutil.copy(
                os.path.join(images_dir, fname),
                os.path.join(output_dir, split, 'images', fname)
            )
            label = os.path.splitext(fname)[0] + '.txt'
            label_src = os.path.join(labels_dir, label)
            if os.path.exists(label_src):
                shutil.copy(label_src,
                            os.path.join(output_dir, split, 'labels', label))
            else:
                print(f"[WARN] 라벨 없음: {label}")

    print(f"완료! train:{len(splits['train'])} / "
          f"val:{len(splits['val'])} / test:{len(splits['test'])}")


# ── full_body ──────────────────────────────────────────────
print(">>> full_body 분할 중...")
split_dataset(
    images_dir = os.path.join(BASE, 'YOLO_full_body_Labeling',    'images'),
    labels_dir = os.path.join(BASE, 'YOLO_full_body_Labeling',    'labels'),
    output_dir = os.path.join(BASE, 'YOLO_full_body_Labeling',    'split_data'),
)

# ── ankle_visible ──────────────────────────────────────────
print(">>> ankle_visible 분할 중...")
split_dataset(
    images_dir = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'images'),
    labels_dir = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'labels'),
    output_dir = os.path.join(BASE, 'YOLO_ankle_visible_Labeling', 'split_data'),
)
