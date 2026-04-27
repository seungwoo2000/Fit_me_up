import os
import shutil
import random

def split_dataset(images_dir, labels_dir, output_dir):
    # 이미지 목록 & 셔플
    random.seed(1337)
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)

    # 비율 분할 (train 80 / val 10 / test 10)
    total = len(image_files)
    splits = {
        'train': image_files[:int(total * 0.8)],
        'val':   image_files[int(total * 0.8):int(total * 0.9)],
        'test':  image_files[int(total * 0.9):]
    }

    # 폴더 생성 & 복사
    for split, files in splits.items():
        for sub in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, sub), exist_ok=True)
        for fname in files:
            shutil.copy(os.path.join(images_dir, fname),
                        os.path.join(output_dir, split, 'images', fname))
            label = os.path.splitext(fname)[0] + '.txt'
            if os.path.exists(os.path.join(labels_dir, label)):
                shutil.copy(os.path.join(labels_dir, label),
                            os.path.join(output_dir, split, 'labels', label))

    print(f"완료! train:{len(splits['train'])} / val:{len(splits['val'])} / test:{len(splits['test'])}")

# ── full_body ──────────────────────────────────────────────
split_dataset(
    images_dir = r"C:\python\FIT_ME_UP\YOLO\YOLO_full_body_Labeling\images",
    labels_dir = r"C:\python\FIT_ME_UP\YOLO\YOLO_full_body_Labeling\labels",
    output_dir = r"C:\python\FIT_ME_UP\YOLO\YOLO_full_body_Labeling\split_data"
)

# ── ankle_visible ──────────────────────────────────────────
split_dataset(
    images_dir = r"C:\python\FIT_ME_UP\YOLO\YOLO_ankle_visible_Labeling\images",
    labels_dir = r"C:\python\FIT_ME_UP\YOLO\YOLO_ankle_visible_Labeling\labels",
    output_dir = r"C:\python\FIT_ME_UP\YOLO\YOLO_ankle_visible_Labeling\split_data"
)