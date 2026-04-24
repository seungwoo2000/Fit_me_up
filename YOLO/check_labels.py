import cv2
import os

# ── 확인할 폴더 경로 ───────────────────────────────────────
images_dir = r"E:\python\FIT_ME_UP\YOLO\YOLO_ankle_visible_Labeling\images"
labels_dir = r"E:\python\FIT_ME_UP\YOLO\YOLO_ankle_visible_Labeling\labels"

# ── 클래스 이름 ────────────────────────────────────────────
class_names = ['chair', 'desk', 'monitor']
colors = [(0,255,0), (0,0,255), (255,0,0)]

# ── 이미지 3장만 확인 ──────────────────────────────────────
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:3]

for fname in image_files:
    img = cv2.imread(os.path.join(images_dir, fname))
    h, w = img.shape[:2]

    label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + '.txt')
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f.readlines():
                cls, cx, cy, bw, bh = map(float, line.strip().split())
                cls = int(cls)
                x1 = int((cx - bw/2) * w)
                y1 = int((cy - bh/2) * h)
                x2 = int((cx + bw/2) * w)
                y2 = int((cy + bh/2) * h)
                cv2.rectangle(img, (x1,y1), (x2,y2), colors[cls], 2)
                cv2.putText(img, class_names[cls], (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colors[cls], 2)

    cv2.imshow(fname, img)
    cv2.waitKey(0)

cv2.destroyAllWindows()