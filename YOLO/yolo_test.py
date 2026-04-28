# =====================================================================
# yolo_test.py | YOLO 추론 테스트 UI
# 실행: python yolo_test.py
# =====================================================================
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, 'fit_me_up', 'combined_gpu', 'weights', 'best.pt')
CONF       = 0.25
IOU        = 0.30   # 겹침 허용 범위 (낮을수록 중복 탐지 제거 적극적)
CLASSES    = ['chair', 'desk', 'monitor']
COLORS     = {
    'chair':   (52,  152, 219),   # 파랑
    'desk':    (46,  204, 113),   # 초록
    'monitor': (231, 76,  60),    # 빨강
}


class YoloTestApp:
    def __init__(self, root):
        self.root       = root
        self.root.title("YOLO 환경 탐지 테스트")
        self.root.geometry("950x680")
        self.root.configure(bg="#f0f0f0")

        self.model      = None
        self.image_path = None
        self.tk_img     = None

        self.setup_ui()
        self.load_model()

    # ── 모델 로드 ─────────────────────────────────────────────────────
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"모델 파일 없음:\n{MODEL_PATH}\n\nrun_all.py를 먼저 실행하세요.")
            return
        try:
            self.model = YOLO(MODEL_PATH)
            self.lbl_status.config(text="모델 로드 완료", fg="#27ae60")
        except Exception as e:
            messagebox.showerror("Error", f"모델 로드 실패: {e}")

    # ── UI 구성 ───────────────────────────────────────────────────────
    def setup_ui(self):
        # 헤더
        tk.Label(
            self.root, text="📦 YOLO 환경 탐지 테스트",
            font=("Malgun Gothic", 20, "bold"),
            bg="#2c3e50", fg="white", pady=10
        ).pack(fill=tk.X)

        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # ── 왼쪽: 이미지 캔버스 ──────────────────────────────────────
        left_frame = tk.Frame(main_frame, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(left_frame, width=500, height=500, bg="white", highlightthickness=1)
        self.canvas.pack()

        btn_frame = tk.Frame(left_frame, bg="#f0f0f0")
        btn_frame.pack(pady=15)

        tk.Button(
            btn_frame, text="📁 이미지 선택",
            command=self.select_image,
            width=15, bg="#3498db", fg="white"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="🔍 탐지 시작",
            command=self.run_detection,
            width=15, bg="#2ecc71", fg="white"
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            btn_frame, text="🔄 초기화",
            command=self.reset,
            width=10, bg="#95a5a6", fg="white"
        ).pack(side=tk.LEFT, padx=5)

        # ── 오른쪽: 결과 패널 ─────────────────────────────────────────
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)

        # 모델 상태
        self.lbl_status = tk.Label(
            right_frame, text="모델 로딩 중...",
            font=("Malgun Gothic", 11), bg="#f0f0f0", fg="#e67e22"
        )
        self.lbl_status.pack(pady=5)

        # 탐지 결과 수
        self.res_panel = tk.Frame(right_frame, width=320, height=90, bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        self.res_panel.pack_propagate(False)
        self.res_panel.pack(pady=10)

        self.lbl_result = tk.Label(
            self.res_panel, text="READY",
            font=("Arial", 28, "bold"), bg="#ecf0f1", fg="#7f8c8d"
        )
        self.lbl_result.pack(expand=True)

        # 클래스별 결과
        tk.Label(right_frame, text="탐지 결과", font=("Malgun Gothic", 13, "bold"), bg="#f0f0f0").pack(pady=(15, 5))

        self.lbl_chair   = tk.Label(right_frame, text="🪑 chair   : -", font=("Malgun Gothic", 12), bg="#f0f0f0", anchor="w", width=28)
        self.lbl_desk    = tk.Label(right_frame, text="🖥  desk    : -", font=("Malgun Gothic", 12), bg="#f0f0f0", anchor="w", width=28)
        self.lbl_monitor = tk.Label(right_frame, text="🖥  monitor : -", font=("Malgun Gothic", 12), bg="#f0f0f0", anchor="w", width=28)

        for lbl in [self.lbl_chair, self.lbl_desk, self.lbl_monitor]:
            lbl.pack(pady=3)

        # 신뢰도 요약
        tk.Label(right_frame, text="최고 신뢰도", font=("Malgun Gothic", 13, "bold"), bg="#f0f0f0").pack(pady=(15, 5))
        self.lbl_conf = tk.Label(right_frame, text="-", font=("Malgun Gothic", 12), bg="#f0f0f0")
        self.lbl_conf.pack(pady=3)

        # 저장 경로
        self.lbl_save = tk.Label(
            right_frame, text="", font=("Malgun Gothic", 9),
            bg="#f0f0f0", fg="#7f8c8d", wraplength=300
        )
        self.lbl_save.pack(pady=10)

    # ── 이미지 선택 ───────────────────────────────────────────────────
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="테스트할 이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not file_path:
            return
        self.image_path = file_path
        self._show_image(file_path)
        self.reset_labels()

    # ── 탐지 실행 ─────────────────────────────────────────────────────
    def run_detection(self):
        if not self.image_path:
            messagebox.showwarning("경고", "이미지를 먼저 선택하세요.")
            return
        if not self.model:
            messagebox.showerror("Error", "모델이 로드되지 않았습니다.")
            return

        results = self.model.predict(source=self.image_path, conf=CONF, iou=IOU, save=True, verbose=False)
        result  = results[0]
        boxes   = result.boxes

        # 결과 이미지 오버레이
        img_cv  = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        counts  = {c: 0    for c in CLASSES}
        confs   = {c: []   for c in CLASSES}

        if boxes and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xyxy   = box.xyxy[0].tolist()
                name   = CLASSES[cls_id] if cls_id < len(CLASSES) else f"class{cls_id}"

                counts[name] += 1
                confs[name].append(conf)

                # 바운딩박스 그리기
                color = COLORS.get(name, (200, 200, 200))
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
                label = f"{name} {conf:.0%}"
                cv2.rectangle(img_rgb, (x1, y1 - 28), (x1 + len(label) * 13, y1), color, -1)
                cv2.putText(img_rgb, label, (x1 + 4, y1 - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            total = sum(counts.values())
            self.lbl_result.config(text=f"{total}개 탐지", bg="#2ecc71", fg="white")
            self.res_panel.config(bg="#2ecc71")
        else:
            self.lbl_result.config(text="미탐지", bg="#e74c3c", fg="white")
            self.res_panel.config(bg="#e74c3c")

        # 클래스별 라벨 업데이트
        def fmt(name, emoji):
            c = counts[name]
            avg_conf = f"{sum(confs[name])/len(confs[name]):.0%}" if confs[name] else "-"
            return f"{emoji} {name:<10}: {c}개  (avg {avg_conf})"

        self.lbl_chair.config(text=fmt('chair',   '🪑'))
        self.lbl_desk.config(text=fmt('desk',     '📋'))
        self.lbl_monitor.config(text=fmt('monitor', '🖥'))

        # 최고 신뢰도
        all_confs = [float(b.conf[0]) for b in boxes] if boxes and len(boxes) > 0 else []
        if all_confs:
            best_idx  = int(boxes.conf.argmax())
            best_name = CLASSES[int(boxes.cls[best_idx])] if int(boxes.cls[best_idx]) < len(CLASSES) else "?"
            self.lbl_conf.config(text=f"{best_name}  {max(all_confs):.1%}")

        # 결과 이미지 표시
        self._show_image_from_array(img_rgb)
        self.lbl_save.config(text=f"저장 위치: {result.save_dir}")

    # ── 초기화 ────────────────────────────────────────────────────────
    def reset(self):
        self.image_path = None
        self.canvas.delete("all")
        self.reset_labels()

    def reset_labels(self):
        self.lbl_result.config(text="READY", bg="#ecf0f1", fg="#7f8c8d")
        self.res_panel.config(bg="#ecf0f1")
        self.lbl_chair.config(text="🪑 chair   : -")
        self.lbl_desk.config(text="📋 desk    : -")
        self.lbl_monitor.config(text="🖥  monitor : -")
        self.lbl_conf.config(text="-")
        self.lbl_save.config(text="")

    # ── 이미지 표시 헬퍼 ─────────────────────────────────────────────
    def _show_image(self, path):
        img = Image.open(path)
        img.thumbnail((500, 500))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(250, 250, image=self.tk_img)

    def _show_image_from_array(self, img_rgb):
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((500, 500))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(250, 250, image=self.tk_img)


if __name__ == "__main__":
    root = tk.Tk()
    app  = YoloTestApp(root)
    root.mainloop()
