import os
import cv2
import numpy as np
import math
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import warnings
import requests # 자동 다운로드를 위해 추가됨

# 설정
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe 버전 대응 (Legacy vs Tasks API)
import mediapipe as mp
try:
    import mediapipe.solutions.pose as mp_pose
    mp_solutions = mp.solutions
    USE_LEGACY = True
except (AttributeError, ImportError, ModuleNotFoundError):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    USE_LEGACY = False

# 경로 및 설정 (상대 경로)
MODEL_PATH_TF = "../models/posture_mobilenetv2_finetuned.h5"
MODEL_PATH_MP = "../models/pose_landmarker.task"
THRESHOLD = 0.60

# [추가] 모델 파일 자동 다운로드 로직
def download_mediapipe_model(target_path):
    if not os.path.exists(target_path):
        print(f"MediaPipe 모델 파일이 없습니다. 다운로드를 시작합니다: {target_path}")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            response = requests.get(url, stream=True)
            with open(target_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("모델 다운로드 완료!")
        except Exception as e:
            print(f"다운로드 실패: {e}")

class PostureTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("자세 분석 테스트 앱 (v2.1)")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        
        self.model = None
        self.mp_detector = None
        self.image_path = None
        self.tk_img = None
        
        self.setup_ui()
        # [수정] 앱 시작 시 자동으로 모델 파일을 체크하고 다운로드함
        download_mediapipe_model(MODEL_PATH_MP)
        self.load_models()

    def load_models(self):
        try:
            if os.path.exists(MODEL_PATH_TF):
                self.model = tf.keras.models.load_model(MODEL_PATH_TF)
        except Exception as e:
            messagebox.showerror("Error", f"모델 로드 실패: {e}")

        try:
            if USE_LEGACY:
                self.mp_detector = mp_solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
            else:
                if os.path.exists(MODEL_PATH_MP):
                    base_options = python.BaseOptions(model_asset_path=MODEL_PATH_MP)
                    options = vision.PoseLandmarkerOptions(
                        base_options=base_options,
                        running_mode=vision.RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.5
                    )
                    self.mp_detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            messagebox.showerror("Error", f"MediaPipe 초기화 실패: {e}")

    def setup_ui(self):
        header = tk.Label(self.root, text="🧘 자세 분석 인퍼런스 테스트 (v2.1)", font=("Malgun Gothic", 20, "bold"), bg="#2c3e50", fg="white", pady=10)
        header.pack(fill=tk.X)
        
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        left_frame = tk.Frame(main_frame, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, padx=10)
        
        self.canvas = tk.Canvas(left_frame, width=450, height=450, bg="white", highlightthickness=1)
        self.canvas.pack()
        
        btn_frame = tk.Frame(left_frame, bg="#f0f0f0")
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="📁 이미지 선택", command=self.select_image, width=15, bg="#3498db", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🔍 분석 시작", command=self.run_analysis, width=15, bg="#2ecc71", fg="white").pack(side=tk.LEFT, padx=5)
        
        right_frame = tk.Frame(main_frame, bg="#f0f0f0")
        right_frame.pack(side=tk.LEFT, padx=30, fill=tk.Y)
        
        self.res_panel = tk.Frame(right_frame, width=300, height=100, bg="#ecf0f1", relief=tk.RIDGE, bd=2)
        self.res_panel.pack_propagate(False)
        self.res_panel.pack(pady=10)
        
        self.lbl_result = tk.Label(self.res_panel, text="READY", font=("Arial", 32, "bold"), bg="#ecf0f1", fg="#7f8c8d")
        self.lbl_result.pack(expand=True)
        
        self.lbl_prob = tk.Label(right_frame, text="확률: -", font=("Malgun Gothic", 14), bg="#f0f0f0")
        self.lbl_prob.pack(pady=5)
        
        self.lbl_tia = tk.Label(right_frame, text="몸통각도(TIA): -", font=("Malgun Gothic", 14), bg="#f0f0f0")
        self.lbl_tia.pack(pady=5)
        
        self.lbl_alert = tk.Label(right_frame, text="", font=("Malgun Gothic", 11, "bold"), bg="#f0f0f0", fg="#e74c3c")
        self.lbl_alert.pack(pady=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not file_path: return
        self.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((450, 450))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(225, 225, image=self.tk_img)

    def run_analysis(self):
        if not self.image_path: return
        img_infer = tf.keras.preprocessing.image.load_img(self.image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_infer) / 255.0
        prob = self.model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0]
        prediction = "GOOD" if prob > THRESHOLD else "BAD"
        conf = prob if prob > THRESHOLD else 1 - prob
        color = "#2ecc71" if prediction == "GOOD" else "#e74c3c"
        self.lbl_result.config(text=prediction, bg=color, fg="white")
        self.res_panel.config(bg=color)
        self.lbl_prob.config(text=f"확률(신뢰도): {conf*100:.1f}%")
        self.analyze_pose()

    def analyze_pose(self):
        img_cv = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w, _ = img_cv.shape
        if USE_LEGACY:
            results = self.mp_detector.process(img_rgb)
            all_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = self.mp_detector.detect(mp_image)
            all_landmarks = results.pose_landmarks[0] if results.pose_landmarks else []

        if not all_landmarks:
            self.lbl_tia.config(text="몸통각도(TIA): 분석 불가")
            return

        try:
            get_xy = lambda idx: (all_landmarks[idx].x * w, all_landmarks[idx].y * h)
            sh_l, sh_r = get_xy(11), get_xy(12)
            hp_l, hp_r = get_xy(23), get_xy(24)
            sh_c = ((sh_l[0] + sh_r[0]) / 2, (sh_l[1] + sh_r[1]) / 2)
            hp_c = ((hp_l[0] + hp_r[0]) / 2, (hp_l[1] + hp_r[1]) / 2)
            angle = abs(math.atan2(sh_c[0] - hp_c[0], sh_c[1] - hp_c[1]) * 180 / math.pi)
            self.lbl_tia.config(text=f"몸통각도(TIA): {angle:.1f}°")
            
            vis_img = img_rgb.copy()
            for pt in [sh_l, sh_r, hp_l, hp_r]:
                cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 10, (255, 255, 0), -1)
            cv2.line(vis_img, (int(sh_c[0]), int(sh_c[1])), (int(hp_c[0]), int(hp_c[1])), (255, 0, 255), 5)
            pil_vis = Image.fromarray(vis_img)
            pil_vis.thumbnail((450, 450))
            self.tk_img = ImageTk.PhotoImage(pil_vis)
            self.canvas.create_image(225, 225, image=self.tk_img)
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureTestApp(root)
    root.mainloop()
