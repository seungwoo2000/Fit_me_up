import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import sys

# ==========================================
# 0. 설정 및 경로
# ==========================================
# 실행 환경에 맞춰 경로를 수정하세요.
MODEL_PATH = "posture_mlp_final.keras"
SCALER_PATH = "posture_scaler.pkl"
THRESHOLD = 0.35  # 최종 확정 임계값

# ==========================================
# 1. 유틸리티 함수
# ==========================================
def vertical_angle(p1, p2):
    """두 점 사이의 수직선 기준 각도 계산"""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return np.degrees(np.arctan2(abs(dx), abs(dy)))

class PosturePredictor:
    def __init__(self, model_path, scaler_path):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Error: 모델({model_path}) 또는 스케일러({scaler_path}) 파일을 찾을 수 없습니다.")
            sys.exit(1)
            
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.mp_pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    def predict(self, frame):
        results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None, "No Pose Detected", None
        
        landmarks = results.pose_landmarks.landmark
        
        # 귓구멍 기준 CVA / TIA 피처 추출
        # 7:왼귀, 8:오른귀, 11:왼어깨, 12:오른어깨, 23:왼골반, 24:오른골반
        
        # 1. 귀 선택 (Visibility 높은 쪽)
        ear = (landmarks[7].x, landmarks[7].y) if landmarks[7].visibility >= landmarks[8].visibility else (landmarks[8].x, landmarks[8].y)
        
        # 2. 어깨 선택
        sh = (landmarks[11].x, landmarks[11].y) if landmarks[11].visibility >= landmarks[12].visibility else (landmarks[12].x, landmarks[12].y)
        
        # 3. 골반 선택
        hip = (landmarks[23].x, landmarks[23].y) if landmarks[23].visibility >= landmarks[24].visibility else (landmarks[24].x, landmarks[24].y)
        
        cva = vertical_angle(ear, sh)
        tia = vertical_angle(sh, hip)
        
        # 정규화 및 추론
        features = np.array([[cva, tia]])
        features_sc = self.scaler.transform(features)
        
        prob_good = self.model.predict(features_sc, verbose=0)[0][0]
        prob_bad = 1.0 - prob_good
        
        label = "Bad" if prob_bad >= THRESHOLD else "Good"
        
        return label, prob_bad, (cva, tia)

# ==========================================
# 2. 메인 실행 (웹캠 또는 이미지 테스트)
# ==========================================
def main():
    predictor = PosturePredictor(MODEL_PATH, SCALER_PATH)
    cap = cv2.VideoCapture(0)  # 웹캠 연결
    
    print(f"--- 자세 분석 시작 (임계값: {THRESHOLD}) ---")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        label, prob_bad, angles = predictor.predict(frame)
        
        color = (0, 0, 255) if label == "Bad" else (0, 255, 0)
        
        if angles:
            cv2.putText(frame, f"Posture: {label} ({prob_bad*100:.1f}%)", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"CVA: {angles[0]:.1f}, TIA: {angles[1]:.1f}", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        cv2.imshow('Posture Analysis (CVA/TIA Only)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
