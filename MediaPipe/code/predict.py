import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import warnings

# 로그 레벨 조정 (TensorFlow 경고 메시지 방지)
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 전역 설정
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/posture_mobilenetv2_finetuned.h5")
THRESHOLD = 0.60
model = None # 싱글톤 패턴을 위한 전역 변수

def predict_posture(image_path):
    """
    이미지 경로를 받아서 자세 판정 결과를 반환하는 함수
    TA팀 연동용 (YOLO 크롭 이미지 입력 가능)
    
    Args:
        image_path: 이미지 파일 경로 (str)
    
    Returns:
        dict: {
            "label": "good" or "bad",
            "confidence": 확률값 (float),
            "threshold": 0.60
        }
    """
    global model
    
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    # 1. 모델 로드 (최초 1회만 수행)
    if model is None:
        if not os.path.exists(MODEL_PATH):
            return {"error": f"Model not found: {MODEL_PATH}"}
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            return {"error": f"Model load failed: {str(e)}"}

    # 2. 이미지 전처리
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 3. 예측
        prediction = model.predict(img_array, verbose=0)
        score = prediction[0][0] # 1(good)에 가까울수록 좋은 자세

        # 4. 결과 정리 (TA팀 요청 형식)
        label = "good" if score > THRESHOLD else "bad"
        confidence = float(score if score > THRESHOLD else 1 - score)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "threshold": THRESHOLD
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# === 사용 예시 ===
# from predict import predict_posture
# result = predict_posture("test.jpg")
# if "error" not in result:
#     print(result["label"])      # good or bad
#     print(result["confidence"]) # 0.87
# ================

if __name__ == "__main__":
    # 사용 예시: python predict.py <이미지경로>
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        test_path = sys.argv[1]
        res = predict_posture(test_path)
        
        if "error" in res:
            print(f"Error: {res['error']}")
        else:
            print("-" * 50)
            print(f"파일: {os.path.basename(test_path)}")
            print(f"결과: {res['label'].upper()}")
            print(f"확률(신뢰도): {res['confidence'] * 100:.2f}%")
            print(f"임계값: {res['threshold']} (Bad Recall 우선 설정)")
            print("-" * 50)
