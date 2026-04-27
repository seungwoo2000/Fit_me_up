import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
import warnings

# 1. 설정 및 경고 필터링
warnings.filterwarnings('ignore')

# 2. 데이터 로드 (MediaPipe로 추출된 관절 좌표 데이터셋)
csv_path = r"C:\Users\user\Desktop\semi2_project\image_data\image_1st\labeled_dataset_v3.csv"
df = pd.read_csv(csv_path)

# 특성(X)과 라벨(y) 분리
# 파일명 제외, 라벨을 0(bad)과 1(good)로 수치화
X = df.drop(columns=['파일명', '라벨'])
y = df['라벨'].map({'good': 1, 'bad': 0})

# 3. 데이터셋 분할 (학습 8:2 테스트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 정규화 및 클래스 가중치 계산
# MLP는 입력 데이터의 스케일에 민감하므로 StandardScaler 필수 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 데이터 불균형 문제를 해결하기 위해 샘플 가중치를 부여합니다.
sample_weights = compute_sample_weight('balanced', y_train)

# 5. MLP(Multi-Layer Perceptron) 모델 구축
# - hidden_layer_sizes: 3개의 은닉층 구성 (128, 64, 32)
# - activation: ReLU 활성화 함수 사용
# - early_stopping: 과적합이 발생하기 전 학습을 조기 종료
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=5
)

# 6. 학습 (샘플 가중치 적용)
mlp.fit(X_train_scaled, y_train, sample_weight=sample_weights)

# 7. 평가 및 파이프라인 저장
y_pred = mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred, target_names=['bad (0)', 'good (1)'])

# 스케일러와 모델을 하나의 파이프라인으로 묶어 저장 (추론 시 편리함)
pipeline = Pipeline([
    ('scaler', scaler),
    ('mlp', mlp)
])
joblib.dump(pipeline, "posture_mlp_final.pkl")

print("="*40)
print(f"Test Accuracy: {acc*100:.2f}%")
print("="*40)
print("\n[Confusion Matrix]")
print(cm)
print("\n[Classification Report]")
print(cr)
