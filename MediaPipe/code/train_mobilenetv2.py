import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.utils.class_weight import compute_class_weight
import warnings

# 1. 환경 설정 및 로그 정리
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. 데이터 구성 및 라벨링 정보 로드
base_dir = r"C:\Users\user\Desktop\semi2_project\image_data\image_1st\train"
folders = [
    os.path.join(base_dir, "full_body"),
    os.path.join(base_dir, "ankle_visible")
]

filenames = []
labels = []

for folder in folders:
    json_path = os.path.join(folder, "_annotations.coco.json")
    if not os.path.exists(json_path):
        continue
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # category_id mapping: 1(bad) -> 0, 2(good) -> 1
    id_to_label = {ann['image_id']: (0 if ann['category_id'] == 1 else 1) for ann in coco_data.get('annotations', [])}
            
    images = coco_data.get('images', [])
    for img_info in images:
        filepath = os.path.join(folder, img_info['file_name'])
        if os.path.exists(filepath) and img_info['id'] in id_to_label:
            filenames.append(filepath)
            labels.append(id_to_label[img_info['id']])

df = pd.DataFrame({'filepath': filenames, 'label': labels})
df['label_str'] = df['label'].astype(str)

# 3. 데이터 분할 (80/10/10) 및 클래스 가중치 계산
# Train 80% / Val 10% / Test 10%
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Data Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# 클래스 가중치는 오직 train_df 기준으로만 계산 (불균형 보정)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label']
)
class_weight_dict = dict(enumerate(class_weights))

# 4. 이미지 전처기 및 증강
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
# 검증 및 테스트는 증강 없이 rescale만 적용
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df, x_col='filepath', y_col='label_str',
    target_size=(224, 224), batch_size=32, class_mode='binary'
)

val_generator = test_datagen.flow_from_dataframe(
    val_df, x_col='filepath', y_col='label_str',
    target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df, x_col='filepath', y_col='label_str',
    target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False
)

# 5. 모델 설계 (Transfer Learning)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
base_model.trainable = False # 1차 단계는 백본 동결

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 6. 평가 함수 정의
def evaluate_and_report(model, generator, title, threshold=0.60):
    print(f"\n{'='*20} {title} {'='*20}")
    y_pred_prob = model.predict(generator, verbose=0)
    y_true = generator.classes
    y_pred = (y_pred_prob > threshold).astype(int).flatten()
    
    print(f"[Threshold: {threshold}]")
    print(classification_report(y_true, y_pred, target_names=['bad', 'good']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    acc = accuracy_score(y_true, y_pred)
    bad_recall = recall_score(y_true, y_pred, pos_label=0)
    good_recall = recall_score(y_true, y_pred, pos_label=1)
    
    print("-" * 30)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Bad Recall:      {bad_recall:.4f}")
    print(f"Good Recall:     {good_recall:.4f}")
    print("-" * 30)
    return acc

# 7. Stage 1: Base Training (Frozen Backbone)
print("\n>>> Stage 1: Training Head only (Backbone Frozen)...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

model.save("posture_mobilenetv2_final.h5")
evaluate_and_report(model, test_generator, "Final Test Report (Stage 1)")

# 8. Stage 2: Fine-tuning (Unfreeze Top 30 Layers)
print("\n>>> Stage 2: Fine-tuning (Unfreezing top 30 layers)...")
base_model.trainable = True
# 하위 레이어는 계속 동결 유지 (약 150여개 중 하위 120개 정도)
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 미세 조정을 위해 매우 낮은 학습률 적용
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

model.save("posture_mobilenetv2_finetuned.h5")
evaluate_and_report(model, test_generator, "Final Test Report (Stage 2 - Fine-tuned)")
