import tensorflow as tf
import os

# ===== 사용자 설정 =====
OUTPUT_DIR = "/home/work/hocheol_dir/workspace/saved_tflite_base_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 모델 이름과 input size 매핑
MODEL_CONFIGS = {
    'EfficientNetV2B0': (tf.keras.applications.EfficientNetV2B0, (224, 224)),
    'EfficientNetV2B1': (tf.keras.applications.EfficientNetV2B1, (240, 240)),
    'EfficientNetV2B2': (tf.keras.applications.EfficientNetV2B2, (260, 260)),
    'EfficientNetV2B3': (tf.keras.applications.EfficientNetV2B3, (300, 300)),
    'EfficientNetV2S' : (tf.keras.applications.EfficientNetV2S,  (384, 384)),
    'EfficientNetV2M' : (tf.keras.applications.EfficientNetV2M,  (480, 480)),
    'EfficientNetV2L' : (tf.keras.applications.EfficientNetV2L,  (480, 480)),
    # EfficientNetV2-XL은 공식 tf.keras.applications에 없음 (따로 작업 필요)
}
# =======================

# 변환 함수
def save_model_as_tflite(model, save_path):
    # ConcreteFunction 변환 (TFLite 변환 오류 방지용)
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, model.input_shape[1], model.input_shape[2], model.input_shape[3]], model.input.dtype)
    )

    # TFLite 변환
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    tflite_model = converter.convert()

    # 파일로 저장
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

# 전체 변환
for model_name, (model_fn, input_size) in MODEL_CONFIGS.items():
    print(f"🔵 변환 시작: {model_name}")

    # 1. EfficientNetV2 Backbone 가져오기 (분류기 제거)
    base_model = model_fn(
        include_top=False,
        weights='imagenet',
        input_shape=(input_size[0], input_size[1], 3)
    )

    # 2. 5-class 커스텀 헤드 붙이기
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

    # 3. 저장
    tflite_path = os.path.join(OUTPUT_DIR, f"{model_name}_5class.tflite")
    save_model_as_tflite(model, tflite_path)

    print(f"✅ 저장 완료: {tflite_path}")

print("🎉 모든 모델 변환 완료!")
