import os
import cv2
import csv
import numpy as np
import tensorflow as tf
import mediapipe as mp
from inference.crop_data import crop_cheeks_and_forehead_from_image, crop_cheeks_from_image
from datetime import datetime

'''
tflite로 저장할 당시의 input size로 지정해야함

EfficientNetV2-B0 : (224, 224)
EfficientNetV2-B1 : (240, 240)
EfficientNetV2-B2 : (260, 260)
EfficientNetV2-B3 : (300, 300)
EfficientNetV2-S : (384, 384)
EfficientNetV2-M : (480, 480)
EfficientNetV2-L : (480, 480)
EfficientNetV2-XL (비공식) : (512, 512)

TASK:
    pigmentation
    age_only1
'''

TASK = 'age_only1'

ROOT_DIR = "/home/work/hocheol_dir/workspace/TFLITE/tflite_results"
images = [
    '/home/work/hocheol_dir/data/045_data/045.new_origin_data/0033/F0033_IND_GM_74_0_01.JPG'
]

# interpreter에서 입력 이미지 받을 수 있지만 명시적으로 RESIZE 하면서 한 번 확인하는 용도로 쓰기
RESIZE = (384, 384)
# IMAGE_PATH = "/home/work/hocheol_dir/workspace/data/0000_CRS_19_01.jpg"
MODEL_PATH = "/home/work/hocheol_dir/workspace/TFLITE/age_only1/epoch100_loss81.6944.tflite"
OUTPUT_CSV_PATH = f"{TASK}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# ===== Mediapipe FaceMesh 초기화 =====
mp_face_mesh = mp.solutions.face_mesh

# ===== 1x1 Convolution 레이어 정의 =====
# 만약 여러장의 추론이 들어가게 된다면 일관성 유지를 위해 convolution layer 고정
conv1x1_layer = tf.keras.layers.Conv2D(3, (1, 1), padding='same')

# ===== 메인 함수 =====
def predict(image_path, model_path, output_csv_path):
    if TASK == 'pigmentation':
        # 볼 crop
        left_cheek, right_cheek = crop_cheeks_from_image(image_path)

        # resize
        left_cheek_resized = cv2.resize(left_cheek, RESIZE).astype(np.float32)
        right_cheek_resized = cv2.resize(right_cheek, RESIZE).astype(np.float32)

        # concat 후 6채널로 만들기
        concat_img = np.concatenate([left_cheek_resized, right_cheek_resized], axis=-1)  # (H, W, 6)
        concat_img = np.expand_dims(concat_img, axis=0)  # (1, H, W, 6)

        # 1x1 Conv로 3채널로 변환
        input_tensor = tf.convert_to_tensor(concat_img, dtype=tf.float32)
        # merged_img = conv1x1_layer(input_tensor) # 모델 내부에 6ch -> 3ch 변환까지 포함되어 있다.

        # TFLite 모델 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # 입출력 tensor 정보 가져오기
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        model_input_size = tuple(input_details[0]['shape'][1:3])

        # 입력 데이터 전처리 (merged_img는 float32이어야 하며 shape도 맞춰야 함)
        # 예시: (1, 224, 224, 3)로 맞추고 float32 변환
        input_data = input_tensor.numpy()

        # 입력 tensor에 데이터 설정
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # 모델 실행
        interpreter.invoke()

        # 출력값 가져오기
        output_data = interpreter.get_tensor(output_details[0]['index'])

        probabilities = output_data[0]

        # 클래스 예측
        pred_class = np.argmax(probabilities)

        col_names = ["image_path", "predicted_class"] + [f"conf_{i}" for i in range(5)]
        outputs = [image_path, pred_class] + list(probabilities)

    elif TASK == 'age':
        forehead, left_cheek, right_cheek = crop_cheeks_and_forehead_from_image(image_path)
        forehead_resized = cv2.resize(forehead, RESIZE)
        left_cheek_resized = cv2.resize(left_cheek, RESIZE)
        right_cheek_resized = cv2.resize(right_cheek, RESIZE)

        forehead_resized = preprocess_input(forehead_resized.astype(np.float32))
        left_cheek_resized = preprocess_input(left_cheek_resized.astype(np.float32))
        right_cheek_resized = preprocess_input(right_cheek_resized.astype(np.float32))
        
        forehead_resized = np.expand_dims(forehead_resized, axis=0)
        left_cheek_resized = np.expand_dims(left_cheek_resized, axis=0)
        right_cheek_resized = np.expand_dims(right_cheek_resized, axis=0)

        forehead_input = tf.convert_to_tensor(forehead_resized, dtype=tf.float32).numpy()
        left_cheek_input = tf.convert_to_tensor(left_cheek_resized, dtype=tf.float32).numpy()
        right_cheek_input = tf.convert_to_tensor(right_cheek_resized, dtype=tf.float32).numpy()

        # TFLite 모델 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        model_input_size = tuple(input_details[0]['shape'][1:3])

        interpreter.set_tensor(input_details[0]['index'], forehead_input)
        interpreter.set_tensor(input_details[1]['index'], left_cheek_input)
        interpreter.set_tensor(input_details[2]['index'], right_cheek_input)
        
        interpreter.invoke()
        
        forehead_output = interpreter.get_tensor(output_details[0]['index'])
        left_cheek_output = interpreter.get_tensor(output_details[1]['index'])
        right_cheek_output = interpreter.get_tensor(output_details[2]['index'])

        col_names = ['image_path', 'forehead_age', 'left_cheek_age', 'right_cheek_age']
        outputs = [image_path, forehead_output[0][0], left_cheek_output[0][0], right_cheek_output[0][0]]
        print(f"forehead: {forehead_output}")
    
    elif TASK == 'age_only1':
        forehead, left_cheek, right_cheek = crop_cheeks_and_forehead_from_image(image_path)
        forehead_resized = cv2.resize(forehead, RESIZE)
        left_cheek_resized = cv2.resize(left_cheek, RESIZE)
        right_cheek_resized = cv2.resize(right_cheek, RESIZE)

        forehead_processed = np.expand_dims(preprocess_input(forehead_resized.astype(np.float32)), axis=0)
        left_cheek_processed = np.expand_dims(preprocess_input(left_cheek_resized.astype(np.float32)), axis=0)
        right_cheek_processed = np.expand_dims(preprocess_input(right_cheek_resized.astype(np.float32)), axis=0)
        
        forehead_input = tf.convert_to_tensor(forehead_processed, dtype=tf.float32).numpy()
        left_cheek_input = tf.convert_to_tensor(left_cheek_processed, dtype=tf.float32).numpy()
        right_cheek_input = tf.convert_to_tensor(right_cheek_processed, dtype=tf.float32).numpy()

        # TFLite 모델 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_images = [forehead_input, left_cheek_input, right_cheek_input]
        outputs = []

        for img in input_images:
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            outputs.append(output_data[0][0])
        
        col_names = ['image_path', 'forehead_age', 'left_cheek_age', 'right_cheek_age']
        outputs = [image_path] + outputs

    try:
        print("Input Details:", input_details)
        print("Output Details:", output_details)
    except Exception as e:
        print(f"입출력 tensor 정보 가져오기 실패: {e}")

    # 14. CSV 저장
    # with open(output_csv_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(col_names)
    #     writer.writerow(outputs)

    # print(f"예측 완료! 결과는 {output_csv_path}에 저장되었습니다.")
    for i in range(len(col_names)):
        print(f"{col_names[i]}: {outputs[i]}")


# ===== 사용 예시 =====
if __name__ == "__main__":
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(ROOT_DIR, start_time))

    for image in images:
        output_csv_path = os.path.join(ROOT_DIR, start_time, f"{TASK}_{image.split('.')[0]}.csv")
        image_path = os.path.join(ROOT_DIR, image)
        predict(image_path, MODEL_PATH, output_csv_path)
    end_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")