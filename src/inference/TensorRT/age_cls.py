from datetime import datetime
start = datetime.now()

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from preprocess.face_crop_roll import age_preprocess
from utils.Mediapipe import MediaPipe
import pandas as pd

# models/age_cls.py의 모델 보면 0번이 male_output, 1번이 female_output임을 알 수 있다.

'''
GENDER:
    male
    female
'''

ENGINE_PATH = '/home/work/hocheol_dir/workspace/inference/TensorRT/TRT/age_cls/250808_221037_S/epoch70_date250808_221037.trt'
IMAGE_PATH = '/home/work/hocheol_dir/workspace/datasets/_origin_data/030_data/data/0989_face.jpg'
GENDER = 'male'
RESIZE = 384
CLASS_NAMES = ['10', '20', '30', '40', '50', '60', '70']
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

output_path = f"/home/work/hocheol_dir/workspace/inference/results/v0.2/age_cls/{os.path.basename(os.path.dirname(ENGINE_PATH))}/{ENGINE_PATH.split('/')[-1].split('_')[0]}_tensorrt.csv"

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class Inferencer:
    def __init__(self, engine_path, resize=(384, 384)):
        self.engine = load_engine(engine_path)
        self.resize = resize
        self.face_landmarker = MediaPipe()
    
    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, input_np, gender):
        context = self.engine.create_execution_context()

        # binding 준비
        bindings, inputs, output_buffers = [], [], {}

        for i, binding in enumerate(self.engine):
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            shape = context.get_tensor_shape(binding)
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if binding == 'input_layer':
                inputs.append((host_mem, device_mem))
            else:
                output_buffers[binding] = (host_mem, device_mem)
        
        np.copyto(inputs[0][0], input_np.ravel())
        cuda.memcpy_htod(inputs[0][1], inputs[0][0])

        context.execute_v2(bindings)

        if gender == 'male':
            output_key = 'output_0'
        elif gender == 'female':
            output_key = 'output_1'
        else:
            raise ValueError(f"{gender}은(는) 지원하지 않는 성별입니다.")
        
        if output_key not in output_buffers:
            raise ValueError(f"Output key {output_key}가 엔진에 존재하지 않습니다.")
        
        host_mem, device_mem = output_buffers[output_key]
        cuda.memcpy_dtoh(host_mem, device_mem)
        probs = host_mem

        pred_class = np.argmax(probs)

        return pred_class, probs

    def run(self, image_path, gender):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]

        preprocessed_image = age_preprocess(landmarks, mp_image)
        preprocessed_image = self.preprocess(preprocessed_image)

        pred, probs = self.predict(preprocessed_image, gender)

        return pred, probs


if __name__ == "__main__":
    inferencer = Inferencer(ENGINE_PATH, resize=(RESIZE, RESIZE))
    pred_class, probs = inferencer.run(IMAGE_PATH, GENDER)

    # print(f"이미지 로딩: {IMAGE_PATH}")
    # image = load_img(IMAGE_PATH, (RESIZE, RESIZE))

    # pred_class, probs = infer_single(engine, image, GENDER)
    col_names = ['image_path', 'predicted_class'] + [f"conf_{i}" for i in CLASS_NAMES]
    outputs = [IMAGE_PATH, CLASS_NAMES[pred_class]] + list(probs)

    print(f"예측 클래스: {CLASS_NAMES[pred_class]}")
    print(f"확률: {probs}")

    df = pd.DataFrame([outputs], columns=col_names)
    print(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

end = datetime.now()
print(f"소요 시간: {end - start}")