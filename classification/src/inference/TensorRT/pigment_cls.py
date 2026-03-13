from datetime import datetime
start = datetime.now()

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from datetime import datetime
from preprocess.crop_data import crop_cheeks_from_detection_result
from utils.Mediapipe import MediaPipe
import pandas as pd
import gc

ENGINE_PATH = '/home/work/hocheol_dir/workspace/inference/TensorRT/TRT/pig_3ch/250821_162100_S/epoch10.trt'
IMAGE_PATH = '/home/work/hocheol_dir/workspace/datasets/_origin_data/030_data/data/0989_face.jpg'
RESIZE = 384
CLASS_NAMES = ['0', '1', '2', '3', '4']
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

output_path = f"/home/work/hocheol_dir/workspace/inference/results/v0.2/pig_3ch/{os.path.basename(os.path.dirname(ENGINE_PATH))}/{ENGINE_PATH.split('/')[-1].split('_')[0]}_tensorrt.csv"

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class Inferencer:
    def __init__(self, engine_path, resize=(384, 384)):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.resize = resize
        self.face_landmarker = MediaPipe()
    
    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, input_np):
        bindings, inputs, output_buffers = [], [], {}

        for binding in self.engine:
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            shape = self.context.get_tensor_shape(binding)
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
        self.context.execute_v2(bindings)

        output_key = list(output_buffers.keys())[0]
        host_mem, device_mem = output_buffers[output_key]
        cuda.memcpy_dtoh(host_mem, device_mem)

        probs = host_mem
        probs = softmax(probs)
        pred_class = np.argmax(probs)

        return pred_class, probs

    def run(self, image_path):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]
        left_cheek, right_cheek = crop_cheeks_from_detection_result(landmarks, mp_image)

        preprocessed_left = self.preprocess(left_cheek)
        preprocessed_right = self.preprocess(right_cheek)

        left_pred, left_probs = self.predict(preprocessed_left)
        right_pred, right_probs = self.predict(preprocessed_right)

        return (left_pred, left_probs), (right_pred, right_probs)

if __name__ == "__main__":
    inferencer = Inferencer(ENGINE_PATH, resize=(RESIZE, RESIZE))
    (left_pred, left_probs), (right_pred, right_probs) = inferencer.run(IMAGE_PATH)

    col_names = ['image_path', 'direction', 'preds'] + [f"conf_{i}" for i in CLASS_NAMES]
    left_outputs = [IMAGE_PATH, 'left', CLASS_NAMES[left_pred]] + list(left_probs)
    right_outputs = [IMAGE_PATH, 'right', CLASS_NAMES[right_pred]] + list(right_probs)

    df = pd.DataFrame([left_outputs, right_outputs], columns=col_names)
    print(df)
    df.to_csv(output_path, index=False)

    del inferencer
    gc.collect()

end = datetime.now()
print(f"소요 시간: {end - start}")