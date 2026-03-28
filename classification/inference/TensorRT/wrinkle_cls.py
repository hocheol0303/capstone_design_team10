from datetime import datetime
start = datetime.now()

import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.Mediapipe import MediaPipe
import pandas as pd
from utils.constants import SECTORS, SECTOR2IDX
import sys
sys.path.append('/home/work/hocheol_dir/workspace/preprocess/v0.2/wrinkle')
from wrinkle_crop import wrinkle_preprocess


ENGINE_PATH = '/home/work/hocheol_dir/workspace/inference/TensorRT/TRT/wrinkle/250819_002635_S/epoch80_250819_002635.trt'
IMAGE_PATH = '/home/work/hocheol_dir/workspace/datasets/_origin_data/030_data/data/1053_face.jpg'
RESIZE = 384
CLASS_NAMES = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4']
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

output_path = f"/home/work/hocheol_dir/workspace/inference/results/v0.2/wrinkle/{os.path.basename(os.path.dirname(ENGINE_PATH))}/{ENGINE_PATH.split('/')[-1].split('_')[0]}_tensorrt.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def load_engine(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class Inferencer:
    def __init__(self, engine_path, resize=384):
        self.engine = load_engine(engine_path)
        self.resize = (resize, resize)
        self.face_landmarker = MediaPipe()

    def preprocess(self, img):
        img = cv2.resize(img, self.resize)
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, input_np, sector):
        sector_id = np.array(SECTOR2IDX[sector], dtype=np.int32).reshape((1,))
        context = self.engine.create_execution_context()

        bindings, inputs, output_buffers = [], {}, {}
        for i, binding in enumerate(self.engine):
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            shape = context.get_tensor_shape(binding)
            size = trt.volume(shape)

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            print(binding)
            if 'input' in binding:
                inputs[binding] = (host_mem, device_mem)
            else:
                output_buffers[binding] = (host_mem, device_mem)
        
        np.copyto(inputs['image_input'][0], input_np.ravel())
        np.copyto(inputs['sector_input'][0], sector_id.ravel())
        cuda.memcpy_htod(inputs['image_input'][1], inputs['image_input'][0])
        cuda.memcpy_htod(inputs['sector_input'][1], inputs['sector_input'][0])

        context.execute_v2(bindings)
        
        host_mem, device_mem = output_buffers['output_0']
        cuda.memcpy_dtoh(host_mem, device_mem)
        probs = host_mem

        pred_class = np.argmax(probs)

        return pred_class, probs
    
    def run(self, image_path):
        with open(image_path, 'rb') as f:
            image = f.read()
        
        mp_image = self.face_landmarker.create_mp_image(image)
        detection_result = self.face_landmarker.landmarker.detect(mp_image)
        landmarks = detection_result.face_landmarks[0]

        parts = dict(zip(SECTORS, wrinkle_preprocess(landmarks, mp_image)))

        for key, value in parts.items():
            parts[key] = self.preprocess(value)
        
        preds, probs = {}, {}
        for sector in SECTORS:
            preds[sector], probs[sector] = self.predict(parts[sector], sector)
        
        return preds, probs

if __name__ == "__main__":
    inferencer = Inferencer(ENGINE_PATH, resize=RESIZE)
    preds, probs = inferencer.run(IMAGE_PATH)
    col_names = ['image_path', 'sector', 'predicted_class'] + CLASS_NAMES
    rows = []
    for sector in SECTORS:
        rows.append([IMAGE_PATH, sector, preds[sector]] + list(probs[sector]))
    df = pd.DataFrame(rows, columns=col_names)
    print(df)
    df.to_csv(output_path, index=False)

end = datetime.now()
print(f"소요 시간: {end - start}")