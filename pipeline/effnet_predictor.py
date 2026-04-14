"""
effnet_predictor.py
===================
TFLite 기반 EfficientNet 추론 모듈.

모델별 특성
-----------
age_reg       : 단일 입력(face crop), 이중 출력 [male_pred, female_pred] shape (1,1) 각각
pigment_reg   : 단일 입력(cheek crop), 단일 출력 shape (1,1)
wrinkle_reg   : 이중 입력 [image_input, sector_input(int32)], 단일 출력 shape (1,1)
homogenity_reg: 단일 입력(face crop), 이중 출력 [rad_out, tex_out] shape (1,1) 각각

입력 전처리
-----------
모든 모델의 입력 dtype = float32, 값 범위 [0, 255] (EfficientNetV2 내부 정규화 포함).
crop ndarray(RGB uint8) → resize → float32 변환 후 배치 차원 추가.
"""

import numpy as np
import cv2
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# wrinkle sector 순서 (model 학습 시 기준)
SECTORS = ["forehead", "right_eye", "left_eye", "nasolabial", "perioral", "right_vol", "left_vol"]
SECTOR2IDX = {s: i for i, s in enumerate(SECTORS)}


def _load_interpreter(model_path: str) -> tf.lite.Interpreter:
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def _resize_and_to_float(crop: np.ndarray, input_size: tuple) -> np.ndarray:
    """(H, W, 3) uint8 → (1, H', W', 3) float32 [0~255]."""
    if crop is None or crop.size == 0:
        return None
    resized = cv2.resize(crop, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)[np.newaxis]   # (1, H, W, 3)


def _run_single_input(interp: tf.lite.Interpreter, img_tensor: np.ndarray) -> np.ndarray:
    """단일 입력 → 첫 번째 출력 tensor 반환."""
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.set_tensor(input_details[0]['index'], img_tensor)
    interp.invoke()
    return interp.get_tensor(output_details[0]['index'])


class EffNetPredictor:
    """
    age / pigment / wrinkle TFLite 모델을 관리하고 추론을 수행.

    Parameters
    ----------
    config  : dict  – config.yaml 내 'tflite' 섹션
    gender  : str   – 'male' | 'female'  (age 이중 헤드 선택에 사용)
    """

    def __init__(self, config: dict, gender: str):
        self._gender = gender.lower()

        tflite_cfg = config['tflite']

        pigment_path    = str(Path(config.get('root_dir', '.')) / tflite_cfg['pigment']['model'])
        wrinkle_path    = str(Path(config.get('root_dir', '.')) / tflite_cfg['wrinkle']['model'])
        age_path        = str(Path(config.get('root_dir', '.')) / tflite_cfg['age']['model'])
        homogenity_path = str(Path(config.get('root_dir', '.')) / tflite_cfg['homogenity']['model'])

        # age 1개 + pigment 2개(left/right) + wrinkle 7개(sector별) + homogenity 1개 병렬 로드
        n_workers = 1 + 2 + len(SECTORS) + 1
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            f_age            = ex.submit(_load_interpreter, age_path)
            f_pig_left       = ex.submit(_load_interpreter, pigment_path)
            f_pig_right      = ex.submit(_load_interpreter, pigment_path)
            f_wrinkle_list   = [ex.submit(_load_interpreter, wrinkle_path) for _ in SECTORS]
            f_homogenity     = ex.submit(_load_interpreter, homogenity_path)

            self._age_interp         = f_age.result()
            self._pigment_interps    = {"left": f_pig_left.result(), "right": f_pig_right.result()}
            self._wrinkle_interps    = {s: f.result() for s, f in zip(SECTORS, f_wrinkle_list)}
            self._homogenity_interp  = f_homogenity.result()

        self._age_input_size        = tuple(tflite_cfg['age']['input_size'])
        self._pigment_input_size    = tuple(tflite_cfg['pigment']['input_size'])
        self._wrinkle_input_size    = tuple(tflite_cfg['wrinkle']['input_size'])
        self._homogenity_input_size = tuple(tflite_cfg['homogenity']['input_size'])

        # homogenity 출력 인덱스 매핑: :0 → rad_out, :1 → tex_out
        self._homogenity_output_map = {}
        for detail in self._homogenity_interp.get_output_details():
            name = detail['name']
            if 'rad_out' in name or name.endswith(':0'):
                self._homogenity_output_map['rad_out'] = detail['index']
            elif 'tex_out' in name or name.endswith(':1'):
                self._homogenity_output_map['tex_out'] = detail['index']

        # wrinkle 입력 텐서 순서 확인 (image vs sector) — 첫 번째 인스턴스 기준
        _sample_wrinkle = next(iter(self._wrinkle_interps.values()))
        self._wrinkle_input_details  = _sample_wrinkle.get_input_details()
        self._wrinkle_output_details = _sample_wrinkle.get_output_details()
        self._wrinkle_sec_idx, self._wrinkle_img_idx = self._resolve_wrinkle_inputs()

        # age 출력 순서 확인 (male=0, female=1)
        self._age_output_details = self._age_interp.get_output_details()

    def _resolve_wrinkle_inputs(self):
        sec_idx = self._wrinkle_input_details[0]['index']
        img_idx = self._wrinkle_input_details[1]['index']
        return sec_idx, img_idx

    # ── 나이 예측 ─────────────────────────────────────────────────────────────

    def predict_age(self, face_crop: np.ndarray) -> float | None:
        """
        Parameters
        ----------
        face_crop : (H, W, 3) RGB uint8

        Returns
        -------
        float  – 예측 나이 (연속값)
        None   – crop이 비어 있는 경우
        """
        img = _resize_and_to_float(face_crop, self._age_input_size)
        if img is None:
            return None

        interp = self._age_interp
        input_details = interp.get_input_details()
        interp.set_tensor(input_details[0]['index'], img)
        interp.invoke()

        # 출력은 접미사로 판단
        for detail in self._age_output_details:
            if self._gender == 'male' and detail['name'].endswith(':0'):
                pred = interp.get_tensor(detail['index'])
            elif self._gender == 'female' and detail['name'].endswith(':1'):
                pred = interp.get_tensor(detail['index'])

        return float(pred[0, 0])

    # ── 색소 예측 ─────────────────────────────────────────────────────────────

    def predict_pigment(self, pigment_crops: dict) -> dict:
        """
        Parameters
        ----------
        pigment_crops : {"left": ndarray, "right": ndarray}

        Returns
        -------
        {"left": float | None, "right": float | None}
        """
        def _infer(side: str, crop: np.ndarray):
            img = _resize_and_to_float(crop, self._pigment_input_size)
            if img is None:
                return side, None
            pred = _run_single_input(self._pigment_interps[side], img)
            return side, float(pred[0, 0])

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = {side: ex.submit(_infer, side, crop) for side, crop in pigment_crops.items()}
            return {side: f.result()[1] for side, f in futures.items()}

    # ── 균질도 예측 ───────────────────────────────────────────────────────────

    def predict_homogenity(self, face_crop: np.ndarray) -> dict:
        """
        Parameters
        ----------
        face_crop : (H, W, 3) RGB uint8

        Returns
        -------
        {"radiance": float | None, "texture": float | None}
        """
        img = _resize_and_to_float(face_crop, self._homogenity_input_size)
        if img is None:
            return {"radiance": None, "texture": None}

        interp = self._homogenity_interp
        input_details = interp.get_input_details()
        interp.set_tensor(input_details[0]['index'], img)
        interp.invoke()

        rad_score = float(interp.get_tensor(self._homogenity_output_map['rad_out'])[0, 0])
        tex_score = float(interp.get_tensor(self._homogenity_output_map['tex_out'])[0, 0])
        return {"radiance": rad_score, "texture": tex_score}

    # ── 주름 예측 ─────────────────────────────────────────────────────────────

    def predict_wrinkle(self, wrinkle_crops: dict) -> dict:
        """
        Parameters
        ----------
        wrinkle_crops : {sector: ndarray | None}

        Returns
        -------
        {sector: float | None}
        """
        def _infer(sector: str, crop: np.ndarray):
            img = _resize_and_to_float(crop, self._wrinkle_input_size)
            if img is None:
                return sector, None
            sector_id = np.array([SECTOR2IDX[sector]], dtype=np.int8).reshape((-1, 1))
            interp = self._wrinkle_interps[sector]
            out_details = interp.get_output_details()
            interp.set_tensor(self._wrinkle_img_idx, img)
            interp.set_tensor(self._wrinkle_sec_idx, sector_id)
            interp.invoke()
            pred = interp.get_tensor(out_details[0]['index'])
            return sector, float(pred[0, 0])

        with ThreadPoolExecutor(max_workers=len(SECTORS)) as ex:
            futures = {s: ex.submit(_infer, s, wrinkle_crops.get(s)) for s in SECTORS}
            return {s: futures[s].result()[1] for s in SECTORS}
