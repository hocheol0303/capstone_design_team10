"""
pipeline.py
===========
원본 이미지 한 장(또는 디렉토리 배치) → age / pigment / wrinkle / 볼처짐 / 턱처짐 예측.

사용법
------
단일 이미지
    python pipeline.py --image /path/to/image.jpg [--gender male]

배치
    python pipeline.py --image_dir /path/to/dir [--gender male] [--output_csv result.csv]

Python API
    from pipeline import SkinPipeline
    pipe = SkinPipeline("config.yaml")
    result = pipe.predict_single("/path/to/img.jpg", gender="male")
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

start = datetime.now()

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess import FacePreprocessor
from effnet_predictor import EffNetPredictor
from sagging_predictor import SaggingPredictor


# ── 결과 평탄화 헬퍼 ──────────────────────────────────────────────────────────

def _flatten_result(result: dict, filename: str = "") -> dict:
    """
    predict_single 반환값 → CSV 1행짜리 flat dict 변환.
    None 값은 빈 문자열 "" 로 변환.
    """
    def _f(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return round(v, 4)
        return v

    row = {"filename": filename}

    # age
    row["age"] = _f(result.get("age"))

    # pigment
    pig = result.get("pigment", {})
    row["pigment_left"]  = _f(pig.get("left"))
    row["pigment_right"] = _f(pig.get("right"))

    # homogenity
    hom = result.get("homogenity", {})
    row["homogenity_radiance"] = _f(hom.get("radiance"))
    row["homogenity_texture"]  = _f(hom.get("texture"))

    # wrinkle
    wrk = result.get("wrinkle", {})
    for sector in ["forehead", "right_eye", "left_eye", "nasolabial", "perioral", "right_vol", "left_vol"]:
        row[f"wrinkle_{sector}"] = _f(wrk.get(sector))

    # sagging
    chk = result.get("cheek_sagging", {})
    row["cheek_right"] = _f(chk.get("right"))   # 0~50
    row["cheek_left"]  = _f(chk.get("left"))    # 0~50
    row["cheek_score"] = _f(chk.get("total"))   # 0~100

    chi = result.get("chin_sagging", {})
    row["chin_right"] = _f(chi.get("right"))    # 0~50
    row["chin_left"]  = _f(chi.get("left"))     # 0~50
    row["chin_score"] = _f(chi.get("total"))    # 0~100

    # gender
    row["gender"] = _f(result.get("gender"))

    row["valid_sagging"] = result.get("valid_sagging", False)
    return row


# ── 파이프라인 클래스 ─────────────────────────────────────────────────────────

class SkinPipeline:
    """
    Parameters
    ----------
    config_path : str | Path  – config.yaml 경로
    gender      : str | None  – 'male' | 'female'. None이면 config 값 사용.
    """

    def __init__(self, config_path: str | Path, gender: str = None):
        print('\033[45mconfig.yaml 로드\033[0m')
        with open(config_path, encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

        self._gender = (gender or self._cfg.get("gender", "male")).lower()
        print(f'\033[45mgender: {self._gender}\033[0m')
        

        # 전처리기 (MediaPipe FaceLandmarker)
        print(f'\033[45mFacePreprocessor 초기화\033[0m')
        self._preprocessor = FacePreprocessor(
            face_landmarker_model=Path(self._cfg.get('root_dir', '.')) / self._cfg["face_landmarker_model"]
        )
        print(f'\033[45mFacePreprocessor 초기화 완료\033[0m')


        # EfficientNet TFLite 예측기
        print(f'\033[45mEfficientNet 예측기 초기화\033[0m')
        self._effnet = EffNetPredictor(
            config=self._cfg,
            gender=self._gender,
        )
        print(f'\033[45mEfficientNet 예측기 초기화 완료\033[0m')

        # 볼처짐·턱처짐 예측기
        print(f'\033[45msagging 예측기 초기화\033[0m')
        sagging_cfg = self._cfg["sagging"]
        self._sagging = SaggingPredictor(
            face_landmarker_model=Path(self._cfg.get('root_dir', '.')) / self._cfg["face_landmarker_model"],
            cheek_params_path=Path(self._cfg.get('root_dir', '.')) / sagging_cfg["cheek"][self._gender],
            chin_params_path=Path(self._cfg.get('root_dir', '.')) / sagging_cfg["chin"][self._gender],
        )
        print(f'\033[45msagging 예측기 초기화 완료\033[0m')

    # ── 단일 이미지 ───────────────────────────────────────────────────────────

    def predict_single(self, image_source, gender: str = None) -> dict | None:
        """
        Parameters
        ----------
        image_source : str | Path | np.ndarray
        gender       : str | None  – 이 호출에서만 성별 override.

        Returns
        -------
        dict 또는 None (얼굴 미검출)
        {
            "age":            float | None
            "pigment":        {"left": float, "right": float}
            "wrinkle":        {sector: float | None, ...}
            "homogenity":     {"radiance": float | None, "texture": float | None}
            "cheek_sagging":  {"right": float(0~50), "left": float(0~50), "total": float(0~100)}
            "chin_sagging":   {"right": float(0~50), "left": float(0~50), "total": float(0~100)}
            "valid_sagging":  bool
        }
        """
        # gender override 처리
        print('gender override 처리')
        if gender is not None and gender.lower() != self._gender:
            # 임시로 성별 바꿔서 별도 파이프라인 생성하는 것보다
            # 재사용 가능한 모듈별 gender 인자를 활용
            return SkinPipeline(
                config_path=self._config_path_for_override(),
                gender=gender,
            ).predict_single(image_source)

        # 전처리
        print('\033[45m전처리\033[0m')
        crops = self._preprocessor.process(image_source)
        if crops is None:
            return None   # 얼굴 미검출

        # EfficientNet 예측
        print('\033[45m--- EfficientNet ---\033[0m')
        print('\033[45mage 예측\033[0m')
        age     = self._effnet.predict_age(crops["age_crop"])
        
        print('\033[45mpigment 예측\033[0m')
        pigment = self._effnet.predict_pigment(crops["pigment_crops"])

        print('\033[45mwrinkle 예측\033[0m')
        wrinkle = self._effnet.predict_wrinkle(crops["wrinkle_crops"])

        print('\033[45mhomogenity 예측\033[0m')
        homogenity = self._effnet.predict_homogenity(crops["age_crop"])

        # 볼처짐 / 턱처짐
        print('\033[45m--- Sagging ---\033[0m')
        sagging_result = {"cheek": {}, "chin": {}}
        if crops["valid_sagging"] and crops["sagging_image"] is not None:
            sagging_result = self._sagging.predict(crops["sagging_image"])
        else:
            null = {"right": None, "left": None, "mean": None}
            sagging_result = {"cheek": null, "chin": null}

        return {
            "age":           age,
            "pigment":       pigment,
            "wrinkle":       wrinkle,
            "homogenity":    homogenity,
            "cheek_sagging": sagging_result["cheek"],
            "chin_sagging":  sagging_result["chin"],
            "valid_sagging": crops["valid_sagging"],
            "gender": self._gender
        }

    # ── 배치 처리 ─────────────────────────────────────────────────────────────

    def predict_batch(self,
                      image_dir: str | Path,
                      gender: str = None,
                      output_csv: str | Path = None) -> pd.DataFrame:
        """
        Parameters
        ----------
        image_dir  : 이미지 루트 디렉토리 (하위 폴더 포함 재귀 스캔)
        gender     : str | None
        output_csv : CSV 저장 경로. None이면 config output_dir에 자동 생성.

        Returns
        -------
        pd.DataFrame
        """
        img_paths = sorted(
            p for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        print(f"\033[45m총 이미지 수: {len(img_paths)}\033[0m")

        rows = []
        for path in tqdm(img_paths, desc="Pipeline inference"):
            try:
                result = self.predict_single(str(path), gender=gender)
                row = _flatten_result(result, filename=path.name) if result else {"filename": path.name}
                rows.append(row)
            except Exception as e:
                print(f"오류: {path} → {e}")
                rows.append({"filename": path.name})

        df = pd.DataFrame(rows)

        # CSV 저장
        if output_csv is None:
            out_dir = Path(self._cfg.get("output_dir", "."))
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%y%m%d_%H%M%S")
            output_csv = out_dir / f"pipeline_batch_{ts}.csv"

        df.to_csv(str(output_csv), index=False, encoding="utf-8-sig")
        print(f"\033[45m저장 완료: {output_csv} (rows={len(df)})\033[0m")
        return df

    # ── private ───────────────────────────────────────────────────────────────

    def _config_path_for_override(self):
        """gender override 시 사용할 config 경로 (현재 인스턴스 생성 시 경로 보존용)."""
        raise NotImplementedError(
            "gender 인자는 SkinPipeline 생성 시 지정하거나 "
            "predict_batch(..., gender='female') 처럼 배치 단위로 지정하세요."
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Skin Analysis Pipeline")
    parser.add_argument("--config",     default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--image",      default=None, help="단일 이미지 경로")
    parser.add_argument("--image_dir",  default=None, help="배치 이미지 디렉토리")
    parser.add_argument("--gender",     default=None, help="male | female (config 값 override)")
    parser.add_argument("--output_csv", default=None, help="배치 결과 CSV 저장 경로")
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("--image 또는 --image_dir 중 하나를 지정하세요.")

    print("\033[45mSkinPipeline 시작\033[0m")
    pipe = SkinPipeline(config_path=args.config, gender=args.gender)
    
    if args.image:
        result = pipe.predict_single(args.image)
        if result is None:
            print("얼굴을 검출하지 못했습니다.")
        else:
            import json
            print(json.dumps(_flatten_result(result, Path(args.image).name), ensure_ascii=False, indent=2))

    if args.image_dir:
        pipe.predict_batch(
            image_dir=args.image_dir,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()

finish = datetime.now()
print(f"총 실행 시간: {finish - start}")