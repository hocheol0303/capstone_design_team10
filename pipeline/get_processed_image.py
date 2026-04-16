"""
get_processed_image.py
======================
preprocess.py 중간/최종 산물을 지정 폴더에 저장하여 시각적으로 확인.

저장 파일 (per image)
---------------------
<output_dir>/<image_stem>/
    remove_non_skin.png   : 회전 + 비피부 마스킹이 적용된 full 이미지
    age_crop.png          : age 모델 입력
    pigment_left.png      : pigment 모델 입력 (왼쪽 볼)
    pigment_right.png     : pigment 모델 입력 (오른쪽 볼)
    wrinkle_<sector>.png  : wrinkle 7개 sector
    sagging_image.png     : sagging 모델 입력 (1024×1024)

사용:
    python get_processed_image.py --image /path/to/img.jpg     --output_dir ./check
    python get_processed_image.py --image_dir /path/to/imgs    --output_dir ./check
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from preprocess import (
    FacePreprocessor,
    WRINKLE_SECTORS,
    _apply_skin_mask,
    _bgr_to_mp_image,
    _compute_rotation,
    _get_coord,
    _make_age_crop,
    _make_pigment_crops,
    _make_sagging_image,
    _make_wrinkle_crops,
)


def _save_rgb(path: Path, rgb: np.ndarray) -> bool:
    if rgb is None or not hasattr(rgb, "size") or rgb.size == 0:
        print(f"  skip (None/empty): {path.name}")
        return False
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    print(f"  saved: {path.name}  ({rgb.shape[1]}x{rgb.shape[0]})")
    return True


def analyze_one(preprocessor: FacePreprocessor,
                bgr: np.ndarray,
                apply_masking: bool = True,
                apply_rotation: bool = True) -> dict | None:
    """FacePreprocessor.process 와 동일 로직 + remove_non_skin full 이미지 반환."""
    result = preprocessor._landmarker.detect(_bgr_to_mp_image(bgr))
    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    h, w = bgr.shape[:2]
    coords = np.array(
        [_get_coord(landmarks, i, w, h) for i in range(len(landmarks))],
        np.int32,
    )

    if apply_rotation:
        input_bgr, coords = _compute_rotation(bgr, coords, landmarks)
    else:
        input_bgr = bgr

    if apply_masking:
        input_bgr = _apply_skin_mask(input_bgr, coords)

    input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)

    age_crop      = _make_age_crop(input_rgb, coords)
    pigment_crops = _make_pigment_crops(input_rgb, coords)
    wrinkle_crops = _make_wrinkle_crops(input_rgb, coords)

    sagging_rgb, yaw_s, pitch_s, valid_sagging = _make_sagging_image(
        bgr, preprocessor._landmarker
    )

    return {
        "remove_non_skin": input_rgb,
        "age_crop":        age_crop,
        "pigment_crops":   pigment_crops,
        "wrinkle_crops":   wrinkle_crops,
        "sagging_image":   sagging_rgb,
        "yaw":             yaw_s,
        "pitch":           pitch_s,
        "valid_sagging":   valid_sagging,
    }


def save_for_image(preprocessor: FacePreprocessor,
                   image_path: Path,
                   output_dir: Path,
                   apply_masking: bool = True,
                   apply_rotation: bool = True) -> None:
    print(f"\n[ {image_path.name} ]")
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print("  FAIL: 이미지 로드 실패")
        return

    out = analyze_one(preprocessor, bgr, apply_masking, apply_rotation)
    if out is None:
        print("  FAIL: 얼굴 미검출")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    _save_rgb(output_dir / "remove_non_skin.png", out["remove_non_skin"])
    _save_rgb(output_dir / "age_crop.png",        out["age_crop"])
    _save_rgb(output_dir / "pigment_left.png",    out["pigment_crops"]["left"])
    _save_rgb(output_dir / "pigment_right.png",   out["pigment_crops"]["right"])
    for sector in WRINKLE_SECTORS:
        _save_rgb(output_dir / f"wrinkle_{sector}.png",
                  out["wrinkle_crops"].get(sector))
    _save_rgb(output_dir / "sagging_image.png",   out["sagging_image"])

    print(f"  yaw={out['yaw']:.3f}  pitch={out['pitch']:.3f}  "
          f"valid_sagging={out['valid_sagging']}")


def main():
    parser = argparse.ArgumentParser(
        description="preprocess.py 중간/최종 산물을 폴더에 저장")
    parser.add_argument("--config",
                        default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--image",      default=None, help="단일 이미지 경로")
    parser.add_argument("--image_dir",  default=None, help="배치 이미지 디렉토리")
    parser.add_argument("--output_dir", required=True,
                        help="crop 결과 저장 폴더 (이미지별 하위 폴더 생성)")
    parser.add_argument("--no_masking",  action="store_true",
                        help="비피부 마스킹 비활성화")
    parser.add_argument("--no_rotation", action="store_true",
                        help="얼굴 정렬 회전 비활성화")
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("--image 또는 --image_dir 중 하나를 지정하세요.")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = Path(cfg.get("root_dir", ".")) / cfg["face_landmarker_model"]
    preprocessor = FacePreprocessor(face_landmarker_model=str(model_path))

    apply_masking  = not args.no_masking
    apply_rotation = not args.no_rotation
    output_root = Path(args.output_dir)

    if args.image:
        p = Path(args.image)
        save_for_image(preprocessor, p, output_root / p.stem,
                       apply_masking, apply_rotation)

    if args.image_dir:
        paths = sorted(
            p for p in Path(args.image_dir).rglob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        print(f"총 {len(paths)}장 처리")
        for p in paths:
            save_for_image(preprocessor, p, output_root / p.stem,
                           apply_masking, apply_rotation)


if __name__ == "__main__":
    main()
