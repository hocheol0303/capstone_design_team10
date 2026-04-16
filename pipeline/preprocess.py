"""
preprocess.py
=============
원본 이미지 → 각 모델별 입력 crop 생성 (MediaPipe FaceLandmarker 단일 실행 기반).

반환 구조 (FacePreprocessor.process)
--------------------------------------
{
    "age_crop":       np.ndarray (H, W, 3) RGB   – 전체 얼굴 마스킹 crop
    "pigment_crops":  {"left": ndarray, "right": ndarray}  – 볼 crop
    "wrinkle_crops":  {sector: ndarray | None}   – 7개 sector crop
    "sagging_image":  np.ndarray (1024, 1024, 3) RGB  – 볼/턱처짐용 resize_noratio
    "yaw":   float,  "pitch": float,  "roll": float   – 원본 오일러 각
    "valid_sagging":  bool   – |yaw|<0.1 and |pitch|<0.1
}
None 반환 → 얼굴 미검출

색상 규약: 내부는 모두 RGB numpy array (TF decode_jpeg 기준).
OpenCV imread(BGR) 사용 시 cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 필요.
"""

import math
import uuid
import shutil
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.core.base_options import BaseOptions


# ── 얼굴 마스킹 인덱스 (age / wrinkle 공용) ──────────────────────────────────
_LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
_UPPER_LIP  = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
_LOWER_LIP  = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
_FACE_OVAL  = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

WRINKLE_SECTORS = ["forehead", "right_eye", "left_eye", "nasolabial", "perioral", "right_vol", "left_vol"]

# sagging 전처리 상수
_SAGGING_TARGET_SIZE = 1024
_SAGGING_ANGLE_THRESH = 0.1   # |yaw|, |pitch| 필터 (rad)

# ── 디버그 토글 ───────────────────────────────────────────────────────────────
# 주석 처리로 True/False 전환
_DEBUG_APPLY_MASKING  = True    # 비피부 마스킹 적용
# _DEBUG_APPLY_MASKING  = False  # 비피부 마스킹 미적용

_DEBUG_APPLY_ROTATION = True    # 얼굴 정렬 회전 적용
# _DEBUG_APPLY_ROTATION = False  # 얼굴 정렬 회전 미적용


# ── 기하 유틸 ─────────────────────────────────────────────────────────────────

def _rot2euler(R: np.ndarray):
    """3×3 회전 행렬 → (yaw=r_x, roll=r_y, pitch=r_z)."""
    R[2, 0] = np.clip(R[2, 0], -1.0, 1.0)
    r_x = np.arcsin(-R[2, 0])
    if abs(R[2, 0]) < 1e-6:
        r_y = 0.0
        r_z = np.arctan2(R[0, 1], R[0, 2])
    else:
        r_y = np.arctan2(R[1, 0], R[0, 0])
        r_z = np.arctan2(R[2, 1], R[2, 2])
    return float(r_x), float(r_y), float(r_z)   # yaw, roll, pitch


def _get_coord(landmarks, idx, w, h):
    return int(landmarks[idx].x * w), int(landmarks[idx].y * h)


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def _rotate_points(coords: np.ndarray, center, angle_deg: float) -> np.ndarray:
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    ones = np.ones((len(coords), 1))
    pts = np.hstack([coords, ones])
    return (M @ pts.T).T.astype(np.int32)


def _get_overlap_rect(p1, p2, p3, p4):
    x1_min, x1_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y1_min, y1_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    x2_min, x2_max = min(p3[0], p4[0]), max(p3[0], p4[0])
    y2_min, y2_max = min(p3[1], p4[1]), max(p3[1], p4[1])
    x_min, x_max = max(x1_min, x2_min), min(x1_max, x2_max)
    y_min, y_max = max(y1_min, y2_min), min(y1_max, y2_max)
    return (x_min, y_min, x_max, y_max) if x_min < x_max and y_min < y_max else None


def _crop_square_with_padding(img: np.ndarray, x0: float, y0: float, side: float) -> np.ndarray:
    """(x0, y0) 기준 side×side 크롭. 이미지 경계 밖은 검정 패딩."""
    H, W = img.shape[:2]
    side = int(round(side))
    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    canvas = np.zeros((side, side, 3), dtype=img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + side), min(H, y0 + side)
    if sx1 <= sx0 or sy1 <= sy0:
        return canvas
    dx0, dy0 = sx0 - x0, sy0 - y0
    canvas[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return canvas


# ── MediaPipe 초기화 ───────────────────────────────────────────────────────────

def _init_landmarker(model_path: str) -> face_landmarker.FaceLandmarker:
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return face_landmarker.FaceLandmarker.create_from_options(options)


def _bgr_to_mp_image(bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _compute_rotation(bgr, coords, landmarks) -> tuple:
    """
    얼굴 랜드마크 기준으로 이미지와 coords를 동일한 중심(cx, cy)으로 회전.

    Returns
    -------
    (rotated_rgb, rotated_coords)
    """
    h, w = bgr.shape[:2]
    left_eye  = _get_coord(landmarks, 33,  w, h)
    right_eye = _get_coord(landmarks, 263, w, h)
    cx, cy = _get_coord(landmarks, 1, w, h)

    angle_deg = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    rotated_coords = _rotate_points(coords, (cx, cy), angle_deg)

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated_rgb = cv2.warpAffine(bgr, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0))

    return rotated_rgb, rotated_coords


def _apply_skin_mask(bgr: np.ndarray, coords) -> np.ndarray:
    """
    비피부 마스킹: face oval 외부 제거 + 눈/눈썹/입술 제거.
    bgr와 coords는 동일한 좌표계여야 함 (둘 다 회전됐거나 둘 다 원본).
    """
    h, w = bgr.shape[:2]

    face_oval_pts = np.array([coords[i] for i in _FACE_OVAL], np.int32)
    mask_face = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_face, [face_oval_pts], 255)
    masked = cv2.bitwise_and(bgr, bgr, mask=mask_face)

    for region in [_LEFT_EYE, _RIGHT_EYE, _LEFT_BROW, _RIGHT_BROW]:
        pts = np.array([coords[i] for i in region], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(masked, [pts], (0, 0, 0))

    lip_pts = np.array([coords[i] for i in (_UPPER_LIP + _LOWER_LIP[::-1])], np.int32)
    cv2.fillPoly(masked, [lip_pts], (0, 0, 0))

    return masked


# ── Age crop ──────────────────────────────────────────────────────────────────

def _make_age_crop(rgb: np.ndarray, coords) -> np.ndarray:
    """
    face_oval bounding-box 크롭.
    입력 rgb는 이미 회전·마스킹이 적용된 이미지여야 함.
    """
    face_pts = np.array([coords[i] for i in _FACE_OVAL], np.int32)
    x, y, cw, ch = cv2.boundingRect(face_pts)
    x, y = max(0, x), max(0, y)
    return rgb[y:y + ch, x:x + cw]


# ── Pigment crop ───────────────────────────────────────────────────────────────

def _make_pigment_crops(rgb: np.ndarray, coords):
    """
    landmarks 0, 34 (right cheek), 0, 264 (left cheek) 기준 크롭.
    입력 rgb는 이미 회전·마스킹이 적용된 이미지여야 함.
    반환: {"left": ndarray, "right": ndarray}
    """
    p0   = coords[0]
    p34  = coords[34]
    p264 = coords[264]

    def _rect_crop(pa, pb):
        x1, y1 = min(pa[0], pb[0]), min(pa[1], pb[1])
        x2, y2 = max(pa[0], pb[0]), max(pa[1], pb[1])
        return rgb[y1:y2, x1:x2]

    return {
        "left":  _rect_crop(p0, p264),
        "right": _rect_crop(p0, p34),
    }


# ── Wrinkle sector crops ───────────────────────────────────────────────────────

def _make_wrinkle_crops(rgb: np.ndarray, coords) -> dict:
    """
    7개 wrinkle sector crop 생성.
    입력 rgb는 이미 회전·마스킹이 적용된 이미지여야 함.
    None이 포함될 수 있음 (nasolabial, perioral box 미검출 시).
    반환 key: WRINKLE_SECTORS 순서와 동일
    """
    r = coords
    masked = rgb

    # forehead
    pt10, pt8 = coords[10], coords[8]
    d = math.hypot(pt10[0] - pt8[0], pt10[1] - pt8[1]) + 1e-12
    ux, uy = (pt10[0] - pt8[0]) / d, (pt10[1] - pt8[1]) / d
    px_ = int(pt10[0] + (d / 2) * ux)
    py_ = int(pt10[1] + (d / 2) * uy)
    xf_min, xf_max = coords[104][0], coords[333][0]
    yf_min, yf_max = py_, pt8[1]
    if yf_min > yf_max:
        yf_min, yf_max = yf_max, yf_min
    forehead = rgb[yf_min:yf_max, xf_min:xf_max]

    # right_eye
    p1, p2 = coords[189], coords[123]
    rt_eye = rgb[min(p1[1], p2[1]):max(p1[1], p2[1]),
                    min(p1[0], p2[0]):max(p1[0], p2[0])]

    # left_eye
    p3, p4 = coords[413], coords[352]
    lf_eye = rgb[min(p3[1], p4[1]):max(p3[1], p4[1]),
                    min(p3[0], p4[0]):max(p3[0], p4[0])]

    # nasolabial
    box = _get_overlap_rect(coords[118], coords[434], coords[347], coords[214])
    nasolabial = rgb[box[1]:box[3], box[0]:box[2]] if box else None

    # perioral
    box = _get_overlap_rect(coords[207], coords[379], coords[427], coords[150])
    perioral = rgb[box[1]:box[3], box[0]:box[2]] if box else None

    # right_vol / left_vol (cheek volume)
    p5, p6 = coords[234], coords[212]
    rt_vol = rgb[min(p5[1], p6[1]):max(p5[1], p6[1]),
                    min(p5[0], p6[0]):max(p5[0], p6[0])]

    p7, p8 = coords[454], coords[432]
    lt_vol = rgb[min(p7[1], p8[1]):max(p7[1], p8[1]),
                    min(p7[0], p8[0]):max(p7[0], p8[0])]

    return {
        "forehead":   forehead,
        "right_eye":  rt_eye,
        "left_eye":   lf_eye,
        "nasolabial": nasolabial,
        "perioral":   perioral,
        "right_vol":  rt_vol,
        "left_vol":   lt_vol,
    }


# ── Sagging resize_noratio image ──────────────────────────────────────────────

def _make_sagging_image(
    bgr: np.ndarray,
    landmarker: face_landmarker.FaceLandmarker,
) -> tuple:
    """
    볼/턱 처짐 전처리: 168-152 수직 정렬 → 크롭 → 1024×1024.

    Returns
    -------
    (sagging_rgb, yaw, pitch, valid)
    valid=False → |yaw| 또는 |pitch| >= 0.1 (처짐 계산 불가)
    """
    # 1차 랜드마크: 오일러각 획득
    result1 = landmarker.detect(_bgr_to_mp_image(bgr))
    if not result1.face_landmarks:
        return None, 0.0, 0.0, False

    R = np.asarray(result1.facial_transformation_matrixes[0][:3, :3])
    yaw, _, pitch = _rot2euler(R)

    valid = (abs(yaw) < _SAGGING_ANGLE_THRESH) and (abs(pitch) < _SAGGING_ANGLE_THRESH)

    h0, w0 = bgr.shape[:2]
    lmk0 = result1.face_landmarks[0]
    xA = lmk0[168].x * w0;  yA = lmk0[168].y * h0
    xB = lmk0[152].x * w0;  yB = lmk0[152].y * h0

    # 168-152를 수직 정렬하는 회전각
    dx = xB - xA
    dy = yB - yA
    angle_deg = math.degrees(math.atan2(dx, dy))
    bgr_v = _rotate_image(bgr, -angle_deg)

    # 2차 랜드마크: 회전된 이미지에서
    result2 = landmarker.detect(_bgr_to_mp_image(bgr_v))
    if not result2.face_landmarks:
        return None, yaw, pitch, False

    lmks = result2.face_landmarks[0]
    h, w = bgr_v.shape[:2]
    xA2 = lmks[168].x * w;  yA2 = lmks[168].y * h
    xB2 = lmks[152].x * w;  yB2 = lmks[152].y * h

    # 위/아래 판정
    if yA2 <= yB2:
        x_up, y_up, x_dn, y_dn = xA2, yA2, xB2, yB2
    else:
        x_up, y_up, x_dn, y_dn = xB2, yB2, xA2, yA2

    # 위 0.7L / 아래 0.3L 연장
    vx, vy = x_dn - x_up, y_dn - y_up
    L = math.hypot(vx, vy) + 1e-12
    ux, uy = vx / L, vy / L
    x_up_ext = x_up - 0.7 * L * ux;  y_up_ext = y_up - 0.7 * L * uy
    x_dn_ext = x_dn + 0.3 * L * ux;  y_dn_ext = y_dn + 0.3 * L * uy

    ext_len = math.hypot(x_dn_ext - x_up_ext, y_dn_ext - y_up_ext)
    side = max(2.0, ext_len)
    cx_s = 0.5 * (x_up_ext + x_dn_ext)
    cy_s = 0.5 * (y_up_ext + y_dn_ext)

    rgb_v = cv2.cvtColor(bgr_v, cv2.COLOR_BGR2RGB)
    crop = _crop_square_with_padding(rgb_v, cx_s - side / 2, cy_s - side / 2, side)
    sagging_rgb = cv2.resize(crop, (_SAGGING_TARGET_SIZE, _SAGGING_TARGET_SIZE),
                             interpolation=cv2.INTER_AREA)

    return sagging_rgb, yaw, pitch, valid


# ── 메인 클래스 ───────────────────────────────────────────────────────────────

class FacePreprocessor:
    """
    원본 이미지 경로 또는 BGR numpy array를 받아
    age / pigment / wrinkle / sagging 각각의 입력 crop을 반환.
    """

    def __init__(self, face_landmarker_model: str):
        self._landmarker = _init_landmarker(face_landmarker_model)

    def process(self, image_source,
                apply_masking: bool = _DEBUG_APPLY_MASKING,
                apply_rotation: bool = _DEBUG_APPLY_ROTATION) -> dict | None:
        """
        Parameters
        ----------
        image_source : str | Path | np.ndarray
            이미지 파일 경로 또는 BGR numpy array.
        apply_masking : bool
            비피부 마스킹 적용 여부. 기본값은 파일 상단 _DEBUG_APPLY_MASKING.
        apply_rotation : bool
            얼굴 정렬 회전 적용 여부. 기본값은 파일 상단 _DEBUG_APPLY_ROTATION.

        Returns
        -------
        dict  – 각 모델 입력 crop (RGB numpy array)
        None  – 얼굴 미검출
        """
        # 이미지 로드
        if isinstance(image_source, (str, Path)):
            bgr = cv2.imread(str(image_source))
            if bgr is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없음: {image_source}")
        else:
            bgr = image_source  # 이미 BGR numpy

        # MediaPipe 검출 (원본 이미지에서 1회)
        mp_img = _bgr_to_mp_image(bgr)
        result = self._landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None

        R = np.asarray(result.facial_transformation_matrixes[0][:3, :3])
        yaw, roll, pitch = _rot2euler(R)
        landmarks = result.face_landmarks[0]
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        h, w = bgr.shape[:2]
        
        coords = np.array([_get_coord(landmarks, i, w, h) for i in range(len(landmarks))], np.int32)

        if apply_rotation:
            input_bgr, coords = _compute_rotation(bgr, coords, landmarks)
        else:
            input_bgr = bgr

        if apply_masking:
            input_bgr = _apply_skin_mask(input_bgr, coords)
        else:
            input_bgr = input_bgr
        

        input_rgb = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB)


        # 각 crop 생성 (이미 회전·마스킹된 input_rgb와 coords 사용)
        age_crop      = _make_age_crop(input_rgb, coords)
        pigment_crops = _make_pigment_crops(input_rgb, coords)
        wrinkle_crops = _make_wrinkle_crops(input_rgb, coords)

        # sagging 전용 preprocessing (168-152 정렬 + 2차 랜드마크)
        sagging_rgb, yaw_s, pitch_s, valid_sagging = _make_sagging_image(input_bgr, self._landmarker)

        return {
            "age_crop":      age_crop,
            "pigment_crops": pigment_crops,
            "wrinkle_crops": wrinkle_crops,
            "sagging_image": sagging_rgb,
            "yaw":           yaw_s,    # sagging용 오일러각 사용 (2차 검출 전)
            "pitch":         pitch_s,
            "roll":          roll,
            "valid_sagging": valid_sagging,
        }
