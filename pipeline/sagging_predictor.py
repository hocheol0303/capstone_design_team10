"""
sagging_predictor.py
====================
MediaPipe 랜드마크 기울기 → Δ 보정 → 0~1 점수화.

볼처짐 (cheek): INDEX_PAIRS 중 right [16,18,20,22,30], left [29,27,25,23,31]
턱처짐 (chin):  INDEX_PAIRS 중 right [2,3,4,5,6],    left [13,12,11,10,9]

처리 흐름
---------
1. sagging_image (resize_noratio, 1024×1024) 에서 MediaPipe 재실행
2. 회전 행렬 → rot2euler → roll 보정
3. 기울기 slopes_rot 계산
4. right / left 방향 각각 raw index 값 추출
5. JSON params (Δ_pitch + Δ_yaw + Δ_ratio) 로 보정
6. robust min-max 점수화 → 0~1
7. right / left 점수 평균 → 최종 스칼라 반환
"""

import math
import json
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.core.base_options import BaseOptions


# ── 공통 INDEX_PAIRS (landmark_utils.py 동일) ─────────────────────────────────
INDEX_PAIRS = [
    [132, 58], [58, 172], [172, 136], [136, 150], [150, 149], [149, 176],
    [176, 148], [148, 152], [152, 377], [377, 400], [400, 378], [378, 379],
    [379, 365], [365, 397], [397, 288], [288, 361],                          # 0~15
    [123, 187], [123, 207], [50, 205], [50, 206],                            # 16~19
    [101, 36],  [101, 203], [100, 142],                                      # 20~22
    [371, 329], [423, 330], [266, 330],                                      # 23~25
    [426, 280], [425, 280], [427, 352], [411, 352],                          # 26~29
    [137, 147], [376, 366],                                                  # 30~31
]

# 볼처짐 / 턱처짐 인덱스 정의
CHEEK_RIGHT_IDX  = [16, 18, 20, 22, 30]
CHEEK_STD2LEFT   = {16: 29, 18: 27, 20: 25, 22: 23, 30: 31}

CHIN_RIGHT_IDX   = [2, 3, 4, 5, 6]
CHIN_STD2LEFT    = {2: 13, 3: 12, 4: 11, 5: 10, 6: 9}


# ── 기하 유틸 ─────────────────────────────────────────────────────────────────

def _rot2euler(R: np.ndarray):
    R[2, 0] = np.clip(R[2, 0], -1.0, 1.0)
    r_x = np.arcsin(-R[2, 0])
    if abs(R[2, 0]) < 1e-6:
        r_y = 0.0
        r_z = np.arctan2(R[0, 1], R[0, 2])
    else:
        r_y = np.arctan2(R[1, 0], R[0, 0])
        r_z = np.arctan2(R[2, 1], R[2, 2])
    return float(r_x), float(r_y), float(r_z)   # yaw, roll, pitch


def _rotate_landmarks(result, r_y: float, w: int, h: int):
    """roll(r_y) 기준 랜드마크 회전 정규화."""
    transformed = []
    for lm in result.face_landmarks[0]:
        x = w * lm.x * np.cos(r_y) - h * lm.y * np.sin(r_y)
        y = w * lm.x * np.sin(r_y) + h * lm.y * np.cos(r_y)
        transformed.append(landmark_pb2.NormalizedLandmark(x=x / w, y=y / h, z=lm.z))
    return transformed


def _calculate_slopes(landmarks, w: int, h: int) -> list:
    """INDEX_PAIRS 순서를 유지하며 slope 계산. 검출 실패 pair는 nan으로 채움."""
    L = len(landmarks)
    slopes = []
    for a, b in INDEX_PAIRS:
        if a < L and b < L:
            ax, ay = w * landmarks[a].x, h * landmarks[a].y
            bx, by = w * landmarks[b].x, h * landmarks[b].y
            slopes.append(math.atan2(by - ay, bx - ax))
        else:
            slopes.append(float("nan"))
    return slopes


def _pixel_distance(landmarks, idx_a: int, idx_b: int, w: int, h: int) -> float:
    if idx_a >= len(landmarks) or idx_b >= len(landmarks):
        return np.nan
    ax, ay = w * landmarks[idx_a].x, h * landmarks[idx_a].y
    bx, by = w * landmarks[idx_b].x, h * landmarks[idx_b].y
    return float(math.hypot(bx - ax, by - ay))


# ── piecewise 보정 유틸 ───────────────────────────────────────────────────────

def _piecewise_eval(bin_edges: np.ndarray, a_list, b_list, x: float) -> float:
    """스칼라 x에 대한 구간별 선형 보정값 계산 (클램프 적용)."""
    edges = np.asarray(bin_edges, dtype=float)
    xc = float(np.clip(x, edges[0], edges[-1]))
    a  = np.asarray(a_list, dtype=float)
    b  = np.asarray(b_list, dtype=float)
    n  = len(a)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if (xc >= lo and xc < hi) or (i == n - 1 and xc <= hi):
            return float(a[i] * xc + b[i])
    return float(a[-1] * xc + b[-1])


def _delta_sum(idx_params: dict,
               pitch_yaw_edges: np.ndarray, ratio_edges: np.ndarray,
               pitch: float, yaw: float, ratio: float,
               ratio_anchor: float) -> float:
    """Δ_pitch + Δ_yaw + Δ_ratio 합산."""
    total = 0.0

    if "pitch" in idx_params:
        g_x = _piecewise_eval(pitch_yaw_edges, idx_params["pitch"]["a"], idx_params["pitch"]["b"], pitch)
        g_0 = _piecewise_eval(pitch_yaw_edges, idx_params["pitch"]["a"], idx_params["pitch"]["b"], 0.0)
        total += g_x - g_0

    if "yaw" in idx_params:
        g_x = _piecewise_eval(pitch_yaw_edges, idx_params["yaw"]["a"], idx_params["yaw"]["b"], yaw)
        g_0 = _piecewise_eval(pitch_yaw_edges, idx_params["yaw"]["a"], idx_params["yaw"]["b"], 0.0)
        total += g_x - g_0

    if "ratio" in idx_params:
        g_x = _piecewise_eval(ratio_edges, idx_params["ratio"]["a"], idx_params["ratio"]["b"], ratio)
        g_a = _piecewise_eval(ratio_edges, idx_params["ratio"]["a"], idx_params["ratio"]["b"], ratio_anchor)
        total += g_x - g_a

    return total


def _score_from_thresholds(new_val: float, thresholds: dict, inverted: bool) -> float:
    """robust min-max 점수화 → 0~10. inverted=True 이면 작을수록 높은 점수 (볼처짐)."""
    vmin = float(thresholds["robust_min"])
    vmax = float(thresholds["robust_max"])
    denom = max(1e-12, vmax - vmin)
    norm = float(np.clip((new_val - vmin) / denom, 0.0, 1.0))
    return 10.0 * ((1.0 - norm) if inverted else norm)


# ── MediaPipe 초기화 ───────────────────────────────────────────────────────────

def _init_landmarker(model_path: str) -> face_landmarker.FaceLandmarker:
    options = face_landmarker.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return face_landmarker.FaceLandmarker.create_from_options(options)


# ── 메인 클래스 ───────────────────────────────────────────────────────────────

class SaggingPredictor:
    """
    볼처짐·턱처짐 점수 계산기.

    Parameters
    ----------
    face_landmarker_model : str
    cheek_params_path     : str  – 볼처짐 JSON (성별별)
    chin_params_path      : str  – 턱처짐 JSON (성별별)
    """

    def __init__(self,
                 face_landmarker_model: str,
                 cheek_params_path: str,
                 chin_params_path: str):
        self._landmarker = _init_landmarker(face_landmarker_model)
        self._cheek_params = self._load_params(cheek_params_path)
        self._chin_params  = self._load_params(chin_params_path)

    @staticmethod
    def _load_params(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _get_slopes_and_meta(self, sagging_rgb: np.ndarray):
        """
        sagging_image(RGB) → slopes_rot, yaw, pitch, ratio.
        Returns None if no face detected.
        """
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=sagging_rgb)
        result = self._landmarker.detect(mp_img)
        if not result.face_landmarks:
            return None

        R = np.asarray(result.facial_transformation_matrixes[0][:3, :3])
        yaw, roll, pitch = _rot2euler(R)

        h, w = sagging_rgb.shape[:2]
        t_lm = _rotate_landmarks(result, roll, w, h)

        d_168_152 = _pixel_distance(t_lm, 168, 152, w, h)
        d_446_226 = _pixel_distance(t_lm, 446, 226, w, h)
        ratio = (d_446_226 / d_168_152
                 if not np.isnan(d_168_152) and d_168_152 != 0.0
                 else np.nan)

        slopes_rot = _calculate_slopes(t_lm, w, h)
        return slopes_rot, yaw, pitch, ratio

    def _score_single_task(self,
                           slopes_rot: list,
                           yaw: float, pitch: float, ratio: float,
                           params: dict,
                           right_indices: list,
                           std_to_left: dict,
                           inverted: bool) -> dict:
        """
        한 태스크(cheek or chin)에 대해 right/left 방향 각각 점수화 후 반환.
        각 index 점수: 0~10, 방향별 합산: 0~50
        Returns {"right": float(0~50), "left": float(0~50)}  or None per direction
        """
        p_edges = np.array(params["pitch_yaw_bin_edges"], dtype=float)
        r_edges = np.array(params["ratio_bin_edges"],     dtype=float)
        r_anch  = float(params.get("ratio_anchor", 0.88))
        delta_p = params["delta_params"]
        score_t = params["scoring_thresholds"]

        ratio_clip = float(np.clip(ratio, r_edges[0], r_edges[-1])) if not np.isnan(ratio) else r_anch

        def _single_score(raw: float, idx_name: str, yaw_val: float) -> float | None:
            if np.isnan(raw) or idx_name not in delta_p:
                return None
            idx_params = delta_p[idx_name]
            d = _delta_sum(idx_params, p_edges, r_edges, pitch, yaw_val, ratio_clip, r_anch)
            new_val = raw - d
            if idx_name not in score_t:
                return None
            return _score_from_thresholds(new_val, score_t[idx_name], inverted)

        right_scores, left_scores = [], []
        for i in right_indices:
            idx_name = f"index{i}"
            raw_r = float(slopes_rot[i]) if i < len(slopes_rot) else np.nan
            s_r = _single_score(raw_r, idx_name, yaw)
            if s_r is not None:
                right_scores.append(s_r)

            src = std_to_left.get(i)
            raw_l = (-float(slopes_rot[src])
                     if src is not None and src < len(slopes_rot)
                     else np.nan)
            s_l = _single_score(raw_l, idx_name, -yaw)
            if s_l is not None:
                left_scores.append(s_l)

        return {
            "right": float(np.sum(right_scores)) if right_scores else None,
            "left":  float(np.sum(left_scores))  if left_scores  else None,
        }

    def predict(self, sagging_image: np.ndarray) -> dict:
        """
        Parameters
        ----------
        sagging_image : (1024, 1024, 3) RGB uint8 (resize_noratio 이미지)

        Returns
        -------
        {
            "cheek": {"right": float, "left": float, "mean": float},
            "chin":  {"right": float, "left": float, "mean": float},
        }
        None 필드 → 해당 방향 검출 실패
        """
        meta = self._get_slopes_and_meta(sagging_image)
        if meta is None:
            return {
                "cheek": {"right": None, "left": None, "mean": None},
                "chin":  {"right": None, "left": None, "mean": None},
            }

        slopes_rot, yaw, pitch, ratio = meta

        cheek = self._score_single_task(
            slopes_rot, yaw, pitch, ratio,
            self._cheek_params, CHEEK_RIGHT_IDX, CHEEK_STD2LEFT,
            inverted=True,
        )
        chin = self._score_single_task(
            slopes_rot, yaw, pitch, ratio,
            self._chin_params, CHIN_RIGHT_IDX, CHIN_STD2LEFT,
            inverted=False,
        )

        def _total(d):
            vals = [v for v in [d["right"], d["left"]] if v is not None]
            return float(np.sum(vals)) if vals else None

        cheek["total"] = _total(cheek)   # 0~100
        chin["total"]  = _total(chin)    # 0~100

        return {"cheek": cheek, "chin": chin}
