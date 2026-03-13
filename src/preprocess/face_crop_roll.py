import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import math

# 좌표 변환 함수
def get_coord(landmarks, idx, w, h):
    return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

def get_rotation_matrix(center, angle_deg, scale=1.0):
    return cv2.getRotationMatrix2D(center, angle_deg, scale)

def rotate_point(p, center, matrix):
    px, py = p
    cx, cy = center
    rotated = matrix @ np.array([px, py, 1])
    return int(rotated[0]), int(rotated[1])

# 마스킹 영역 인덱스
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]

def age_preprocess(landmarks, mp_image):
    image = mp_image.numpy_view()
    # cvtColor 빼는게 정답에 더 가까움 -> BRG로 되어있지 않다?
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    coords = np.array([get_coord(landmarks, i, w, h) for i in range(len(landmarks))], np.int32)

    # 중심 좌표 및 roll 값 (얼굴 기준 중심점 사용)
    center_idx = 1  # nose tip
    cx, cy = get_coord(landmarks, center_idx, w, h)
    roll_rad = landmarks[center_idx].z  # roll 추정이 어려워 직접 계산할 수도 있음
    left_eye = get_coord(landmarks, 33, w, h)
    right_eye = get_coord(landmarks, 263, w, h)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    # 회전 행렬 생성 및 이미지 회전
    M = get_rotation_matrix((cx, cy), angle_deg)
    rotated_img = cv2.warpAffine(image, M, (w, h))

    # 좌표 회전
    rotated_coords = np.array([rotate_point(p, (cx, cy), M) for p in coords], np.int32)

    # 마스킹
    face_oval_pts = np.array([rotated_coords[i] for i in FACE_OVAL], np.int32)
    mask_face = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_face, [face_oval_pts], 255)

    masked_image = cv2.bitwise_and(rotated_img, rotated_img, mask=mask_face)

    for region in [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW]:
        pts = np.array([rotated_coords[i] for i in region], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(masked_image, [pts], (0, 0, 0))

    LIPS_COMBINED = UPPER_LIP_OUTER + LOWER_LIP_OUTER[::-1]
    lip_pts = np.array([rotated_coords[i] for i in LIPS_COMBINED], np.int32)
    cv2.fillPoly(masked_image, [lip_pts], (0, 0, 0))

    x, y, w_crop, h_crop = cv2.boundingRect(face_oval_pts)
    if x < 0: x = 0
    if y < 0: y = 0
    cropped = masked_image[y:y + h_crop, x:x + w_crop]

    return cropped