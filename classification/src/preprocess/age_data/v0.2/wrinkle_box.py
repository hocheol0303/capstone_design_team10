import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import math

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

def get_overlap_rect(p1, p2, p3, p4):
    x1_min, x1_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y1_min, y1_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    x2_min, x2_max = min(p3[0], p4[0]), max(p3[0], p4[0])
    y2_min, y2_max = min(p3[1], p4[1]), max(p3[1], p4[1])
    x_min, x_max = max(x1_min, x2_min), min(x1_max, x2_max)
    y_min, y_max = max(y1_min, y2_min), min(y1_max, y2_max)
    return (x_min, y_min, x_max, y_max) if x_min < x_max and y_min < y_max else None

# 원래 age process 결과물과 주름 박스 좌표(age process 결과물에 위치해야함)의 좌표를 반환
def combined_process(source_path, face_mesh=None):
    image = cv2.imread(source_path)

    h, w, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # age 영역 마스킹
    results = face_mesh.process(rgb_image)

    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([get_coord(landmarks, i, w, h) for i in range(len(landmarks))], np.int32)

    center_idx = 1
    cx, cy = get_coord(landmarks, center_idx, w, h)

    left_eye = get_coord(landmarks, 33, w, h)
    right_eye = get_coord(landmarks, 263, w, h)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    M = get_rotation_matrix((cx, cy), angle_deg)
    rotated_img = cv2.warpAffine(image, M, (w, h))

    rotated_coords = np.array([rotate_point(p, (cx, cy), M) for p in coords], np.int32)

    face_oval_pts = np.array([rotated_coords[i] for i in FACE_OVAL], np.int32)
    mask_face = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_face, [face_oval_pts], 255)

    # 얼굴 외부 마스킹
    masked_image = cv2.bitwise_and(rotated_img, rotated_img, mask=mask_face)

    # 피부 아닌 부분 마스킹
    for region in [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW]:
        pts = np.array([rotated_coords[i] for i in region], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(masked_image, [pts], (0, 0, 0))
    
    LIPS_COMBINED = UPPER_LIP_OUTER + LOWER_LIP_OUTER[::-1]
    lip_pts = np.array([rotated_coords[i] for i in LIPS_COMBINED], np.int32)
    cv2.fillPoly(masked_image, [lip_pts], (0, 0, 0))
    # age 영역 마스킹 끝

    # wrinkle box
    wrinkle_boxes = {}

    # forehead
    pt10, pt8 = rotated_coords[10], rotated_coords[8]
    d = math.hypot(pt10[0] - pt8[0], pt10[1] - pt8[1])
    ux, uy = (pt10[0] - pt8[0]) / d, (pt10[1] - pt8[1]) / d
    px, py = int(pt10[0] + (d / 2) * ux), int(pt10[1] + (d / 2) * uy)
    x_min, x_max = rotated_coords[104][0], rotated_coords[333][0]
    y_min, y_max = py, pt8[1]
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    wrinkle_boxes['forehead'] = {'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}
    
    # right eye
    p1, p2 = rotated_coords[189], rotated_coords[123]
    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    wrinkle_boxes['right_eye'] = {'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}
    
    # left eye
    p3, p4 = rotated_coords[413], rotated_coords[352]
    y_min, y_max = min(p3[1], p4[1]), max(p3[1], p4[1])
    x_min, x_max = min(p3[0], p4[0]), max(p3[0], p4[0])
    wrinkle_boxes['left_eye'] = {'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}

    # nasolabial
    box = get_overlap_rect(rotated_coords[118], rotated_coords[434], rotated_coords[347], rotated_coords[214])
    
    if box:
        x_min, y_min, x_max, y_max = box
    wrinkle_boxes['nasolabial'] = {'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}

    # oral
    box = get_overlap_rect(rotated_coords[207], rotated_coords[379], rotated_coords[427], rotated_coords[150])

    if box:
        x_min, y_min, x_max, y_max = box
    wrinkle_boxes['oral'] = {'y_min': y_min, 'y_max': y_max, 'x_min': x_min, 'x_max': x_max}

    # right cheek
    # left cheek


    # 잘라낸 이미지랑 그에 해당하는 좌표들 반환
    x, y, w_crop, h_crop = cv2.boundingRect(face_oval_pts)
    cropped_image = masked_image[y:y + h_crop, x:x + w_crop]

    adjusted_boxes = {}
    for sector, box in wrinkle_boxes.items():
        adjusted_boxes[sector] = {
            'y_min': wrinkle_boxes[sector]['y_min'] - y,
            'y_max': wrinkle_boxes[sector]['y_max'] - y,
            'x_min': wrinkle_boxes[sector]['x_min'] - x,
            'x_max': wrinkle_boxes[sector]['x_max'] - x
        }


    return masked_image, wrinkle_boxes, cropped_image, adjusted_boxes