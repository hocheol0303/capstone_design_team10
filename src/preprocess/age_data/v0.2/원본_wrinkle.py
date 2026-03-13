import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import math
import os

# 현재 실행 파일이 위치한 폴더
# input_folder = Path(__file__).parent
valid_exts = [".jpg", ".jpeg", ".png"]

# # 하위 저장 폴더 설정
# region_names = ['forehead', 'rt_eye', 'lf_eye', 'nasolabial', 'oral', 'rt_cheek', 'lt_cheek']
# region_folders = {name: input_folder / name for name in region_names}
# for folder in region_folders.values():
#     folder.mkdir(exist_ok=True)

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# 마스킹 인덱스
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

def get_coord(landmarks, idx, w, h):
    return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

def get_rotation_matrix(center, angle_deg, scale=1.0):
    return cv2.getRotationMatrix2D(center, angle_deg, scale)

def rotate_point(p, center, matrix):
    px, py = p
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

# 메인 루프
def wrinkle_preprocess(file_path: str):
# for file in input_folder.iterdir():
    if os.path.splitext(file_path)[-1].lower() not in valid_exts:
        print(f"Skipping unsupported file: {file_path}")
        return

    image = cv2.imread(file_path)
    if image is None:
        print(f"Failed to read image: {file_path}")
        return

    h, w = image.shape[:2]
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        print(f"No face landmarks found in image: {file_path}")
        return

    landmarks = results.multi_face_landmarks[0].landmark
    coords = np.array([get_coord(landmarks, i, w, h) for i in range(len(landmarks))])

    # 회전 보정
    left_eye = get_coord(landmarks, 33, w, h)
    right_eye = get_coord(landmarks, 263, w, h)
    cx, cy = get_coord(landmarks, 1, w, h)
    angle_deg = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    M = get_rotation_matrix((cx, cy), angle_deg)
    rotated_img = cv2.warpAffine(image, M, (w, h))
    rotated_coords = np.array([rotate_point(p, (cx, cy), M) for p in coords])

    # 눈/눈썹/입술 마스킹
    mask_img = rotated_img.copy()
    for region in [LEFT_EYE, RIGHT_EYE, LEFT_BROW, RIGHT_BROW]:
        pts = np.array([rotated_coords[i] for i in region], np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask_img, [pts], (0, 0, 0))
    lips = UPPER_LIP_OUTER + LOWER_LIP_OUTER[::-1]
    lip_pts = np.array([rotated_coords[i] for i in lips], np.int32)
    cv2.fillPoly(mask_img, [lip_pts], (0, 0, 0))

    # --- forehead ---
    pt10, pt8 = rotated_coords[10], rotated_coords[8]
    d = math.hypot(pt10[0] - pt8[0], pt10[1] - pt8[1])
    ux, uy = (pt10[0] - pt8[0]) / d, (pt10[1] - pt8[1]) / d
    px, py = int(pt10[0] + (d / 2) * ux), int(pt10[1] + (d / 2) * uy)
    x_min, x_max = rotated_coords[104][0], rotated_coords[333][0]
    y_min, y_max = py, pt8[1]
    if y_min > y_max: y_min, y_max = y_max, y_min
    forehead_crop = mask_img[y_min:y_max, x_min:x_max]
    # print(str(region_folders['forehead'] / f"{file.stem}_forehead{file.suffix}"),"forehead_crop")

    # --- rt_eye ---
    p1, p2 = rotated_coords[189], rotated_coords[123]
    rt_eye_crop = mask_img[min(p1[1], p2[1]):max(p1[1], p2[1]),
                           min(p1[0], p2[0]):max(p1[0], p2[0])]
    # print(str(region_folders['rt_eye'] / f"{file.stem}_rt_eye{file.suffix}"),"rt_eye_crop")

    # --- lf_eye ---
    p3, p4 = rotated_coords[413], rotated_coords[352]
    lf_eye_crop = mask_img[min(p3[1], p4[1]):max(p3[1], p4[1]),
                           min(p3[0], p4[0]):max(p3[0], p4[0])]
    # print(str(region_folders['lf_eye'] / f"{file.stem}_lf_eye{file.suffix}"),"lf_eye_crop")

    # --- nasolabial ---
    box = get_overlap_rect(rotated_coords[118], rotated_coords[434],
                           rotated_coords[347], rotated_coords[214])
    if box:
        x_min, y_min, x_max, y_max = box
        nasolabial_crop = mask_img[y_min:y_max, x_min:x_max]
        # print(str(region_folders['nasolabial'] / f"{file.stem}_nasolabial{file.suffix}"),"nasolabial_crop")

    # --- oral ---
    box = get_overlap_rect(rotated_coords[207], rotated_coords[379],
                           rotated_coords[427], rotated_coords[150])
    if box:
        x_min, y_min, x_max, y_max = box
        oral_crop = mask_img[y_min:y_max, x_min:x_max]
        # print(str(region_folders['oral'] / f"{file.stem}_oral{file.suffix}"),"oral_crop")

    # --- rt_cheek ---
    p1, p2 = rotated_coords[234], rotated_coords[212]
    x_min, y_min = min(p1[0], p2[0]), min(p1[1], p2[1])
    x_max, y_max = max(p1[0], p2[0]), max(p1[1], p2[1])
    rt_cheek_crop = mask_img[y_min:y_max, x_min:x_max]
    # print(str(region_folders['rt_cheek'] / f"{file.stem}_rt_cheek{file.suffix}"),"rt_cheek_crop")

    # --- lt_cheek ---
    p3, p4 = rotated_coords[454], rotated_coords[432]
    x_min, y_min = min(p3[0], p4[0]), min(p3[1], p4[1])
    x_max, y_max = max(p3[0], p4[0]), max(p3[1], p4[1])
    lt_cheek_crop = mask_img[y_min:y_max, x_min:x_max]
    # print(str(region_folders['lt_cheek'] / f"{file.stem}_lt_cheek{file.suffix}"),"lt_cheek_crop")

    # print(f"✅ Processed: {file_path}")

    return forehead_crop, rt_eye_crop, lf_eye_crop, nasolabial_crop, oral_crop, rt_cheek_crop, lt_cheek_crop