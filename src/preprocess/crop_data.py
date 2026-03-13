import cv2
import mediapipe as mp

def crop_cheeks_from_detection_result(landmarks, mp_image):
    image = mp_image.numpy_view()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    def to_px(lm): return int(lm.x * w), int(lm.y * h)

    point_0 = to_px(landmarks[0])
    point_264 = to_px(landmarks[264])
    point_34 = to_px(landmarks[34])

    left_x1, left_y1 = min(point_0[0], point_264[0]), min(point_0[1], point_264[1])
    left_x2, left_y2 = max(point_0[0], point_264[0]), max(point_0[1], point_264[1])
    left_cheek = image[left_y1:left_y2, left_x1:left_x2]

    right_x1, right_y1 = min(point_0[0], point_34[0]), min(point_0[1], point_34[1])
    right_x2, right_y2 = max(point_0[0], point_34[0]), max(point_0[1], point_34[1])
    right_cheek = image[right_y1:right_y2, right_x1:right_x2]

    return left_cheek, right_cheek

def crop_left_cheek_from_detection_result(landmarks, mp_image):
    image = mp_image.numpy_view()
    h, w, _ = image.shape
    def to_px(lm): return int(lm.x * w), int(lm.y * h)
    
    point_0 = to_px(landmarks[0])
    point_264 = to_px(landmarks[264])

    left_x1, left_y1 = min(point_0[0], point_264[0]), min(point_0[1], point_264[1])
    left_x2, left_y2 = max(point_0[0], point_264[0]), max(point_0[1], point_264[1])
    left_cheek = image[left_y1:left_y2, left_x1:left_x2]

    return left_cheek

def crop_right_cheek_from_detection_result(landmarks, mp_image):
    image = mp_image.numpy_view()
    h, w, _ = image.shape
    def to_px(lm): return int(lm.x * w), int(lm.y * h)

    point_0 = to_px(landmarks[0])
    point_34 = to_px(landmarks[34])

    right_x1, right_y1 = min(point_0[0], point_34[0]), min(point_0[1], point_34[1])
    right_x2, right_y2 = max(point_0[0], point_34[0]), max(point_0[1], point_34[1])
    right_cheek = image[right_y1:right_y2, right_x1:right_x2]

    return right_cheek

def crop_cheeks_from_image(image_path):
    """
    주어진 이미지에서 mediapipe face landmarks를 이용하여
    좌측볼, 우측볼 이미지를 crop하여 반환합니다.

    Returns:
        left_cheek (numpy array), right_cheek (numpy array)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("얼굴이 감지되지 않았습니다.")

        landmarks = results.multi_face_landmarks[0].landmark

        def to_px(lm): return int(lm.x * w), int(lm.y * h)

        pt_0 = to_px(landmarks[0])
        pt_264 = to_px(landmarks[264])
        pt_34 = to_px(landmarks[34])

        # 좌측볼 crop (pt_0 ~ pt_264)
        x1_l, y1_l = min(pt_0[0], pt_264[0]), min(pt_0[1], pt_264[1])
        x2_l, y2_l = max(pt_0[0], pt_264[0]), max(pt_0[1], pt_264[1])
        left_cheek = image[y1_l:y2_l, x1_l:x2_l]

        # 우측볼 crop (pt_0 ~ pt_34)
        x1_r, y1_r = min(pt_0[0], pt_34[0]), min(pt_0[1], pt_34[1])
        x2_r, y2_r = max(pt_0[0], pt_34[0]), max(pt_0[1], pt_34[1])
        right_cheek = image[y1_r:y2_r, x1_r:x2_r]

        # left_cheek = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2RGB)
        # right_cheek = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2RGB)
        
        return left_cheek, right_cheek

# mp_image로 바꿔야함
def crop_left_cheek_from_image(image_path):
    """
    얼굴 왼쪽면에서 왼쪽 볼을 추출한다.
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("얼굴이 감지되지 않았습니다.")

        landmarks = results.multi_face_landmarks[0].landmark

        def to_px(lm): return int(lm.x * w), int(lm.y * h)

        pt_0 = to_px(landmarks[0])
        pt_264 = to_px(landmarks[264])

        # 좌측볼 crop (pt_0 ~ pt_264)
        x1_l, y1_l = min(pt_0[0], pt_264[0]), min(pt_0[1], pt_264[1])
        x2_l, y2_l = max(pt_0[0], pt_264[0]), max(pt_0[1], pt_264[1])
        left_cheek = image[y1_l:y2_l, x1_l:x2_l]
        
        return left_cheek

# def crop_right_cheek_from_image(image_path):
#     """
#     얼굴 왼쪽면에서 왼쪽 볼을 추출한다.
#     """
#     # 이미지 로드
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

#     h, w, _ = image.shape
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
#         results = face_mesh.process(image_rgb)

#         if not results.multi_face_landmarks:
#             raise ValueError("얼굴이 감지되지 않았습니다.")

#         landmarks = results.multi_face_landmarks[0].landmark

#         def to_px(lm): return int(lm.x * w), int(lm.y * h)

#         pt_0 = to_px(landmarks[0])
#         pt_34 = to_px(landmarks[34])

#         # 우측볼 crop (pt_0 ~ pt_34)
#         x1_r, y1_r = min(pt_0[0], pt_34[0]), min(pt_0[1], pt_34[1])
#         x2_r, y2_r = max(pt_0[0], pt_34[0]), max(pt_0[1], pt_34[1])
#         right_cheek = image[y1_r:y2_r, x1_r:x2_r]
        
#         return right_cheek


# def crop_cheeks_and_forehead_from_image(image_path):
#     """
#     주어진 이미지에서 mediapipe face landmarks를 이용하여
#     좌측볼, 우측볼 이미지를 crop하여 반환합니다.

#     Returns:
#         left_cheek (numpy array), right_cheek (numpy array)
#     """
#     # 이미지 로드
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"이미지를 열 수 없습니다: {image_path}")

#     h, w, _ = image.shape
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # mediapipe face mesh
#     mp_face_mesh = mp.solutions.face_mesh
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
#         results = face_mesh.process(image_rgb)

#         if not results.multi_face_landmarks:
#             raise ValueError("얼굴이 감지되지 않았습니다.")

#         landmarks = results.multi_face_landmarks[0].landmark

#         def to_px(lm): return int(lm.x * w), int(lm.y * h)

#         # 볼용
#         pt_0 = to_px(landmarks[0])
#         pt_264 = to_px(landmarks[264])
#         pt_34 = to_px(landmarks[34])

#         # 이마용
#         pt_67 = to_px(landmarks[67])    # 좌상
#         pt_27 = to_px(landmarks[27])    # 좌하
#         pt_297 = to_px(landmarks[297])  # 우상
#         pt_257 = to_px(landmarks[257])  # 우하

#         # cv2의 좌표계: 오른쪽이 큰쪽, 아래쪽이 큰쪽
#         # 좌측볼 crop (pt_0 ~ pt_264)
#         x1_l, y1_l = min(pt_0[0], pt_264[0]), min(pt_0[1], pt_264[1])
#         x2_l, y2_l = max(pt_0[0], pt_264[0]), max(pt_0[1], pt_264[1])
#         left_cheek = image[y1_l:y2_l, x1_l:x2_l]

#         # 우측볼 crop (pt_0 ~ pt_34)
#         x1_r, y1_r = min(pt_0[0], pt_34[0]), min(pt_0[1], pt_34[1])
#         x2_r, y2_r = max(pt_0[0], pt_34[0]), max(pt_0[1], pt_34[1])
#         right_cheek = image[y1_r:y2_r, x1_r:x2_r]

#         # 이마 crop : 좌상 67, 우하 257 박스, 좌하 27, 우상 297 박스의 intersection
#         x1_f = max(pt_67[0], pt_27[0])
#         y1_f = max(pt_67[1], pt_297[1])
        
#         x2_f = min(pt_257[0], pt_297[0])
#         y2_f = min(pt_257[1], pt_27[1])
#         forehead = image[y1_f:y2_f, x1_f:x2_f]
        
#         # left_cheek = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2RGB)
#         # right_cheek = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2RGB)
        
#         return forehead, left_cheek, right_cheek
