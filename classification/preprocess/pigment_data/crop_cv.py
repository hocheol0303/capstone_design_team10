import cv2
import mediapipe as mp



def crop_cheeks(image_rgb, landmarks=None):
    """
    주어진 이미지에서 mediapipe face landmarks를 이용하여
    좌측볼, 우측볼 이미지를 crop하여 반환합니다.

    Returns:
        left_cheek (numpy array), right_cheek (numpy array)
    """
    # 이미지 로드
    h, w, _ = image_rgb.shape

    def to_px(lm): return int(lm.x * w), int(lm.y * h)

    pt_0 = to_px(landmarks[0])
    pt_264 = to_px(landmarks[264])
    pt_34 = to_px(landmarks[34])

    # 좌측볼 crop (pt_0 ~ pt_264)
    x1_l, y1_l = min(pt_0[0], pt_264[0]), min(pt_0[1], pt_264[1])
    x2_l, y2_l = max(pt_0[0], pt_264[0]), max(pt_0[1], pt_264[1])
    left_cheek = image_rgb[y1_l:y2_l, x1_l:x2_l]

    # 우측볼 crop (pt_0 ~ pt_34)
    x1_r, y1_r = min(pt_0[0], pt_34[0]), min(pt_0[1], pt_34[1])
    x2_r, y2_r = max(pt_0[0], pt_34[0]), max(pt_0[1], pt_34[1])
    right_cheek = image_rgb[y1_r:y2_r, x1_r:x2_r]

    # left_cheek = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2RGB)
    # right_cheek = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2RGB)
    
    return left_cheek, right_cheek

def crop_left_cheek(image_rgb, landmarks=None):
    """
    주어진 이미지에서 mediapipe face landmarks를 이용하여
    좌측볼 이미지를 crop하여 반환합니다.

    Returns:
        left_cheek (numpy array)
    """

    # 이미지 로드
    h, w, _ = image_rgb.shape

    def to_px(lm): return int(lm.x * w), int(lm.y * h)

    pt_0 = to_px(landmarks[0])
    pt_264 = to_px(landmarks[264])

    # 좌측볼 crop (pt_0 ~ pt_264)
    x1_l, y1_l = min(pt_0[0], pt_264[0]), min(pt_0[1], pt_264[1])
    x2_l, y2_l = max(pt_0[0], pt_264[0]), max(pt_0[1], pt_264[1])
    left_cheek = image_rgb[y1_l:y2_l, x1_l:x2_l]

    return left_cheek

def crop_right_cheek(image_rgb, landmarks=None):
    """
    주어진 이미지에서 mediapipe face landmarks를 이용하여
    우측볼 이미지를 crop하여 반환합니다.

    Returns:
    right_cheek (numpy array)
    """
    # 이미지 로드
    h, w, _ = image_rgb.shape

    def to_px(lm): return int(lm.x * w), int(lm.y * h)

    pt_0 = to_px(landmarks[0])
    pt_34 = to_px(landmarks[34])

    # 우측볼 crop (pt_0 ~ pt_34)
    x1_r, y1_r = min(pt_0[0], pt_34[0]), min(pt_0[1], pt_34[1])
    x2_r, y2_r = max(pt_0[0], pt_34[0]), max(pt_0[1], pt_34[1])
    right_cheek = image_rgb[y1_r:y2_r, x1_r:x2_r]

    return right_cheek