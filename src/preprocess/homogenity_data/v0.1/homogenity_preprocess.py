"""
Homogenity 데이터셋 전처리 스크립트

1. age_data/v0.3의 이미지에 대해 radiance/texture 점수 추출
2. 전체 데이터 기준 percentile로 10개 클래스 레이블 변환 (0=최상, 9=최하)
3. user_id 기반 8:1:1 train/val/test split (tr_val_test_split 사용)
4. no_track/datasets/homogenity/v0.1/{dataset}/label/ 에 JSON 저장
   포맷: [{"user_id": ..., "path": ..., "radiance_label": ..., "texture_label": ...}, ...]
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe.tasks.python.core.base_options import BaseOptions

# ============================================================
# 경로 설정
# ============================================================
AGE_DATA_ROOT    = "/home/hocheol/inskin_ai/no_track/datasets/age_data/v0.3/v0.3_data"
ORIGIN_DATA_ROOT = "/home/hocheol/inskin_ai/no_track/datasets/origin_data-simple"
OUTPUT_ROOT      = "/home/hocheol/inskin_ai/no_track/datasets/homogenity/v0.1"
MODEL_PATH       = "/home/hocheol/inskin_ai/EfficientNet/src/face_landmarker.task"
INTERMEDIATE_CSV = os.path.join(OUTPUT_ROOT, "raw_scores.csv")
MAX_WORKERS      = 8

# 분석 파라미터 (radiance_texture_test.py 동일)
SHADOW_P    = 7
R_CHEEK_EXT = [31, 228, 229, 230, 120, 100, 142, 36, 205, 207, 214, 192, 213, 147, 123, 111]
L_CHEEK_EXT = [261, 448, 449, 450, 349, 329, 371, 266, 425, 427, 434, 416, 433, 376, 352, 340]

# ============================================================
# 분석 유틸 (radiance_texture_test.py에서 복사)
# ============================================================

def get_normalized_smoothness(l_ch, mask):
    """Radiance 점수: 높을수록 균질 (좋음)"""
    valid_m = (mask == 255)
    if np.sum(valid_m) == 0:
        return None
    mean_l      = np.mean(l_ch[valid_m])
    dx          = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    dy          = cv2.Sobel(l_ch, cv2.CV_64F, 0, 1, ksize=3)
    avg_grad    = np.mean(np.sqrt(dx**2 + dy**2)[valid_m])
    return 100 / (1 + (avg_grad / (mean_l + 1e-5)) * 10)


class FaceAnalyzer:
    def __init__(self, model_path):
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = face_landmarker.FaceLandmarker.create_from_options(options)

    def _to_mp_img(self, bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def process(self, bgr):
        """얼굴 정렬·크롭 후 (final_img, final_det) 반환. 실패 시 (None, None)."""
        h, w = bgr.shape[:2]
        det = self.detector.detect(self._to_mp_img(bgr))
        if not det.face_landmarks:
            return None, None

        lmk   = det.face_landmarks[0]
        angle = np.degrees(np.arctan2(lmk[152].x*w - lmk[168].x*w,
                                      lmk[152].y*h - lmk[168].y*h))
        M       = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
        aligned = cv2.warpAffine(bgr, M, (w, h))

        det_v = self.detector.detect(self._to_mp_img(aligned))
        if not det_v.face_landmarks:
            return None, None
        lmkv = det_v.face_landmarks[0]

        L_val = np.hypot(lmkv[152].x*w - lmkv[168].x*w,
                         lmkv[152].y*h - lmkv[168].y*h)
        cx, cy = lmkv[168].x*w, lmkv[168].y*h + 0.35*L_val
        side   = int(L_val * 2.0)
        x0, y0 = int(cx - side/2), int(cy - side/2)

        crop = np.zeros((side, side, 3), dtype=np.uint8)
        sx, sy = max(0, x0), max(0, y0)
        ex, ey = min(w, x0+side), min(h, y0+side)
        crop[max(0,-y0):max(0,-y0)+(ey-sy),
             max(0,-x0):max(0,-x0)+(ex-sx)] = aligned[sy:ey, sx:ex]
        final_img = cv2.resize(crop, (1024, 1024))

        final_det = self.detector.detect(self._to_mp_img(final_img))
        if not final_det.face_landmarks:
            return None, None
        return final_img, final_det


def compute_scores(final_img, final_det):
    """정렬된 이미지와 landmarks로 radiance/texture 점수 계산."""
    lmks  = final_det.face_landmarks[0]
    l_ch  = cv2.split(cv2.cvtColor(final_img, cv2.COLOR_BGR2Lab))[0]

    cheek_m = np.zeros((1024, 1024), dtype=np.uint8)
    pts = [np.array([(lmks[i].x*1024, lmks[i].y*1024)
                     for i in R_CHEEK_EXT + L_CHEEK_EXT], np.int32)]
    cv2.fillPoly(cheek_m, pts, 255)

    if not np.any(cheek_m == 255):
        return None, None

    shadow_val = np.percentile(l_ch[cheek_m == 255], SHADOW_P)
    clean_m    = cv2.subtract(cheek_m,
                              ((l_ch < shadow_val) & (cheek_m == 255)).astype(np.uint8) * 255)

    rad_score = get_normalized_smoothness(l_ch, clean_m)

    l_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_ch)
    lap        = np.absolute(cv2.Laplacian(cv2.GaussianBlur(l_enhanced, (3, 3), 0), cv2.CV_64F))
    tex_score  = float(np.mean(lap[clean_m == 255])) if np.any(clean_m == 255) else None

    return rad_score, tex_score


# ============================================================
# tr_val_test_split (age_split.ipynb에서 복사, label 컬럼 파라미터화)
# ============================================================

def tr_val_test_split(df, label_col='age_class'):
    class_counts  = df[label_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    stratify_df   = df[df[label_col].isin(valid_classes)]
    rare_df       = df[~df[label_col].isin(valid_classes)]

    train_df, tmp_df = train_test_split(
        stratify_df, test_size=0.2, random_state=42, stratify=stratify_df[label_col])
    train_df = pd.concat([train_df, rare_df], ignore_index=True)

    try:
        val_df, test_df = train_test_split(
            tmp_df, test_size=0.5, stratify=tmp_df[label_col], random_state=42)
    except ValueError:
        tmp_class_counts  = tmp_df[label_col].value_counts()
        tmp_valid_classes = tmp_class_counts[tmp_class_counts >= 2].index
        tmp_stratify_df   = tmp_df[tmp_df[label_col].isin(tmp_valid_classes)]
        tmp_rare_df       = tmp_df[~tmp_df[label_col].isin(tmp_valid_classes)]
        val_df, test_df   = train_test_split(
            tmp_stratify_df, test_size=0.5, random_state=42,
            stratify=tmp_stratify_df[label_col])
        val_df = pd.concat([val_df, tmp_rare_df], ignore_index=True)

    return train_df, val_df, test_df


# ============================================================
# 점수 → 클래스 레이블 변환
# ============================================================

def scores_to_classes(scores, reverse=True, n_bins=10):
    """
    scores : list of float | None
    reverse: True  → 높은 점수 = class 0 (radiance: 높을수록 좋음)
             False → 낮은 점수 = class 0 (texture: 낮을수록 좋음)
    Returns: list of int, None → -1
    """
    valid_idx    = [i for i, s in enumerate(scores) if s is not None]
    valid_scores = [scores[i] for i in valid_idx]
    if not valid_scores:
        return [-1] * len(scores)

    # 순위 → 클래스 (상위 10% = class 0)
    order = sorted(range(len(valid_scores)),
                   key=lambda i: valid_scores[i], reverse=reverse)
    rank_label = [0] * len(valid_scores)
    for rank, idx in enumerate(order):
        rank_label[idx] = min(int(rank / len(valid_scores) * n_bins), n_bins - 1)

    result = [-1] * len(scores)
    for i, orig_idx in enumerate(valid_idx):
        result[orig_idx] = rank_label[i]
    return result


# ============================================================
# 메인
# ============================================================

def collect_image_list():
    """age JSON에서 전체 이미지 목록 수집 (user_id, path 포함).

    - path: age_data 기준 상대경로 (모델 입력용)
    - origin_img_path: origin_data-simple 경로 (점수 추출용)
    age_data에 존재하는 이미지 중 origin_data-simple에도 존재하는 것만 수집.
    """
    items = []
    seen  = set()
    for dataset in sorted(os.listdir(AGE_DATA_ROOT)):
        print(f"데이터셋 '{dataset}' 처리 중...")
        dataset_path = os.path.join(AGE_DATA_ROOT, dataset)
        if not os.path.isdir(dataset_path):
            continue
        label_dir = os.path.join(dataset_path, 'label')
        for split in ('train', 'val', 'test'):
            json_path = os.path.join(label_dir, f'age_{split}.json')
            if not os.path.exists(json_path):
                continue
            with open(json_path, 'r') as f:
                for rec in json.load(f):
                    if rec['path'] in seen:
                        continue
                    seen.add(rec['path'])

                    filename = os.path.basename(rec['path'])
                    user_id  = str(rec['user_id'])
                    # origin_data-simple/{dataset}/{user_id}/{filename}
                    origin_path = os.path.join(ORIGIN_DATA_ROOT, dataset, user_id, filename)
                    if not os.path.exists(origin_path):
                        continue

                    items.append({
                        'dataset':         dataset,
                        'user_id':         user_id,
                        'path':            rec['path'],        # age_data 상대경로
                        'origin_img_path': origin_path,       # 점수 추출용
                    })
    return items


# ============================================================
# 멀티프로세싱 워커 (hobbang_multiprocess와 동일한 패턴)
# ============================================================

_analyzer = None  # 워커 프로세스별 전역 인스턴스

def _init_worker(model_path):
    """ProcessPoolExecutor initializer: 워커 프로세스마다 한 번 실행."""
    global _analyzer
    _analyzer = FaceAnalyzer(model_path)


def _process_item(item):
    """단일 이미지 처리 (워커 프로세스에서 실행).
    성공 시 결과 dict, 실패/건너뜀 시 None 반환.
    """
    img = cv2.imread(item['origin_img_path'])
    if img is None:
        return None
    final_img, final_det = _analyzer.process(img)
    if final_img is None:
        return None
    rad_score, tex_score = compute_scores(final_img, final_det)
    if rad_score is None and tex_score is None:
        return None
    return {
        'dataset':   item['dataset'],
        'user_id':   item['user_id'],
        'path':      item['path'],
        'rad_score': rad_score,
        'tex_score': tex_score,
    }


def run_analysis(items):
    """얼굴 분석 실행 후 DataFrame 반환 (멀티프로세싱).
    점수 추출은 origin_data-simple 이미지로, path는 age_data 기준 저장.
    """
    rows = []
    fail = 0
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(MODEL_PATH,)
    ) as executor:
        futures = {executor.submit(_process_item, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(items), desc='Face analysis'):
            try:
                result = future.result()
                if result is not None:
                    rows.append(result)
            except Exception as e:
                fail += 1
    if fail:
        print(f"  처리 실패: {fail}개")
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. 이미지 목록 수집
    print("이미지 목록 수집 중...")
    items = collect_image_list()
    print(f"  총 {len(items)}개 이미지")

    # 2. 얼굴 분석 (중간 결과 CSV가 있으면 재사용)
    if os.path.exists(INTERMEDIATE_CSV):
        print(f"기존 분석 결과 로드: {INTERMEDIATE_CSV}")
        df = pd.read_csv(INTERMEDIATE_CSV, dtype={'user_id': str})
    else:
        print("얼굴 분석 실행 중 (시간이 오래 걸릴 수 있습니다)...")
        df = run_analysis(items)
        df.to_csv(INTERMEDIATE_CSV, index=False)
        print(f"  분석 완료 ({len(df)}개), 저장: {INTERMEDIATE_CSV}")

    # 3. 전체 기준 클래스 레이블 할당 (0=최상, 9=최하)
    df['radiance_label'] = scores_to_classes(df['rad_score'].tolist(), reverse=True)
    df['texture_label']  = scores_to_classes(df['tex_score'].tolist(), reverse=False)

    # 둘 다 계산 실패(-1)인 경우 제외
    df = df[~((df['radiance_label'] == -1) & (df['texture_label'] == -1))].copy()
    print(f"레이블 할당 완료 ({len(df)}개)")

    # (dataset, user_id) 복합키: 데이터셋 간 user_id 충돌 방지
    df['user_key'] = df['dataset'] + '/' + df['user_id']

    # 4. 데이터셋별 train/val/test split 후 JSON 저장
    label_paths = []
    for dataset, ds_df in df.groupby('dataset'):
        # user_key별 대표 label: radiance 중앙값, 5구간으로 축소하여 stratify 안정화
        rad_med = ds_df.groupby('user_key')['radiance_label'].median().round().astype(int).clip(0, 9)
        user_label = rad_med.reset_index()
        user_label.columns = ['user_key', 'split_label']
        user_label['split_label'] = (user_label['split_label'] // 2).clip(0, 4)

        train_u, val_u, test_u = tr_val_test_split(user_label, label_col='split_label')
        train_keys = set(train_u['user_key'])
        val_keys   = set(val_u['user_key'])
        test_keys  = set(test_u['user_key'])

        out_dir = os.path.join(OUTPUT_ROOT, dataset, 'label')
        os.makedirs(out_dir, exist_ok=True)

        cols = ['user_id', 'path', 'radiance_label', 'texture_label']
        for split_name, keys in [('train', train_keys), ('val', val_keys), ('test', test_keys)]:
            records = ds_df[ds_df['user_key'].isin(keys)][cols].to_dict(orient='records')
            out_path = os.path.join(out_dir, f'homogenity_{split_name}.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

        print(f"  [{dataset}] users train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}"
              f" / images train={len(ds_df[ds_df['user_key'].isin(train_keys)])}"
              f" val={len(ds_df[ds_df['user_key'].isin(val_keys)])}"
              f" test={len(ds_df[ds_df['user_key'].isin(test_keys)])}")
        label_paths.append(out_dir)

    # 5. 상위 label_paths.json 저장
    paths_file = os.path.join(OUTPUT_ROOT, 'homogenity_label_paths.json')
    with open(paths_file, 'w', encoding='utf-8') as f:
        json.dump(label_paths, f, indent=2, ensure_ascii=False)

    print(f"\n완료! 저장 위치: {OUTPUT_ROOT}")
    print(f"label_paths.json: {paths_file}")


if __name__ == '__main__':
    main()
