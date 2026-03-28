import sys
import os
import pandas as pd
from multiprocessing import get_context
from tqdm import tqdm
from worker import process_wrinkle_image, initializer

# filtered_df를 읽어올 csv 경로 (셀3에서 사용한 csv와 동일)
CSV_PATH = '/home/hocheol/inskin_ai/no_track/datasets/wrinkle_data/labels/wrinkle-260219_034.csv'

# filtered_df 생성 (셀3과 동일하게)
data_df = pd.read_csv(CSV_PATH).iloc[:, :-1]
filtered_df = data_df[~data_df.iloc[:, 1:].isna().all(axis=1)].reset_index(drop=True)

# filename 컬럼이 첫 번째 컬럼이라고 가정
filenames = filtered_df.iloc[:, 0].tolist()

NUM_CORES = 4

if __name__ == '__main__':
    try:
        with get_context("spawn").Pool(processes=NUM_CORES, initializer=initializer) as pool:
            results = list(tqdm(pool.imap_unordered(process_wrinkle_image, filenames), total=len(filenames)))
        
        success_count = sum(1 for r in results if r.startswith("success"))
        fail_count = sum(1 for r in results if r.startswith("failed"))
        skipped_count = sum(1 for r in results if r.startswith("skipped"))

        fail_list = [r for r in results if r.startswith('failed')]

        print(f"Wrinkle preprocessing completed. Success: {success_count}, Fail: {fail_count}, Skipped: {skipped_count}")
        for fail_item in fail_list:
            print(fail_item)
    except Exception as e:
        print(e)
        raise e
