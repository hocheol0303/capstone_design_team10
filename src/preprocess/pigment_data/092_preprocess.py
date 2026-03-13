import matplotlib.pyplot as plt
import cv2
import sys
import os
import json
import re
from tqdm import tqdm

import os
from multiprocessing import get_context

sys.path.append('/home/work/hocheol_dir/workspace/_dataset/preprocess/pigment_data/v0.4/v0.4.0')
from worker import initializer, process_image2

if __name__ == '__main__':
    source_root = '/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data/092_data/origin_data'
    json_data = []
    filenames = []

    names = [name for name in os.listdir(source_root) if name.startswith('[라벨]')]
    for name in names:
        target_jsons = os.listdir(os.path.join(source_root, name))
        for target_json in target_jsons:
            target_path = os.path.join(source_root, name, target_json)
            with open(target_path, 'r') as f:
                for data in json.load(f):
                    json_data.append({
                        'user_id': re.search(r".*_.{1}_[0-9]+", data['filename']).group(),
                        'filename': data['filename'],
                    })
            print(len(json_data))

    for data in json_data:
        filenames.append(data['filename'])

    try:
        with get_context('spawn').Pool(processes=4, initializer=initializer) as pool:
            results = list(tqdm(pool.imap_unordered(process_image2, filenames), total=len(filenames)))
        success_count = sum(1 for r in results if r.startswith('success'))
        failure_count = len(results) - success_count

        fail_list = [r for r in results if r.startswith('failed')]
        print(f"Failed images: {len(fail_list)}")

    except Exception as e:
        print(f"An error occurred during multiprocessing: {str(e)}")

    os.makedirs('/home/work/hocheol_dir/workspace/_dataset/datasets/pigment_data/v0.4/v0.4.1/092_test_data/label/pigment_test.json', exist_ok=True)
    with open('/home/work/hocheol_dir/workspace/_dataset/datasets/pigment_data/v0.4/v0.4.1/092_test_data/label/pigment_test.json', 'w') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)