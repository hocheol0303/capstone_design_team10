import os
import pandas as pd
from tqdm import tqdm
from exif import Image


if __name__ == "__main__":
    root = "/home/work/hocheol_dir/workspace/_dataset/datasets/_origin_data"
    for dataset in os.listdir(root):
        if dataset.endswith('_data'):
            save_path = os.path.join(root, dataset, f"{dataset}_EXIF_info.csv")
            if os.path.exists(save_path):
                print(f"EXIF info for dataset {dataset} already exists. Skipping...")
                continue
            print(f"Processing dataset: {dataset}")
            file_paths = []
            file_info = {
                'brightness_value':[], 
                'photographic_sensitivity':[], 
                'exposure_time':[], 
                'exposure_bias_value':[]
            }

            dir_name = os.path.join(root, dataset, 'data')

            for file in tqdm(os.listdir(dir_name)):
                file_path = os.path.join(dir_name, file)
                if file_path.lower().endswith('.png'):
                    continue
                with open(file_path, 'rb') as f:
                    image = Image(f)
                    if not image.has_exif:
                        continue
                    file_paths.append(file)
                    for key in file_info.keys():
                        value = image.get(key, "unknown")
                        file_info[key].append(value)
            
            file_info['file_path'] = file_paths
            df = pd.DataFrame(file_info)
            df.to_csv(save_path, index=False)