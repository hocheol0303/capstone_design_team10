import json
import os
import tensorflow as tf
from utils.constants import SECTORS
from dataloaders.base_dataloader import Augmentation

keys_tensor = tf.constant(SECTORS)
vals_tensor = tf.constant(range(len(SECTORS)), dtype=tf.int32)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1
)


def load_json(meta_file, split='train'):
    items = []
    with open(meta_file, 'r') as f:
        path_list = json.load(f)

    filename = f'wrinkle_{split}.json'
    for json_dir in path_list:
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"JSON file not found: {json_dir}")
        with open(os.path.join(json_dir, filename), 'r') as f:
            items.extend(json.load(f))
    return items


class DataLoader:
    def __init__(self, item_list, root_dir, config, is_training=True):
        self.img_shape = config['input_shape'][:2]
        self.batch_size = config['batch_size'] if is_training else 16
        self.is_training = is_training
        self.aug_list = config.get('aug_list', []) if is_training else []

        self.img_paths = []
        self.labels = []
        self.sectors = []

        for item in item_list:
            for sector in SECTORS:
                label = item.get(sector, -1)
                path = item.get(f'{sector}_path', '')
                if label == -1 or not path:
                    continue
                self.img_paths.append(os.path.join(root_dir, path))
                self.labels.append(float(label))
                self.sectors.append(sector)

    def _preprocess(self, path, label, sector):
        img = tf.io.read_file(path)
        if tf.strings.regex_full_match(path, r'.*\.png'):
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_shape)
        sector_id = table.lookup(sector)

        if self.aug_list:
            img = Augmentation.apply_aug(img / 255.0, self.aug_list) * 255.0
            img = tf.clip_by_value(img, 0.0, 255.0)

        return {'image_input': img, 'sector_input': sector_id}, label

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels, self.sectors))
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=min(len(self.img_paths), 10000))
        dataset = dataset.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        return dataset.prefetch(tf.data.AUTOTUNE)


def get_datasets(config):
    meta_file = config['meta_file']
    root_dir = config['root_dir']

    train_items = load_json(meta_file, split='train')
    val_items = load_json(meta_file, split='val')

    train_loader = DataLoader(train_items, root_dir, config, is_training=True)
    val_loader = DataLoader(val_items, root_dir, config, is_training=False)

    print(f"\033[42m훈련 샘플 수: {len(train_loader.img_paths)}\033[0m")
    print(f"\033[42m검증 샘플 수: {len(val_loader.img_paths)}\033[0m")

    return train_loader.get_dataset(), val_loader.get_dataset()


def get_test_datasets(config):
    test_items = load_json(config['meta_file'], split='test')
    print(f"\033[36mLoaded \033[91m{len(test_items)}\033[36m test samples.\033[0m")
    return DataLoader(test_items, config['root_dir'], config, is_training=False).get_dataset()
