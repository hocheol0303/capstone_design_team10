import json
import os
import pandas as pd
import tensorflow as tf
from dataloaders.base_dataloader import Augmentation


def load_json(meta_file, split='train'):
    dataset_bundle = []
    with open(meta_file, 'r') as f:
        path_list = json.load(f)

    filename = f'pig_{split}.json'
    for json_dir in path_list:
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"JSON file not found: {json_dir}")
        with open(os.path.join(json_dir, filename), 'r') as f:
            dataset_bundle.extend(json.load(f))

    items = []
    for item in dataset_bundle:
        for side in ('left', 'right'):
            path = item.get(f'{side}_path')
            label = item.get(f'{side}_label')
            if pd.isna(path) or pd.isna(label) or len(path) == 0:
                continue
            items.append({
                'user_id': item['user_id'],
                'path': path,
                'label': int(label)
            })
    return items


class DataLoader:
    def __init__(self, item_list, root_dir, config, is_training=True):
        self.img_shape = config['input_shape'][:2]
        self.batch_size = config['batch_size'] if is_training else 16
        self.is_training = is_training
        self.onehot = config.get('one_hot', False)
        self.aug_list = config.get('aug_list', []) if is_training else []

        self.img_paths = []
        self.labels = []

        for item in item_list:
            if is_training and (item['label'] == 6 or item['label'] == -1):
                continue
            self.img_paths.append(os.path.join(root_dir, item['path']))
            self.labels.append(item['label'])

    def _preprocess(self, path, label):
        img = tf.io.read_file(path)
        if tf.strings.regex_full_match(path, r'.*\.png'):
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_shape)
        img = tf.cast(img, tf.float32)

        if self.onehot:
            label = tf.one_hot(label, depth=5)
        else:
            label = tf.cast(label, tf.int32)

        if self.aug_list:
            img = Augmentation.apply_aug(img / 255.0, self.aug_list) * 255.0
            img = tf.clip_by_value(img, 0.0, 255.0)

        return img, label

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
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

    print(f"\033[36mLoaded \033[91m{len(train_items)}\033[36m training samples and \033[91m{len(val_items)}\033[36m validation samples.\033[0m")
    train_ds = DataLoader(train_items, root_dir, config, is_training=True).get_dataset()
    val_ds = DataLoader(val_items, root_dir, config, is_training=False).get_dataset()
    return train_ds, val_ds


def get_test_dataset(config):
    test_items = load_json(config['meta_file'], split='test')
    print(f"\033[36mLoaded \033[91m{len(test_items)}\033[36m test samples.\033[0m")
    return DataLoader(test_items, config['root_dir'], config, is_training=False).get_dataset()
