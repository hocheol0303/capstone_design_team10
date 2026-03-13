import os
import json
import tensorflow as tf
from .base_dataloader import Augmentation


def load_json(meta_file, split='train'):
    items = []
    with open(meta_file, 'r') as f:
        path_list = json.load(f)

    filename = f'homogenity_{split}.json'
    for json_dir in path_list:
        with open(os.path.join(json_dir, filename), 'r') as f:
            items.extend(json.load(f))
    return items


class DataLoader:
    def __init__(self, item_list, root_dir, config, is_training=True):
        self.img_shape = config['input_shape'][:2]
        self.batch_size = config['batch_size'] if is_training else 16
        self.is_training = is_training
        self.onehot = config.get('one_hot', False)
        self.aug_list = config.get('aug_list', []) if is_training else []

        self.img_paths = []
        self.rad_labels = []
        self.tex_labels = []

        for item in item_list:
            path = os.path.join(root_dir, item['path'])

            if not os.path.exists(path) or os.path.isdir(path):
                continue
            self.img_paths.append(path)
            self.rad_labels.append(item['radiance_label'])
            self.tex_labels.append(item['texture_label'])

    def _preprocess(self, path, label):
        img = tf.io.read_file(path)
        if tf.strings.regex_full_match(path, r'.*\.png'):
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_shape)

        if self.onehot:
            label['rad_out'] = tf.one_hot(label['rad_out'], depth=10)
            label['tex_out'] = tf.one_hot(label['tex_out'], depth=10)
        else:
            label['rad_out'] = tf.cast(label['rad_out'], tf.int32)
            label['tex_out'] = tf.cast(label['tex_out'], tf.int32)
            
        if self.aug_list:
            img = Augmentation.apply_aug(img / 255.0, self.aug_list) * 255.0
            img = tf.clip_by_value(img, 0.0, 255.0)

        return img, label

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, {'rad_out': self.rad_labels, 'tex_out': self.tex_labels}))

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
