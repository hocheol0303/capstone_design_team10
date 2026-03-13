import json
import os
import tensorflow as tf
from dataloaders.base_dataloader import Augmentation

BOX_KEYS = ['forehead', 'right_eye', 'left_eye', 'nasolabial', 'oral']


def load_json(meta_file, split='train'):
    dataset = []
    with open(meta_file, 'r') as f:
        path_list = json.load(f)

    filename = f'age_{split}.json'
    for json_dir in path_list:
        if not os.path.exists(json_dir):
            raise FileNotFoundError(f"JSON file not found: {json_dir}")
        with open(os.path.join(json_dir, filename), 'r') as f:
            dataset.extend(json.load(f))

    if split != 'test':
        dataset = [item for item in dataset
                   if item['size']['width'] >= 384 and item['size']['height'] >= 384]
    return dataset


def _mask_image(img, box):
    y_min = tf.cast(box[0], tf.int32)
    x_min = tf.cast(box[1], tf.int32)
    y_max = tf.cast(box[2], tf.int32)
    x_max = tf.cast(box[3], tf.int32)

    height = tf.shape(img)[0]
    width = tf.shape(img)[1]

    rows = tf.range(height, dtype=tf.int32)
    cols = tf.range(width, dtype=tf.int32)
    rows_mask = (rows >= y_min) & (rows < y_max)
    cols_mask = (cols >= x_min) & (cols < x_max)
    mask_2d = tf.logical_not(tf.logical_and(rows_mask[:, tf.newaxis], cols_mask[tf.newaxis, :]))
    mask_3d = tf.cast(mask_2d[:, :, tf.newaxis], img.dtype)
    return img * mask_3d


class DataLoader:
    def __init__(self, item_list, root_dir, config, is_training=True):
        self.img_shape = config['input_shape'][:2]
        self.batch_size = config['batch_size'] if is_training else 16
        self.is_training = is_training
        self.onehot = config.get('one_hot', False)
        self.aug_list = config.get('aug_list', []) if is_training else []
        self.use_wrinkle_mask = is_training and 'wrinkle_mask' in self.aug_list

        self.img_paths = []
        self.labels = []
        self.genders = []
        self.boxes = {key: [] for key in BOX_KEYS}

        for item in item_list:
            self.img_paths.append(os.path.join(root_dir, item['path']))
            self.labels.append(item['age_class'])
            self.genders.append(item['gender'])
            if self.use_wrinkle_mask and 'sector_boxes' in item:
                for key in BOX_KEYS:
                    box = item['sector_boxes'][key]
                    self.boxes[key].append((box['y_min'], box['x_min'], box['y_max'], box['x_max']))

    def _preprocess(self, path, label, gender):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.img_shape)
        img = tf.cast(img, tf.float32)

        if self.onehot:
            label_tensor = tf.one_hot(label, depth=7)
        else:
            label_tensor = tf.cast(label, tf.int32)

        male_weight = tf.cast(tf.equal(gender, b'male'), tf.float32)
        female_weight = tf.cast(tf.equal(gender, b'female'), tf.float32)

        if self.aug_list:
            img = Augmentation.apply_aug(img / 255.0, self.aug_list) * 255.0
            img = tf.clip_by_value(img, 0.0, 255.0)

        return img, {'male': label_tensor, 'female': label_tensor}, {'male': male_weight, 'female': female_weight}

    def _preprocess_with_mask(self, path, label, gender, forehead, right_eye, left_eye, nasolabial, oral):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)

        if tf.random.uniform(()) < 0.5:
            img = _mask_image(img, forehead)
        if tf.random.uniform(()) < 0.5:
            img = _mask_image(img, right_eye)
            img = _mask_image(img, left_eye)
        if tf.random.uniform(()) < 0.5:
            img = _mask_image(img, nasolabial)
        if tf.random.uniform(()) < 0.5:
            img = _mask_image(img, oral)
        if tf.random.uniform(()) < 0.5:
            img = tf.image.flip_left_right(img)

        img = tf.image.resize(img, self.img_shape)
        img = tf.cast(img, tf.float32)

        if self.onehot:
            label_tensor = tf.one_hot(label, depth=7)
        else:
            label_tensor = tf.cast(label, tf.int32)

        male_weight = tf.cast(tf.equal(gender, b'male'), tf.float32)
        female_weight = tf.cast(tf.equal(gender, b'female'), tf.float32)

        if self.aug_list:
            img = Augmentation.apply_aug(img / 255.0, self.aug_list) * 255.0
            img = tf.clip_by_value(img, 0.0, 255.0)

        return img, {'male': label_tensor, 'female': label_tensor}, {'male': male_weight, 'female': female_weight}

    def get_dataset(self):
        if self.use_wrinkle_mask:
            dataset = tf.data.Dataset.from_tensor_slices((
                self.img_paths, self.labels, self.genders,
                *[self.boxes[k] for k in BOX_KEYS]
            ))
            preprocess_fn = self._preprocess_with_mask
        else:
            dataset = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels, self.genders))
            preprocess_fn = self._preprocess

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=min(len(self.img_paths), 10000))
        dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
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


def get_test_datasets(config):
    test_items = load_json(config['meta_file'], split='test')
    print(f"\033[36mLoaded \033[91m{len(test_items)}\033[36m test samples.\033[0m")
    return DataLoader(test_items, config['root_dir'], config, is_training=False).get_dataset()
