import os
import tensorflow as tf
# GPU를 사용할만큼만 메모리 할당
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
import pandas as pd
from _dataloaders.pig_3ch import get_test_dataset
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import cv2
import re
import yaml

'''
tflite로 저장할 당시의 input size로 지정해야함

EfficientNetV2-B0 : (224, 224)
EfficientNetV2-B1 : (240, 240)
EfficientNetV2-B2 : (260, 260)
EfficientNetV2-B3 : (300, 300)
EfficientNetV2-S : (384, 384)
EfficientNetV2-M : (480, 480)
EfficientNetV2-L : (480, 480)
EfficientNetV2-XL (비공식) : (512, 512)
'''

MODEL_PATH = "/home/work/hocheol_dir/workspace/save_models/pig_3ch/v0.4/251015_084023_S/epoch70_251015_084023.keras"
CONFIG_PATH = '/home/work/hocheol_dir/workspace/configs/pig_test.yaml'

RESULTS_DIR = '/home/work/hocheol_dir/workspace/inference/results'
COL_NAMES = ['path', 'label', 'pred', 'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4']

def get_feature_extractor(model, backbone_name="efficientnetv2-s", target_layer="top_activation"):
    backbone = model.get_layer(backbone_name)
    try:
        internal_layer = backbone.get_layer(target_layer)
    except ValueError:
        print(f"Layer {target_layer} not found in backbone {backbone_name}. Available layers:{[layer for layer in backbone.layers[-5:]]}")
        raise

    extractor = tf.keras.Model(inputs=backbone.input, outputs=internal_layer.output)
    return extractor

def make_saliency_map(img_tensor, model):
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)

        output = model(img_tensor, training=False)
        pred = tf.argmax(output[0])

        top_class_score = output[:, pred]
    
    grads = tape.gradient(top_class_score, img_tensor)

    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
    saliency_map = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-8)

    return saliency_map, int(pred.numpy())

def make_gradcam_manual(img_array, model, feature_extractor):
    with tf.GradientTape() as tape:
        # 여기서 정확도를 올리려면 backbone의 top_activation 출력을 사용해야한다.
        conv_output = feature_extractor(img_array, training=False)
        tape.watch(conv_output)

        x = model.get_layer('global_average_pooling2d')(conv_output)
        output = model.get_layer('dense')(x)
        pred = tf.argmax(output[0])
        top_class_channel = output[:, pred]
    
    grads = tape.gradient(top_class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy(), int(pred.numpy())

def make_heatmap(heatmap, input_tensor, path, pred_class, map_type):
    img = input_tensor[0].numpy().astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap_jet = plt.cm.jet(heatmap_resized)[:, :, :3]
    heatmap_jet = (heatmap_jet * 255).astype(np.uint8)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_jet, 0.4, 0)
    relative_path = re.search(r'/[0-9]+_data.*', path).group(0)[1:]

    fig, ax = plt.subplots(1, 3, figsize=(15,5))

    ax[0].imshow(img)
    ax[0].set_title(relative_path)
    ax[0].axis('off')

    ax[1].imshow(heatmap_resized, cmap='jet')
    ax[1].set_title(f'pred_class: {pred_class}')
    ax[1].axis('off')
    
    ax[2].imshow(superimposed_img)
    ax[2].set_title('Superimposed Image')
    ax[2].axis('off')

    image_path = os.path.join(
        RESULTS_DIR, 
        *MODEL_PATH.split('/')[-4:-2], 
        os.path.basename(os.path.splitext(MODEL_PATH)[0]),
        map_type, 
        relative_path
        )

    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    plt.savefig(image_path)
    plt.close(fig)


if __name__ == '__main__':
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    config['batch_size'] = 1

    dataset = get_test_dataset(config=config)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    feature_extractor = get_feature_extractor(model, backbone_name='efficientnetv2-s', target_layer='top_activation')

    for batch in tqdm(dataset):
        images, labels, paths = batch
        
        saliency_map, s_pred_class = make_saliency_map(images, model)
        grad_cam, g_pred_class = make_gradcam_manual(images, model, feature_extractor)
        
        make_heatmap(saliency_map, images, paths.numpy()[0].decode('utf-8'), s_pred_class, map_type='saliency_map')
        make_heatmap(grad_cam, images, paths.numpy()[0].decode('utf-8'), g_pred_class, map_type='grad_cam')