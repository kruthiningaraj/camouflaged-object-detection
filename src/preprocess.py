import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image_mask_pairs(image_dir, mask_dir, target_size=(256, 256)):
    images, masks = [], []
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, img_file.replace(".jpg", ".png"))
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size)
            mask = cv2.resize(mask, target_size)
            images.append(img / 255.0)
            masks.append(np.expand_dims(mask / 255.0, axis=-1))
    return np.array(images), np.array(masks)

def get_augmentation():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
