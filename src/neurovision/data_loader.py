import os
import cv2
import numpy as np
from collections import Counter
import kagglehub


def download_dataset():
    kagglehub.login()

    path = kagglehub.dataset_download("anasdjamaoui/data-mri")

    print("Path to dataset files:", path)
    return path


def load_images_from_folder_resize(folder):
    images = []
    labels = []
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(valid_extensions):
                continue

            image_path = os.path.join(class_path, filename)

            img = cv2.imread(image_path, cv2.IMREAD_COLOR)

            if img is not None:
                img = cv2.resize(img, (224, 224))
                images.append(img)
                labels.append(class_name)

        print(f"{class_name}: {len(images)} images loaded so far")

    return labels, images


def load_data(path):
    DATA_DIR = os.path.join(path, 'Data')

    labels, images = load_images_from_folder_resize(DATA_DIR)

    print("Total loaded:", len(images))
    print("Labels found:", set(labels))
    return labels, images


def to_arrays(images, labels):
    x = np.array(images)
    y = np.array(labels)
    return x, y


def get_class_counts(y):
    #repartition des images
    class_counts = Counter(y)
    return class_counts
