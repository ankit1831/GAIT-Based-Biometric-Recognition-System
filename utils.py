import os
from PIL import Image
import numpy as np

def load_silhouette_folder(folder_path):
    images = []

    files = sorted(os.listdir(folder_path))[:30]  # limit for speed

    for file in files:
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(folder_path, file)

            img = Image.open(path).convert("L")
            img = img.resize((96, 96))

            img = np.array(img) / 255.0
            images.append(img)

    return np.array(images)