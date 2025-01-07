import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import torch

class ImageProcessor:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)

    def extract_colors(self, image_path, num_colors=3):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_.astype(int)

    def find_car(self, input_dir, update_table_callback, progress_bar):
        files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png'))]
        imgs = [cv2.imread(os.path.join(input_dir, file_name)) for file_name in files]

        results = self.model(imgs)
        num_files = len(files)

        for i, file_name in enumerate(files):
            result = results.pandas().xyxy[i]
            if any(result['name'] == 'car'):
                image_path = os.path.join(input_dir, file_name)
                image = cv2.imread(image_path)
                h, w, _ = image.shape
                size_in_bytes = os.path.getsize(image_path)
                size_in_mbytes = size_in_bytes / 1024 / 1024
                update_table_callback(file_name, w, h, size_in_mbytes, input_dir)
            progress_bar.setValue(int((i + 1) / num_files * 100))