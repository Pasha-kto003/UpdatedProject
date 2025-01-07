import csv
import sys
import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableView, QPushButton, QHBoxLayout, QFileDialog, \
    QLabel, QDialog
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QColor
from PIL import Image
import os
import cv2
from sklearn.cluster import KMeans

from view.ModalWindow import ImageInfoDialog


class MyApplication(QWidget):
    def __init__(self):
        super().__init__()
        style_sheet = """
            QTableView {
                background-color: #FCBABA;
                border: 2px solid red;
                border-radius: 10px;
                gridline-color: #fd0000;
                
            }

            QTableView::item {
                padding: 5px;
            }

            QTableView::item:selected {
                background-color: #a6a6a6;
                color: #ffffff;
            }
        """
        self.init_ui()
        self.table_view.setStyleSheet(style_sheet)

    def extract_colors(self, image_path, num_colors=2):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)
        boxes = results.xyxy[0][:, :4].cpu().numpy()
        classes = results.xyxy[0][:, 5].cpu().numpy()
        car_boxes = boxes[classes == 2]
        car_pixels = []
        for box in car_boxes:
            x, y, x2, y2 = box.astype(int)
            car_pixels.extend(image_rgb[y:y2, x:x2])
        car_pixels = np.array(car_pixels).reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(car_pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)

        return dominant_colors

    def find_car(self, input_dir, output_cars='output.csv'):
        cars, imgs = ['car', 'truck', 'bus'], []
        for file_name in os.listdir(input_dir):
            imgs.append(cv2.imread(os.path.join(input_dir, file_name)))
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        results = model(imgs)
        output_folder = 'Datasets/images'
        os.makedirs(output_folder, exist_ok=True)
        with open(output_cars, 'w', newline='') as f:
            writer = csv.writer(f)
            for i, file_name in enumerate(os.listdir(input_dir)):
                res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]
                has_car = bool(sum(res))
                writer.writerow([file_name, has_car])
                if has_car:
                    output_path = os.path.join(output_folder, file_name)
                    cv2.imwrite(output_path, imgs[i])
                    print(f"Фото с машиной сохранено: {output_path}")
                    image_path = os.path.join(input_dir, file_name)
                    image = Image.open(image_path)
                    width, height = image.size
                    size_in_bytes = os.path.getsize(image_path)
                    size_in_mbytes = float(size_in_bytes / 1048576)
                    self.update_table(file_name, width, height, size_in_mbytes, input_dir)

    def findcar_onimage(self):
        self.find_car('datatest')

    def detectButtonClicked(self):
        selected_index = self.table_view.selectionModel().currentIndex()
        if selected_index.isValid():
            selected_data = selected_index.siblingAtColumn(3).data(Qt.ItemDataRole.DisplayRole)
            image_path = f'../Datasets/datatest/{selected_data}'
            dominant_colors = self.extract_colors(image_path, num_colors=1)
            print("Dominant Colors:")
            for color in dominant_colors:
                print(f"RGB: {color}")
            message = f"{selected_data}\n" + "Доминирующий цвет: " + str(dominant_colors)
            self.result_label.setText(f"Выбранная запись: {message}")
            color = dominant_colors[0]
            color = [min(max(c, 0), 255) for c in color]
            color_string = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            print(color_string)

            pixmap = QPixmap(50, 50)  # Размер квадрата (50x50)
            pixmap.fill(QColor(color_string))
            self.color_square_label.setPixmap(pixmap)

            image_info_dialog = ImageInfoDialog(image_path, dominant_colors, selected_data)
            image_info_dialog.exec()
            image_info_dialog.open()
        else:
            self.result_label.setText("Выберите строку в таблице")

    def init_ui(self):
        button_sheet = """
            QPushButton {
                background-color: #FCBABA;
                border: 2px solid red;
                border-radius: 10px;
                padding: 5px 10px;
            }

            QPushButton:hover {
                background-color: #FAD9D9;
            }

            QPushButton:pressed {
                background-color: #F79E9E;
            }
        """
        self.model = QStandardItemModel(0, 3)
        self.model.setHorizontalHeaderLabels(["Фото", "Разрешение", "Вес"])
        self.result_label = QLabel("Выбранная запись:")
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        btn_view_result = QPushButton('Выбрать директорию', self)
        # btn_detection = QPushButton('Просмотреть', self)
        btn_exit = QPushButton('Выход', self)
        detect_button = QPushButton("Просмотреть)")
        find_button = QPushButton("Find Car")

        detect_button.setStyleSheet(button_sheet)
        find_button.setStyleSheet(button_sheet)
        btn_exit.setStyleSheet(button_sheet)
        btn_view_result.setStyleSheet(button_sheet)

        btn_view_result.clicked.connect(self.view_result)
        # btn_detection.clicked.connect(self.detect)
        btn_exit.clicked.connect(self.exit_app)
        detect_button.clicked.connect(self.detectButtonClicked)
        find_button.clicked.connect(self.findcar_onimage)
        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_view_result)
        # button_layout.addWidget(btn_detection)
        button_layout.addWidget(detect_button)
        button_layout.addWidget(btn_exit)
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_view)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)
        self.color_square_label = QLabel()
        main_layout.addWidget(self.color_square_label)
        style_sheet = """
            main_layout {
                background-color: #FCBABA;
            }
        """
        self.setStyleSheet(style_sheet)
        self.setLayout(main_layout)
        self.setFixedSize(1200, 800)
        self.setWindowTitle('WipeMyTearsCV')
        self.show()

    def view_result(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Выберите папку с изображениями')
        if folder_path:
            self.find_car(folder_path)

    def update_table(self, file_name, width, height, size_in_mbytes, image_path):
        # Добавляем данные в таблицу
        row_position = self.model.rowCount()
        self.model.insertRow(row_position)
        img = image_path + '/' + file_name
        image_item = QImage(img)
        pixmap = QPixmap.fromImage(image_item)
        label = QLabel()
        label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio))
        self.table_view.setIndexWidget(self.model.index(row_position, 0), label)
        self.model.setItem(row_position, 1, QStandardItem(f"{width}x{height}"))
        self.model.setItem(row_position, 2, QStandardItem(f"{round(size_in_mbytes, 1)} Mbytes"))
        self.model.setItem(row_position, 3, QStandardItem(f"{file_name}"))
        self.table_view.setColumnWidth(0, 500)
        self.table_view.setRowHeight(row_position, 300)
        self.table_view.setColumnHidden(3, True)

    def create_bar(self, height, width, color):
        bar = np.zeros((height, width, 3), np.uint8)
        bar[:] = color
        red, green, blue = int(color[2]), int(color[1]), int(color[0])
        return bar, (red, green, blue)

    def calc_metric(self, image, x, y, w, h):
        # image = image[(y - h):y, x:(x + w)]
        data = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        percentages = (np.unique(label, return_counts=True)[1]) / data.shape[0]
        bar, color = self.create_bar(400, 400, center[np.argmax(percentages)])
        img_bar = np.hstack((bar,))
        return img_bar, color

    def exit_app(self):
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApplication()
    sys.exit(app.exec())
