import sys

from PyQt6.QtGui import QImage, QPixmap, QStandardItem
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar, QLabel
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import os
import cv2
import torch
import csv
from PIL import Image

class Worker(QThread):
    progress_update = pyqtSignal(int)

    def __init__(self, input_dir, output_cars):
        super().__init__()
        self.input_dir = input_dir
        self.output_cars = output_cars

    def run(self):
        self.find_car()

    def find_car(self):
        cars, imgs = ['car', 'truck', 'bus'], []
        for file_name in os.listdir(self.input_dir):
            imgs.append(cv2.imread(os.path.join(self.input_dir, file_name)))

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        results = model(imgs)
        output_folder = 'Datasets/images'
        os.makedirs(output_folder, exist_ok=True)

        with open(self.output_cars, 'w', newline='') as f:
            writer = csv.writer(f)
            for i, file_name in enumerate(os.listdir(self.input_dir)):
                res = [n in results.pandas().xyxy[i]['name'].unique() for n in cars]
                has_car = bool(sum(res))
                writer.writerow([file_name, has_car])

                if has_car:
                    output_path = os.path.join(output_folder, file_name)
                    cv2.imwrite(output_path, imgs[i])
                    print(f"Фото с машиной сохранено: {output_path}")
                    image_path = os.path.join(self.input_dir, file_name)
                    image = Image.open(image_path)
                    width, height = image.size
                    size_in_bytes = os.path.getsize(image_path)
                    size_in_mbytes = float(size_in_bytes / 1048576)
                    self.update_table(file_name, width, height, size_in_mbytes, self.input_dir)

                # Обновление прогресса
                progress_percentage = int((i + 1) / len(os.listdir(self.input_dir)) * 100)
                self.progress_update.emit(progress_percentage)

    def update_table(self, file_name, width, height, size_in_mbytes, image_path):
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


class YourClassName(QWidget):
    def __init__(self):
        super().__init__()

        # Создаем QProgressBar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        # Создаем кнопку для запуска процесса
        start_button = QPushButton('Запустить процесс', self)
        start_button.clicked.connect(self.start_process)

        # Располагаем виджеты на форме
        layout = QVBoxLayout(self)
        layout.addWidget(self.progress_bar)
        layout.addWidget(start_button)

        # Инициализируем фоновый поток
        self.worker_thread = QThread()
        self.worker = Worker(input_dir=f'../Datasets/datatest/', output_cars='output.csv')
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(self.update_progress)

    def start_process(self):
        # Запускаем фоновый поток
        self.worker_thread.start()

    def update_progress(self, value):
        # Обновляем значение прогресса
        self.progress_bar.setValue(value)

        # Проверяем, завершился ли процесс
        if value == 100:
            self.worker_thread.quit()
            self.worker_thread.wait()
            print("Процесс завершен.")

if __name__ == "__main__":
    app = QApplication([])
    your_instance = YourClassName()
    your_instance.show()
    sys.exit(app.exec())