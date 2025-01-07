import csv
import sys
import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableView, QPushButton, QHBoxLayout, QFileDialog, \
    QLabel, QMessageBox, QProgressBar
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QColor
from PIL import Image
import os
import cv2
from PySide6.QtWidgets import QFrame
from sklearn.cluster import KMeans
from view.ModalWindow import ImageInfoDialog  # Предположим, что этот модуль существует


class MyApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def show_error_message(self, title, text):
        message = QMessageBox()
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle(title)
        message.setText(text)
        message.exec()

    def extract_colors(self, image_path, num_colors=3):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)
        boxes = results.xyxy[0][:, :4].cpu().numpy()
        classes = results.xyxy[0][:, 5].cpu().numpy()

        car_boxes = boxes[classes == 2]
        car_boxes1 = boxes[classes == 3]
        car_boxes2 = boxes[classes == 7]

        car_pixels = []

        try:
            # Извлечение пикселей из найденных машин
            for box in np.concatenate([car_boxes, car_boxes1, car_boxes2]):
                x, y, x2, y2 = box.astype(int)
                car_pixels.extend(image_rgb[y:y2, x:x2])

            car_pixels = np.array(car_pixels).reshape((-1, 3))
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(car_pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            pixels = image_rgb.reshape((-1, 3))
            kmeans = KMeans(n_clusters=num_colors)
            kmeans.fit(pixels)
            dominant_colors = kmeans.cluster_centers_.astype(int)

        return dominant_colors

    def find_car(self, input_dir, output_cars='output.csv'):
        cars = ['car', 'truck', 'bus']
        files = os.listdir(input_dir)
        num_files = len(files)
        imgs = [cv2.imread(os.path.join(input_dir, file_name)) for file_name in files]

        # Проверка доступности GPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)
        results = model(imgs)

        output_folder = 'Datasets/images'
        os.makedirs(output_folder, exist_ok=True)

        try:
            with open(output_cars, 'w', newline='') as f:
                writer = csv.writer(f)
                for i, file_name in enumerate(files):
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

                    # Обновление прогресс-бара
                    self.progress_bar.setValue(int((i + 1) / num_files * 100))

            self.progress_bar.setValue(100)
        except Exception as e:
            message = QMessageBox()
            message.setIcon(QMessageBox.Icon.Warning)
            message.setWindowTitle("Ошибка")
            message.setText(f"Ошибка при обработке изображений: {str(e)}")
            message.exec()
        finally:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)

    def detectButtonClicked(self):
        # Получаем выбранную строку из таблицы
        selected_index = self.table_view.selectionModel().currentIndex()
        if selected_index.isValid():
            # Извлекаем путь к изображению из скрытого столбца (например, 3-го)
            file_name = selected_index.siblingAtColumn(3).data(Qt.ItemDataRole.DisplayRole)

            # Проверяем, что файл существует
            image_path = os.path.join(self.current_folder, file_name)  # self.current_folder хранит путь к текущей папке
            if os.path.exists(image_path):
                # Извлекаем доминирующие цвета
                dominant_colors = self.extract_colors(image_path, num_colors=3)

                # Обновляем интерфейс с информацией о цветах
                print("Dominant Colors:")
                for color in dominant_colors:
                    print(f"RGB: {color}")

                message = f"{file_name}\n" + "Доминирующий цвет: " + str(dominant_colors)
                self.result_label.setText(f"Выбранная запись: {message}")

                # Отображение квадрата с цветом
                for color in dominant_colors:
                    color = [min(max(c, 0), 255) for c in color]  # Ограничиваем значения в диапазоне 0-255
                    color_string = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                    print(color_string)
                    pixmap = QPixmap(50, 50)
                    pixmap.fill(QColor(color_string))
                    self.color_square_label.setPixmap(pixmap)

                # Показываем модальное окно с деталями изображения
                image_info_dialog = ImageInfoDialog(image_path, dominant_colors, file_name)
                image_info_dialog.exec()
            else:
                self.show_error_message("Ошибка", f"Файл {file_name} не найден.")
        else:
            self.show_error_message("Ошибка", "Вы не выбрали запись.")



    def init_ui(self):
        # Стиль для кнопок
        button_sheet = """
            QPushButton {
                background-color: #6C63FF;
                border: 2px solid #4e4dff;
                border-radius: 12px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
                transition: background-color 0.3s ease;
            }

            QPushButton:hover {
                background-color: #4e4dff;
                color: white;
            }

            QPushButton:pressed {
                background-color: #3e3dff;
            }

            QPushButton:disabled {
                background-color: #d3d3d3;
                color: #a1a1a1;
            }
        """

        info_style = """
            QLabel {
                font-size: 18px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #333333;
            }

            QFrame {
                border: 1px solid #dcdcdc;
                border-radius: 8px;
                padding: 15px;
                background-color: #f4f4f9;
            }

            .color-square {
                height: 50px;
                width: 50px;
                border-radius: 8px;
                margin-right: 10px;
            }

            .dominant-colors {
                display: flex;
                align-items: center;
                flex-wrap: wrap;
            }
        """

        # Стиль для таблицы
        table_style = """
            QTableView {
                background-color: #ffffff;
                border: 1px solid #dcdcdc;
                border-radius: 10px;
                font-size: 16px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 10px;
                alternate-background-color: #f6f6f6;
            }

            QTableView::item {
                padding: 10px;
                border-radius: 8px;
            }

            QTableView::item:selected {
                background-color: #6C63FF;
                color: white;
                font-weight: bold;
                border-radius: 8px;
            }

            QTableView::horizontalHeader {
                background-color: #6C63FF;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 8px;
                height: 40px;
            }

            QTableView::horizontalHeader::section {
                padding-left: 15px;
                padding-right: 15px;
                border: none;
            }

            QTableView::verticalHeader {
                background-color: #6C63FF;
                color: white;
                font-weight: bold;
                border: none;
            }

            QTableView::verticalHeader::section {
                padding-left: 15px;
                padding-right: 15px;
                border: none;
            }

            QTableView::indicator:unchecked {
                background-color: #6C63FF;
            }

            QTableView::indicator:checked {
                background-color: #4e4dff;
            }

            /* Эффект при наведении */
            QTableView::item:hover {
                background-color: #f0f0f5;
            }
        """

        self.model = QStandardItemModel(0, 3)
        self.model.setHorizontalHeaderLabels(["Фото", "Разрешение", "Вес"])

        # Инициализация виджетов
        self.result_label = QLabel("Выбранная запись:")
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.setStyleSheet(table_style)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.info_frame = QFrame()
        self.info_frame.setStyleSheet(info_style)

        self.info_label = QLabel("Информация о файле:")
        self.info_frame_layout = QVBoxLayout()
        self.info_frame_layout.addWidget(self.info_label)

        # Кнопки
        btn_view_result = QPushButton('Выбрать директорию', self)
        btn_exit = QPushButton('Выход', self)
        detect_button = QPushButton("Просмотреть")
        find_button = QPushButton("Find Car")

        # Применяем стиль ко всем кнопкам
        for button in [btn_view_result, btn_exit, detect_button, find_button]:
            button.setStyleSheet(button_sheet)

        btn_view_result.clicked.connect(self.view_result)
        btn_exit.clicked.connect(self.exit_app)
        detect_button.clicked.connect(self.detectButtonClicked)
        find_button.clicked.connect(self.find_car)

        btn_collapse = QPushButton('-')
        btn_expand = QPushButton('+')

        btn_collapse.clicked.connect(lambda: self.table_view.hide())
        btn_expand.clicked.connect(lambda: self.table_view.show())

        # Компоновка кнопок
        button_layout = QHBoxLayout()
        button_layout.addWidget(btn_view_result)
        button_layout.addWidget(detect_button)
        button_layout.addWidget(btn_exit)



        # Основной лейаут
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_view)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.result_label)

        self.color_square_label = QLabel()
        main_layout.addWidget(self.color_square_label)
        main_layout.addWidget(self.progress_bar)

        self.table_view.setFixedSize(1000, 600)  # Фиксированная ширина и высота таблицы
        self.table_view.setColumnWidth(0, 200)  # Фиксированная ширина для первого столбца
        self.table_view.setColumnWidth(1, 300)  # Фиксированная ширина для второго столбца
        self.table_view.setColumnWidth(2, 200)  # Фиксированная ширина для третьего столбца
        self.table_view.setRowHeight(0, 100)  # Фиксированная высота для первой строки

        self.setLayout(main_layout)
        self.setFixedSize(1050, 800)
        self.setWindowTitle('WipeMyTearsCV')
        self.setStyleSheet("background-color: #f9f9f9;")  # фон для всего приложения
        self.show()

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

    def view_result(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Выберите папку с изображениями')
        if folder_path:
            self.current_folder = folder_path
            self.model.removeRows(0, self.model.rowCount())
            self.find_car(folder_path)

    def exit_app(self):
        sys.exit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApplication()
    sys.exit(app.exec())