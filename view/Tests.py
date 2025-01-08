import csv
import sys
import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableView, QPushButton, QFileDialog, QLabel, QMessageBox, QProgressBar
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QImage, QPixmap, QColor
from PIL import Image
import os
import cv2
from sklearn.cluster import KMeans


class ImageProcessor:
    """Класс для обработки изображений и извлечения доминирующих цветов"""

    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    def extract_colors(self, image_path, num_colors=3):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb)
        boxes = results.xyxy[0][:, :4].cpu().numpy()
        classes = results.xyxy[0][:, 5].cpu().numpy()

        car_boxes = boxes[classes == 2]
        car_boxes1 = boxes[classes == 3]
        car_boxes2 = boxes[classes == 7]

        car_pixels = []

        try:
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


class ColorManager:
    """Класс для обработки изображений и извлечения доминирующих цветов"""

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def detectButtonClicked(self, table_view, result_label):
        """Обработчик нажатия кнопки для извлечения цветов из выбранного изображения"""

        selected_index = table_view.selectionModel().currentIndex()
        if selected_index.isValid():
            file_name = selected_index.siblingAtColumn(3).data(Qt.ItemDataRole.DisplayRole)
            image_path = os.path.join(self.current_folder, file_name)

            # Проверка на существование изображения
            if os.path.exists(image_path):
                # Извлечение доминирующих цветов
                dominant_colors = self.image_processor.extract_colors(image_path, num_colors=3)

                # Отображение результата на интерфейсе
                result_label.setText(f"Выбранная запись: {file_name}\nДоминирующий цвет: {dominant_colors}")

                # Отображение каждого доминирующего цвета в интерфейсе
                for color in dominant_colors:
                    color = [min(max(c, 0), 255) for c in color]  # Ограничение значений цвета
                    color_string = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

                    # Отображение цвета в виде пикселя
                    pixmap = QPixmap(50, 50)
                    pixmap.fill(QColor(color_string))
                    self.color_square_label.setPixmap(pixmap)
            else:
                QMessageBox.warning(None, "Ошибка", f"Файл {file_name} не найден.")


class CarFinderManager:
    """Класс для поиска машин в изображениях"""

    @staticmethod
    def find_car(input_dir, output_cars='output.csv', model=None):
        cars = ['car', 'truck', 'bus']
        files = os.listdir(input_dir)
        num_files = len(files)
        imgs = [cv2.imread(os.path.join(input_dir, file_name)) for file_name in files]

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
            return files, num_files
        except Exception as e:
            print(f"Ошибка при обработке изображений: {str(e)}")
            return [], 0

    def view_result(self, file_handler, table_updater):
        folder_path = QFileDialog.getExistingDirectory(None, 'Выберите папку с изображениями')
        if folder_path:
            self.current_folder = folder_path
            table_updater.model.removeRows(0, table_updater.model.rowCount())
            files, num_files = file_handler.find_car(folder_path)
            for i, file_name in enumerate(files):
                table_updater.update_progress_bar(i, num_files)


class TableUpdater:
    """Класс для обновления таблицы с результатами"""

    def __init__(self, model, progress_bar):
        self.model = model
        self.progress_bar = progress_bar

    def update_table(self, file_name, width, height, size_in_mbytes, image_path):
        row_position = self.model.rowCount()
        self.model.insertRow(row_position)
        img = image_path + '/' + file_name
        image_item = QImage(img)
        pixmap = QPixmap.fromImage(image_item)
        label = QLabel()
        label.setPixmap(pixmap.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio))
        self.model.setIndexWidget(self.model.index(row_position, 0), label)
        self.model.setItem(row_position, 1, QStandardItem(f"{width}x{height}"))
        self.model.setItem(row_position, 2, QStandardItem(f"{round(size_in_mbytes, 1)} Mbytes"))
        self.model.setItem(row_position, 3, QStandardItem(f"{file_name}"))
        self.model.setColumnWidth(0, 500)
        self.model.setRowHeight(row_position, 300)
        self.model.setColumnHidden(3, True)

    def update_progress_bar(self, current, total):
        self.progress_bar.setValue(int((current + 1) / total * 100))


from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableView, QLabel, QProgressBar, QMessageBox


class UIManager:
    def init_ui(self, car_finder_manager, color_manager):
        self.model = QStandardItemModel(0, 3)
        self.model.setHorizontalHeaderLabels(["Фото", "Разрешение", "Вес"])

        self.result_label = QLabel("Выбранная запись:")
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        # Кнопки
        btn_view_result = QPushButton('Выбрать директорию', self)
        btn_exit = QPushButton('Выход', self)
        detect_button = QPushButton("Просмотреть")
        find_button = QPushButton("Find Car")

        # Применение стилей для кнопок
        btn_view_result.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;  /* Зеленый фон */
                color: white;               /* Белый текст */
                padding: 10px 20px;         /* Отступы */
                border: none;               /* Без рамки */
                border-radius: 5px;         /* Скругленные углы */
                font-size: 16px;            /* Размер шрифта */
            }
            QPushButton:hover {
                background-color: #45a049;  /* Цвет при наведении */
            }
        """)
        btn_exit.setStyleSheet("""
            QPushButton {
                background-color: #f44336;  /* Красный фон */
                color: white;               /* Белый текст */
                padding: 10px 20px;         /* Отступы */
                border: none;               /* Без рамки */
                border-radius: 5px;         /* Скругленные углы */
                font-size: 16px;            /* Размер шрифта */
            }
            QPushButton:hover {
                background-color: #da190b;  /* Цвет при наведении */
            }
        """)
        detect_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;  /* Синий фон */
                color: white;               /* Белый текст */
                padding: 10px 20px;         /* Отступы */
                border: none;               /* Без рамки */
                border-radius: 5px;         /* Скругленные углы */
                font-size: 16px;            /* Размер шрифта */
            }
            QPushButton:hover {
                background-color: #0b7dda;  /* Цвет при наведении */
            }
        """)
        find_button.setStyleSheet("""
            QPushButton {
                background-color: #ff9800;  /* Оранжевый фон */
                color: white;               /* Белый текст */
                padding: 10px 20px;         /* Отступы */
                border: none;               /* Без рамки */
                border-radius: 5px;         /* Скругленные углы */
                font-size: 16px;            /* Размер шрифта */
            }
            QPushButton:hover {
                background-color: #e68900;  /* Цвет при наведении */
            }
        """)

        # Таблица: стиль для ячеек
        self.table_view.setStyleSheet("""
            QTableView {
                border: 1px solid #ccc;          /* Рамка таблицы */
                gridline-color: #ddd;            /* Цвет линий сетки */
                font-size: 14px;                  /* Размер шрифта */
                background-color: #f9f9f9;       /* Цвет фона */
            }
            QHeaderView::section {
                background-color: #2196F3;       /* Фон заголовков */
                color: white;                    /* Белый текст */
                font-weight: bold;               /* Жирный шрифт */
                padding: 10px;                   /* Отступы */
            }
            QTableView::item {
                padding: 10px;                   /* Отступы внутри ячеек */
                border-bottom: 1px solid #ddd;   /* Рамка между строками */
            }
            QTableView::item:selected {
                background-color: #2196F3;       /* Цвет выбранных ячеек */
                color: white;                    /* Белый текст в выбранных ячейках */
            }
        """)

        # Размещение элементов на экране
        btn_view_result.clicked.connect(car_finder_manager.view_result)
        btn_exit.clicked.connect(self.exit_app)
        detect_button.clicked.connect(color_manager.detectButtonClicked)
        find_button.clicked.connect(car_finder_manager.view_result)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table_view)
        main_layout.addWidget(btn_view_result)
        main_layout.addWidget(detect_button)
        main_layout.addWidget(self.result_label)
        self.setLayout(main_layout)
        self.show()

    def show_error_message(self, title, text):
        message = QMessageBox()
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle(title)
        message.setText(text)
        message.exec()



class MyApplication(QWidget):
    """Основной класс приложения"""

    def __init__(self):
        super().__init__()

        # Создание объектов
        self.image_processor = ImageProcessor()
        self.car_finder_manager = CarFinderManager()
        self.ui_manager = UIManager()
        self.table_updater = TableUpdater(self.ui_manager.model, self.ui_manager.progress_bar)
        self.color_manager = ColorManager(self.image_processor)

        # Инициализация интерфейса
        self.ui_manager.init_ui(self.car_finder_manager, self.color_manager)
        self.setWindowTitle("Car Finder")
        self.resize(800, 600)
        self.show()

    def init(self):
        pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_app = MyApplication()
    sys.exit(app.exec())
