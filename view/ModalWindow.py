import os
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QDialog, QPushButton
from PyQt6.QtGui import QPixmap, QColor
import cv2


class ImageInfoDialog(QDialog):
    def __init__(self, image_path, dominant_colors, file_name, parent=None):
        super().__init__(parent)

        # Настройки окна
        self.setWindowTitle("Информация о картинке")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # Убираем стандартные кнопки
        self.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ccc;")
        self.setMinimumSize(850, 600)

        # Основной лейаут
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Заголовок с кнопками управления
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(0)

        title_label = QLabel("Информация о картинке")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Кнопка сворачивания окна
        minimize_button = QPushButton("–")
        minimize_button.setFixedSize(30, 30)
        minimize_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                color: black;
                font-size: 18px;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #cccccc;
            }
        """)
        minimize_button.clicked.connect(self.showMinimized)

        # Кнопка закрытия окна
        close_button = QPushButton("✖")
        close_button.setFixedSize(30, 30)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                border: none;
                border-radius: 15px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
            QPushButton:pressed {
                background-color: #b71c1c;
            }
        """)
        close_button.clicked.connect(self.close)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(minimize_button)
        header_layout.addWidget(close_button)

        main_layout.addLayout(header_layout)

        # Отображение картинки
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Загрузка модели YOLOv5
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=False)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Обработка картинки и сохранение результата
        new_path = os.path.join(os.getcwd(), 'processed_images')
        os.makedirs(new_path, exist_ok=True)
        results = model(image_rgb)
        output_image = results.render()[0]
        cv2.imwrite(os.path.join(new_path, file_name), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
        img_path = os.path.join(new_path, file_name)

        # Отображение изображения
        pixmap = QPixmap(img_path)
        image_label.setPixmap(pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio))
        main_layout.addWidget(image_label)

        # Текст с доминирующими цветами
        color_info_label = QLabel("Доминирующие цвета:")
        color_info_label.setStyleSheet("font-size: 16px; color: #333; margin: 10px 0;")
        main_layout.addWidget(color_info_label)

        # Лейаут для отображения квадратов доминирующих цветов
        color_layout = QHBoxLayout()
        color_layout.setSpacing(15)
        color_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Отображение каждого доминирующего цвета
        for color in dominant_colors:
            color_square = self.create_color_square(color)
            color_layout.addWidget(color_square)

        main_layout.addLayout(color_layout)
        self.setLayout(main_layout)

    def create_color_square(self, color):
        """Создает квадрат цвета с подписью."""
        # Ограничение значений цвета в диапазоне 0-255
        color = [min(max(c, 0), 255) for c in color]
        color_string = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

        # Создание квадрата
        color_square_label = QLabel()
        pixmap = QPixmap(100, 100)
        pixmap.fill(QColor(color_string))
        color_square_label.setPixmap(pixmap)
        color_square_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Подпись под квадратом
        color_label = QLabel(f"RGB: {color[0]}, {color[1]}, {color[2]}")
        color_label.setStyleSheet("font-size: 12px; color: #555;")
        color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Лейаут для квадрата и подписи
        layout = QVBoxLayout()
        layout.addWidget(color_square_label)
        layout.addWidget(color_label)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container = QDialog()
        container.setLayout(layout)
        return container