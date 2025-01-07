import sys

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTableView, QProgressBar, QFileDialog
from PyQt6.QtCore import Qt
from view.TableManager import TableManager
from view.ImageProcessor import ImageProcessor

class MyApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.image_processor = ImageProcessor()  # Инициализация процессора
        self.table_manager = TableManager()  # Инициализация менеджера таблицы
        self.init_ui()

    def init_ui(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.table_view = QTableView()
        self.table_manager.setup_table(self.table_view)

        self.btn_view_result = QPushButton('Выбрать директорию')
        self.btn_exit = QPushButton('Выход')
        self.detect_button = QPushButton("Просмотреть")
        self.find_button = QPushButton("Найти машины")

        self.btn_view_result.clicked.connect(self.view_result)
        self.detect_button.clicked.connect(self.detect_button_clicked)
        self.btn_exit.clicked.connect(self.exit_app)

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.btn_view_result)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.btn_exit)
        self.setLayout(layout)
        self.setWindowTitle('WipeMyTearsCV')
        self.show()

    def view_result(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Выберите папку с изображениями')
        if folder_path:
            self.image_processor.find_car(
                folder_path,
                self.table_manager.update_table,
                self.progress_bar
            )

    def detect_button_clicked(self):
        selected_row = self.table_manager.get_selected_row(self.table_view)
        if selected_row:
            file_name, image_path = selected_row
            dominant_colors = self.image_processor.extract_colors(image_path)
            self.table_manager.show_colors(dominant_colors, file_name)

    def exit_app(self):
        sys.exit()
