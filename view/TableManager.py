import os
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPixmap, QImage
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

class TableManager:
    def __init__(self):
        self.model = QStandardItemModel(0, 3)
        self.model.setHorizontalHeaderLabels(["Фото", "Разрешение", "Вес"])

    def setup_table(self, table_view):
        table_view.setModel(self.model)
        table_view.setColumnWidth(0, 300)
        table_view.setColumnWidth(1, 200)
        table_view.setColumnWidth(2, 100)

    def update_table(self, file_name, width, height, size_in_mbytes, image_path):
        row_position = self.model.rowCount()
        self.model.insertRow(row_position)
        img = image_path + '/' + file_name
        image_item = QImage(img)
        pixmap = QPixmap.fromImage(image_item)
        label = QLabel()
        label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
        ##table_view.setIndexWidget(self.model.index(row_position, 0), label)
        self.model.setItem(row_position, 1, QStandardItem(f"{width}x{height}"))
        self.model.setItem(row_position, 2, QStandardItem(f"{round(size_in_mbytes, 1)} Mbytes"))
        self.model.setItem(row_position, 3, QStandardItem(f"{file_name}"))

    def get_selected_row(self, table_view):
        selected_index = table_view.selectionModel().currentIndex()
        if selected_index.isValid():
            file_name = selected_index.siblingAtColumn(3).data(Qt.ItemDataRole.DisplayRole)
            return file_name, os.path.join(self.current_folder, file_name)
        return None

    def show_colors(self, dominant_colors, file_name):
        for color in dominant_colors:
            print(f"RGB: {color}")