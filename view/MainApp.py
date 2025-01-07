import sys
from PyQt6.QtWidgets import QApplication
from view.MyApplication import MyApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApplication()
    sys.exit(app.exec())