# main.py 수정 버전
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QIcon
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

from kpi_page import KPIPage
from stats_page import StatsPage
from widgets import *
from modules import *
from ml_page import MLPage
os.environ["QT_FONT_DPI"] = "96"

widgets = None
GLOBAL_WAFER_GROUPS = {
    "Main": (2901, 2943),
    "Over": (3101, 3143),
    "Low":  (3301, 3343)
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # -----------------------------
        # UI LOAD
        # -----------------------------
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # -----------------------------
        # GLOBAL FILTER STATE
        # -----------------------------
        self.current_wafer_filter = "ALL"
        widgets.comboWaferFilter.addItems([
            "전체", "2900 (Main)", "3100 (Over)", "3300 (Low)"
        ])
        widgets.comboWaferFilter.currentTextChanged.connect(self.on_wafer_filter_changed)

        # -----------------------------
        # TITLE BAR / UI SETTINGS
        # -----------------------------
        Settings.ENABLE_CUSTOM_TITLE_BAR = True
        self.setWindowTitle("TCP 9600")
        widgets.titleRightInfo.setText("TCP 9600.")
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))
        UIFunctions.uiDefinitions(self)

        # -----------------------------
        # LOAD PAGES
        # -----------------------------
        self.kpi_page = KPIPage()  # PySide6 QWidget 기반
        self.stats_page = StatsPage(GLOBAL_WAFER_GROUPS)
        self.ml_page = MLPage(get_group_cb=lambda: self.current_wafer_filter)



        widgets.stackedWidget.addWidget(self.kpi_page)
        widgets.stackedWidget.addWidget(self.stats_page)
        widgets.stackedWidget.addWidget(self.ml_page)
        widgets.stackedWidget.setCurrentWidget(self.kpi_page)

        # -----------------------------
        # BUTTONS
        # -----------------------------
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)

        # -----------------------------
        # THEME
        # -----------------------------
        UIFunctions.theme(self, "themes/py_dracula_light.qss", True)
        AppFunctions.setThemeHack(self)

        # -----------------------------
        # APPLY INITIAL FILTER
        # -----------------------------
        self.stats_page.set_wafer_filter(self.current_wafer_filter)
        self.show()
        widgets.btn_home.click()

    # -----------------------------
    # WAFFER FILTER HANDLER
    # -----------------------------
    def on_wafer_filter_changed(self, text):
        if "2900" in text:
            self.current_wafer_filter = "Main"
        elif "3100" in text:
            self.current_wafer_filter = "Over"
        elif "3300" in text:
            self.current_wafer_filter = "Low"
        else:
            self.current_wafer_filter = "ALL"

        if hasattr(self, "stats_page"):
            self.stats_page.set_wafer_filter(self.current_wafer_filter)

    # -----------------------------
    # UI LOADER
    # -----------------------------
    def load_ui(self, ui_file):
        loader = QUiLoader()
        file = QFile(ui_file)
        file.open(QFile.ReadOnly)
        widget = loader.load(file, self)
        file.close()
        return widget

    # -----------------------------
    # BUTTON CLICK
    # -----------------------------
    def buttonClick(self):
        btn = self.sender()
        btnName = btn.objectName()

        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(self.kpi_page)
        elif btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(self.stats_page)
        elif btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(self.ml_page)

        UIFunctions.resetStyle(self, btnName)
        btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))
        print(f'Button "{btnName}" pressed!')

    # -----------------------------
    # RESIZE EVENT
    # -----------------------------
    def resizeEvent(self, event):
        UIFunctions.resize_grips(self)

    # -----------------------------
    # MOUSE EVENT
    # -----------------------------
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec())
