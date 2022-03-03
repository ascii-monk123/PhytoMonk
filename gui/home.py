from PyQt5.QtWidgets import QMainWindow, QApplication,QPushButton, QToolButton, QLabel, QRadioButton, QLineEdit, QFileDialog, QMessageBox
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import sys

#main ui window
class UI(QMainWindow):
    def __init__(self):
        super(UI, self).__init__()

        #load the ui file
        uic.loadUi("home.ui", self)
        #set window title
        self.setWindowTitle("PhytoMonk GUI ⛑")


        #define the widgets
        self.UIComponents()

        self.showMaximized()
    
    def UIComponents(self):
        #select the ui components
        self.push_button = self.findChild(QPushButton, "submit")
        self.tool_button = self.findChild(QToolButton, "select_path")
        self.label = self.findChild(QLabel, "label")
        self.radio1 = self.findChild(QRadioButton, "mildew_select")
        self.radio2 = self.findChild(QRadioButton, "rust_select")
        self.info_line = self.findChild(QLineEdit, "path")
        self.original_image = self.findChild(QLabel, "image_original")

        #select the path selection button
        self.tool_button.clicked.connect(self.imageSelection)

    #method to save image selection
    def imageSelection(self):
        #save imagepath
        image_path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Image Files (*.jpg *.png *.tiff *.jpeg *.JPEG)")
        if len(image_path) == 0 or image_path is None:
            return
        
        if not image_path.lower().endswith(('.png', '.jpeg', '.jpg', 'tiff')):
            self.showErrorMessage("Invalid file type selected")
            return

        #set filepath text
        self.info_line.setText(image_path)
        #set image
        self.pixmap = QPixmap(image_path)
        self.original_image.setPixmap(self.pixmap)

    def showErrorMessage(self, text:str):
        err = QMessageBox()
        #set error message
        err.setText(text)
        #set title
        err.setWindowTitle("Error")
        #set buttons
        err.setStandardButtons(QMessageBox.Ok)

        retval = err.exec_()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    ui = UI()
    sys.exit(App.exec())


