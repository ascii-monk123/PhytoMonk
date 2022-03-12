from PyQt5.QtWidgets import QWidget, QApplication,QPushButton, QToolButton, QLabel, QRadioButton, QLineEdit, QFileDialog, QMessageBox
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
import sys
from connect import get_results
import cv2

#convert numpy array ti qpix
def np_to_qpix(image):
    w, h, ch = image.shape
    q_image = QImage(image.data, h, w, 3*h, QImage.Format_RGB888)
    q_pix = QPixmap(q_image)
    return q_pix

#main ui window
class Home(QWidget):
    def __init__(self):
        super().__init__()

        #load the ui file
        uic.loadUi("home.ui", self)
        #set window title
        self.setWindowTitle("PhytoMonk GUI ⛑")


        #define the widgets
        self.UIComponents()
        self.setFixedSize(1300, 730)
        self.show()

    def UIComponents(self):
        #select the ui components
        self.push_button = self.findChild(QPushButton, "submit")
        self.tool_button = self.findChild(QToolButton, "select_path")
        self.label = self.findChild(QLabel, "label")
        self.radio1 = self.findChild(QRadioButton, "mildew_select")
        self.radio2 = self.findChild(QRadioButton, "rust_select")
        self.info_line = self.findChild(QLineEdit, "path")
        self.original_image = self.findChild(QLabel, "image_original")
        self.radio_bindings = ['mildew', 'rust']
        #path selection handler
        self.tool_button.clicked.connect(self.imageSelection)
        #submit button handler
        self.push_button.clicked.connect(self.send_image)

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

    #method to submit data to the server
    def send_image(self):
        #check if radio buttons are enabled
        if not (self.radio1.isChecked() or self.radio2.isChecked()):
            self.showErrorMessage("Please Select a disease type")
            return

        #check if image path is specified or not
        if not(self.info_line.text() and len(self.info_line.text()) > 0):
            self.showErrorMessage("Path not specified")
            return
        
        disease = ''
        if self.radio1.isChecked():
            disease = self.radio_bindings[0]
        
        elif self.radio2.isChecked():
            disease = self.radio_bindings[1]
        
        self.results(disease)

    #show result window
    def results(self, disease):
        self.window2 = QtWidgets.QMainWindow()
        self.ui = Results(disease, self.info_line.text())
        self.ui.setupUi(self.window2)
        self.window2().show()

    def showErrorMessage(self, text:str):
        err = QMessageBox()
        #set error message
        err.setText(text)
        #set title
        err.setWindowTitle("Error")
        #set buttons
        err.setStandardButtons(QMessageBox.Ok)

        retval = err.exec_()
    
#place to show the results
class Results(QWidget):
    def __init__(self, disease_type, image_path):
        super(Results, self).__init__()
        #load the ui file
        uic.loadUi("result.ui", self)
        #set window title
        self.setWindowTitle("PhytoMonk GUI ⛑")
        self.disease = disease_type
        self.image_path = image_path
        self.titl1 = ''
        self.titl2 = ''
        if disease_type == "mildew":
            self.titl1 = 'k-means a*b* seg'
            self.titl2 = 'kmeans  rgb seg'
        
        elif disease_type == "rust":
            self.titl1 = 'a* threshold'
            self.title2 = 'h threshold'

        #define the widgets
        self.UIComponents()
        self.setMaximumSize(1300, 730)
        self.show()
    
    def UIComponents(self):
        #select the ui components
        self.label = self.findChild(QLabel, "disease_type")
        self.result1 = self.findChild(QLabel, "result1")
        self.result2 = self.findChild(QLabel, "result2")
        self.severity1 = self.findChild(QLabel, "severity1")
        self.severity2 = self.findChild(QLabel, "severity2")
        self.type1 = self.findChild(QLabel, "type1")
        self.type2 = self.findChild(QLabel, "type2")
        self.label.setText(self.disease)
        self.type1.setText(self.titl1)
        self.type2.setText(self.titl2)
        #get detection results
        res1, res2, quant1, quant2 = self.getres(self.disease, self.image_path, 'http://127.0.0.1:8000/detect/')
        quant1 = round(quant1, 4)
        quant2 = round(quant2, 4)
        if res1 is None or res2 is None:
            self.result1.setText("Error!")
            self.result2.setText("Error!")
            return
    
        self.severity1.setText(str(quant1))
        self.severity2.setText(str(quant2))

        self.set_images(res1, res2)


    #set result images on screes
    def set_images(self, image1, image2):
        qpix1 = np_to_qpix(image1)
        qpix2 = np_to_qpix(image2)
        #set pixmaps
        self.result2.setPixmap(qpix2)
        self.result1.setPixmap(qpix1)




     #get results from server
    def getres(self, disease_type, image_path, server_path):
       
        try:
            cache1, cache2, quant1, quant2 = get_results(server_path, image_path, disease_type)
            
            return (cache1, cache2, quant1, quant2)
        except:
            return None, None

        




if __name__ == "__main__":
    App = QApplication(sys.argv)
    ui = Home()
    sys.exit(App.exec())


