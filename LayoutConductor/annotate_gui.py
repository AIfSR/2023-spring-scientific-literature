from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
    qApp, QFileDialog, QSlider, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QComboBox, QApplication

import sys
import json
from os import listdir

class ImageRanker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_paths = ["start.jpeg"]
        self.image_dict = {}
        self.path = "./outputs/"
        
        # Create widgets
        self.imageLabel2 = QLabel()
        self.imageLabel2.setBackgroundRole(QPalette.Base)
        self.imageLabel2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.nextButton = QPushButton('Next')
        self.submitButton = QPushButton('Submit')
        
        # Create slider and label for ranking images
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 3)
        self.slider.setValue(1)
        self.slider.setTickInterval(2)
        self.slider.setTickPosition(QSlider.TicksBelow)

        self.sliderLabel = QLabel('Score: ' + str(self.slider.value()))

        # Connect the slider's valueChanged signal to a function that updates the label
        self.slider.valueChanged.connect(self.sliderValueChange)
        
        self.nextButton.clicked.connect(self.displayNextImage)
        self.submitButton.clicked.connect(self.submitDict)
        
        # Create a dropdown list for folder names
        self.directoryName = QComboBox(self)
        self.directoryName.addItem("Select Directory")
        self.directoryName.activated[str].connect(self.onActivated)

        # Populate the list with folder names
        for folder_name in listdir(self.path):
            if '_out' in folder_name and 'orig' not in folder_name:
                self.directoryName.addItem(folder_name)
        
        # Create layouts
        self.sliderLayout = QHBoxLayout()
        self.sliderLayout.addWidget(self.slider)
        self.sliderLayout.addWidget(self.sliderLabel)
        self.sliderLayout.addWidget(self.nextButton)
        self.sliderLayout.addWidget(self.submitButton)
        self.sliderLayout.addWidget(self.directoryName)
        
        self.imageLayout = QHBoxLayout()
        self.imageLayout.addWidget(self.imageLabel2)
        
        self.totalLayout = QVBoxLayout()
        self.totalLayout.addLayout(self.imageLayout)
        self.totalLayout.addLayout(self.sliderLayout)
        
        self.widget = QWidget()
        self.widget.setLayout(self.totalLayout)
        self.setCentralWidget(self.widget)

        # Set window properties
        self.setWindowTitle("Image Scorer")
        self.scrollArea = None
        
        self.current_image = 0
        image = QImage(self.image_paths[self.current_image])
        self.imageLabel2.setPixmap(QPixmap.fromImage(image))
        QMessageBox.information(self, "Image Viewer", "Select a folder from the dropdown to start")
        
    def sliderValueChange(self, value):
        self.sliderLabel.setText('Score: ' + str(value))
        self.image_dict[self.curr_fname][self.image_paths[self.current_image]] = self.slider.value()
        
    def onActivated(self, text):
        # Set image paths
        self.curr_fname = text
        self.image_paths = []
        for image_name in listdir(self.path+text+'/'):
            if '.jpg' in image_name or '.png' in image_name:
                self.image_paths.append(text+'/' + image_name)
        # Set main key of the saved image_dict
        self.image_dict[text] = {}
        
        self.current_image = -1
        self.displayNextImage()
        
    def displayNextImage(self):
        self.current_image += 1
        if self.current_image >= len(self.image_paths):
            QMessageBox.information(self, "Image Viewer", "Last image, press submit to submit all scores, then select a new folder from the dropdown.")
            return
        image = QImage(self.path+self.image_paths[self.current_image])
        self.imageLabel2.setPixmap(QPixmap.fromImage(image).scaled(self.imageLabel2.width(), self.imageLabel2.height(), Qt.KeepAspectRatio))
        
        self.slider.setValue(2)
        self.image_dict[self.curr_fname][self.image_paths[self.current_image]] = self.slider.value()
        
    def submitDict(self):
        # Store the dictionary in a json
        with open(self.path+'score_dict.json', 'w') as fp:
            json.dump(self.image_dict, fp)
        self.directoryName.removeItem(self.directoryName.findText(self.curr_fname))
        QMessageBox.information(self, "Image Viewer", "Stored successfully, select new folder from dropdown")

app = QApplication(sys.argv)
imageRanker = ImageRanker()
imageRanker.show()
sys.exit(app.exec_())