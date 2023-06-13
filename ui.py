from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QPushButton, QSlider, QVBoxLayout, QWidget, QHBoxLayout, QSizePolicy, QSpacerItem, QLabel, QDoubleSpinBox
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QColor, QPainterPath, QPainterPathStroker

from Painter import PaintWidget

def init_ui(self):
    desktop = QApplication.desktop()
    screen_rect = desktop.screenGeometry()
    self.width, self.height = screen_rect.width(), screen_rect.height()
    self.setFixedSize(int(self.width*0.9), int(self.height*0.9))
    self.train_thread = None
    self.model = None

    main_widget = QWidget()
    layout = QVBoxLayout()

    self.paint_widget = PaintWidget()
    #layout.addWidget(self.paint_widget)
    
    image_file = 'predictions/192.168.42.17-00597-AHNOB3.jpg'
    pixmap = QPixmap(image_file)
    self.image_label = QLabel()
    self.image_label.setPixmap(QPixmap(pixmap))
    imagelayout = QHBoxLayout()
    imagelayout.addWidget(self.paint_widget)
    imagelayout.addWidget(self.image_label)
    layout.addLayout(imagelayout)
    self.prev_button = QPushButton('Prev')
    self.prev_button.clicked.connect(self.prev_image)

    self.next_button = QPushButton('Next')
    self.next_button.clicked.connect(self.next_image)

    self.thickness_slider = QSlider(Qt.Horizontal)
    self.thickness_slider.setMinimum(1)
    self.thickness_slider.setMaximum(100)
    self.thickness_slider.setValue(self.paint_widget.thickness)
    self.thickness_slider.valueChanged.connect(self.set_thickness)
    self.thickness_slider.setFixedWidth(self.width//6)

    self.foreground_button = QPushButton('Foreground')
    self.foreground_button.clicked.connect(self.add_foreground_pixels)
    self.foreground_button.setStyleSheet("background-color: #90ee90; border: 3px solid black; border-radius: 10px")


    self.foreground_button.setFixedWidth(self.width//12)

    self.background_button = QPushButton('Background')
    self.background_button.clicked.connect(self.add_background_pixels)
    #make the button #b92f2f
    self.background_button.setStyleSheet("background-color: #b92f2f; border: 3px solid black; border-radius: 10px")

    self.background_button.setFixedWidth(self.width//12)

    self.resolution_spinbox = QDoubleSpinBox()
    self.resolution_spinbox.setRange(0.01, 1.0)
    self.resolution_spinbox.setSingleStep(0.01)
    self.resolution_spinbox.setValue(self.resolution)
    self.resolution_spinbox.setSuffix("x")
    self.statusBar().addPermanentWidget(self.resolution_spinbox)

    # Connect the QDoubleSpinBox widget to the update_resolution slot
    self.resolution_spinbox.valueChanged.connect(self.update_resolution)


    prev_next_layout = QHBoxLayout()
    prev_next_layout.setContentsMargins(0, 0, 0, 0)
    prev_next_layout.setSpacing(10)  # adjust the spacing between buttons here

    prev_next_layout.addWidget(self.prev_button)
    prev_next_layout.addWidget(self.next_button)
    prev_next_layout.addWidget(self.thickness_slider)
    prev_next_layout.addWidget(self.foreground_button)
    prev_next_layout.addWidget(self.background_button)

    # Add spacer to push buttons to the left
    spacer = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)
    prev_next_layout.addItem(spacer)

    layout.addLayout(prev_next_layout)

    self.clear_button = QPushButton('Clear')
    self.clear_button.clicked.connect(self.clear_painted_pixels)
    layout.addWidget(self.clear_button)

    self.train_button = QPushButton('Train Model')
    self.train_button.clicked.connect(self.train_model)
    layout.addWidget(self.train_button)

    main_widget.setLayout(layout)
    self.setCentralWidget(main_widget)

    self.reset_data_button = QPushButton('Reset Data')
    self.reset_data_button.clicked.connect(self.reset_data)
    layout.addWidget(self.reset_data_button)

    self.prev_button.setFixedSize(self.width // 20, self.height // 30)
    self.next_button.setFixedSize(self.width // 20, self.height // 30)

    self.open_image_button = QPushButton('Open Images')
    self.open_image_button.clicked.connect(self.open_image)
    layout.addWidget(self.open_image_button)

    self.predict_button = QPushButton('Predict')
    self.predict_button.clicked.connect(self.predict)
    layout.addWidget(self.predict_button)

    train_pred = QHBoxLayout()
    train_pred.setContentsMargins(0, 0, 0, 0)
    train_pred.setSpacing(10)
    train_pred.addWidget(self.train_button)
    train_pred.addWidget(self.predict_button)  
    layout.addLayout(train_pred)