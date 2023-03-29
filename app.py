import sys
from PyQt5.QtCore import Qt, QPoint, QRectF, QSize, QThread, pyqtSignal, QSizeF, QPointF
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QColor, QBrush, QPainterPath, QPainterPathStroker
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget, QHBoxLayout, QSpinBox, QMessageBox, QInputDialog
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.color as color
from PIL import Image

from Trainer import TrainModelThread
from Painter import PaintWidget



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Painter')
        self.setGeometry(100, 100, 800, 600)
        self.train_thread = None
        self.model = None

        main_widget = QWidget()
        layout = QVBoxLayout()

        self.paint_widget = PaintWidget()
        layout.addWidget(self.paint_widget)

        self.open_image_button = QPushButton('Open Image')
        self.open_image_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_image_button)

        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setMinimum(1)
        self.thickness_slider.setMaximum(100)
        self.thickness_slider.setValue(self.paint_widget.thickness)
        self.thickness_slider.valueChanged.connect(self.set_thickness)
        layout.addWidget(self.thickness_slider)

        self.foreground_button = QPushButton('Foreground')
        self.foreground_button.clicked.connect(self.add_foreground_pixels)
        layout.addWidget(self.foreground_button)

        self.background_button = QPushButton('Background')
        self.background_button.clicked.connect(self.add_background_pixels)
        layout.addWidget(self.background_button)

        self.current_pixel_label = QLabel()
        layout.addWidget(self.current_pixel_label)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.clear_button = QPushButton('Clear')
        self.clear_button.clicked.connect(self.clear_painted_pixels)
        layout.addWidget(self.clear_button)

        self.train_button = QPushButton('Train Model')
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        # self.predict_button = QPushButton('Predict')
        # self.predict_button.clicked.connect(self.predict)
        # layout.addWidget(self.predict_button)

        self.reset_data_button = QPushButton('Reset Data')
        self.reset_data_button.clicked.connect(self.reset_data)
        layout.addWidget(self.reset_data_button)

        self.prediction_label = QLabel()
        layout.addWidget(self.prediction_label)


    def reset_data(self):
        self.paint_widget.reset_painted_pixels()


    def qimage_to_numpy_array(self, qimage):
        # Get the QImage dimensions
        width, height = qimage.width(), qimage.height()
        print(f"Image dimensions: {width} x {height}")

        # Get the QImage format and bytes per line
        fmt = qimage.format()
        bpl = qimage.bytesPerLine()

        # Get the QImage buffer as a bytes object
        buffer = qimage.constBits().asarray(height * bpl)

        # Check the QImage format and create the corresponding numpy array
        if fmt == QImage.Format_RGB32 or fmt == QImage.Format_ARGB32:
             im = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
             return im[:, :, :3]
        elif fmt == QImage.Format_RGB888:
            return np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 3))
        else:
            raise ValueError(f"Unsupported QImage format: {fmt}")


    def clear_painted_pixels(self):
        self.paint_widget.clear_temp_painted_pixels()


    def open_image(self, predict=False):

        if predict == True:
            file_name = "preds2.png"
            image = QImage(file_name)
            self.image_preds = image.scaled(800, 600)
            self.paint_widget.set_image2(self.image)
            self.paint_widget.update_pixmap2()
            self.image_array = self.qimage_to_numpy_array(qimage=self.image)
            self.image_array = np.reshape(self.image_array, (self.image_array.shape[0]*self.image_array.shape[1], 3))
            self.paint_widget.update()

        else:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)

        if file_name:
            
            image = QImage(file_name)
            self.image = image.scaled(800, 600)
            self.paint_widget.set_image(self.image)
            self.paint_widget.update_pixmap()
            self.image_array = self.qimage_to_numpy_array(qimage=self.image)
            self.image_array = np.reshape(self.image_array, (self.image_array.shape[0]*self.image_array.shape[1], 3))
            self.paint_widget.update()
            if predict == False:
                io.imsave("base.png", self.image_array.reshape(self.image.height(), self.image.width(), 3))

    def set_thickness(self, thickness):
        self.paint_widget.set_thickness(thickness)

    def add_foreground_pixels(self):
        l = self.paint_widget.temp_painted_pixels.shape[0]
        root = int(np.sqrt(l))
        print(self.paint_widget.temp_painted_pixels[:root**2,:].reshape(root,root,3))
        print(self.paint_widget.temp_painted_pixels[:root**2,:].reshape(root,root,3).shape)
        plt.figure()
        plt.imshow(self.paint_widget.temp_painted_pixels[:root**2,:].reshape(root,root,3).astype(np.uint8))
        plt.show()
        
        foreground = np.hstack((self.paint_widget.temp_painted_pixels, np.ones((self.paint_widget.temp_painted_pixels.shape[0], 1))))
        self.paint_widget.painted_pixels = np.vstack((self.paint_widget.painted_pixels, foreground))
        self.paint_widget.clear_temp_painted_pixels()
        print(f"{foreground.shape} pixels added to painted pixels array. {self.paint_widget.painted_pixels.shape}")

    def add_background_pixels(self):
        background = np.hstack((self.paint_widget.temp_painted_pixels, np.zeros((self.paint_widget.temp_painted_pixels.shape[0], 1))))
        self.paint_widget.painted_pixels = np.vstack((self.paint_widget.painted_pixels, background))
        self.paint_widget.clear_temp_painted_pixels()
        print(f"{background.shape} pixels added to painted pixels array. {self.paint_widget.painted_pixels.shape}")

    def update_current_pixel_label(self, pos):
        if self.paint_widget.image is not None:
            x = int(pos.x())
            y = int(pos.y())
            color = QColor(self.paint_widget.image.pixel(x, y))
            self.current_pixel_label.setText(f"Current pixel: ({x}, {y}) R:{color.red()} G:{color.green()} B:{color.blue()}")

    def train_model(self):
        if len(self.paint_widget.painted_pixels) == 0:
            QMessageBox.warning(self, "Warning", "Please paint some pixels before training.")
            return

        #num_rounds, ok = QInputDialog.getInt(self, "Training Parameters", "Number of rounds:", value=100)
        self.train_thread = TrainModelThread(self.paint_widget.painted_pixels, 
                                             num_rounds=100, image_array=self.image_array)
        self.train_thread.training_finished.connect(self.training_finished)
        self.train_thread.start()
    

    
    def training_finished(self, accuracy):
        # Show a message box with the training accuracy
        QMessageBox.information(self, "Training Finished", f"Training finished with accuracy of  {np.round(100*accuracy,2)}%.")

        # Get the prediction from the training thread
        prediction = self.train_thread.prediction

        # Reshape the prediction to the original image size
        prediction = np.reshape(prediction, (self.image.height(), self.image.width()))

        # Scale the prediction values to the range [0, 255]
        prediction = np.round(prediction).astype(np.uint8)*255

        # Convert the grayscale prediction to an RGB image
        img_rgb = color.gray2rgb(prediction)
        img_rgb[:,:,0:2]=0

        # Create a new NumPy array with 4 color channels (RGB and alpha)
        img_rgba = np.zeros((prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)

        # Copy the RGB values from the image into the first 3 channels of the new array
        img_rgba[:, :, :3] = img_rgb

        # Set the alpha channel values to a constant value (here we use 128)
        alpha = np.zeros((prediction.shape[0], prediction.shape[1]), dtype=np.uint8)
        alpha.fill(128)

        # Copy the alpha channel values into the fourth channel of the new array
        img_rgba[:, :, 3] = alpha
        
        io.imsave('preds.png', img_rgba)

        image1 = Image.open("base.png")
        image2 = Image.open("preds.png")
        alpha = np.zeros((np.array(image1).shape[0], np.array(image1).shape[1]), dtype=np.uint8)
        alpha.fill(50)
        mask_arr = (np.array(image2)[:,:,2]/255*50).astype(np.uint8)
        mask = Image.fromarray(mask_arr)

        image1.paste(image2, (0, 0), mask)

        image1.save('preds2.png',"PNG")
        # Save the RGBA image as a PNG file
        self.vis_pred()

    def vis_pred(self):
        self.open_image(predict=True)


    # def predict(self):
    #     if self.train_thread is None:
    #         QMessageBox.warning(self, "Warning", "Please train the model before predicting.")
    #         return
    #     prediction = self.model.predict(self.image_array)
    #     prediction = np.reshape(prediction, (self.image.height(), self.image.width()))
    #     prediction = np.round(prediction).astype(np.uint8)
    #     plt.figure()
    #     plt.imshow(prediction, cmap="Spectral", interpolation="nearest")
    #     plt.show()
    
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
