import sys
import os
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import numpy as np
import skimage.io as io
import skimage.transform as transform
import skimage.color as color
from PIL import Image
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
import lightgbm as lgb

from Trainer import TrainModelThread
from ui import init_ui

import os
import sys

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.images = []  # list of images
        self.image_arrays = []  # list of image arrays
        self.index = 0  # current index of the displayed image
        self.max_index = None  # max index of the displayed image
        self.displayed_name = None  # name of the displayed image
        self.resolution = 0.2  # resolution of the predictions
        init_ui(self)
        self.foreground_button.setVisible(False)
        self.background_button.setVisible(False)
        self.clear_button.setVisible(False)
        self.train_button.setVisible(False)
        self.reset_data_button.setVisible(False)
        self.prev_button.setVisible(False)
        self.next_button.setVisible(False)
        self.thickness_slider.setVisible(False)
        self.paint_widget.setVisible(False)
        self.image_label.setVisible(False)
        self.resolution_spinbox.setVisible(False)
        self.predict_button.setVisible(False)
        

    def reset_data(self):
        self.paint_widget.reset_painted_pixels()
        self.paint_widget.painted_pixels = np.empty((0, 4))



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
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)

        if file_names:
            self.max_index = len(file_names) - 1
            self.image_names = []  # Save relative image paths in an attribute
            images = []
            for file_name in file_names:
                image_path = os.path.relpath(file_name)  # Get the relative path of the image
                print(f"Loading image from {image_path}")
                image_array = io.imread(file_name)
               
                image_array = np.array(image_array)  
                # Check if the image has an alpha channel
                if image_array.shape[-1] == 4:
                    # Remove the alpha channel
                    image_array = image_array[..., :3]
                #self.image_arrays.append(image_array)
                image_name, ext = os.path.splitext(os.path.basename(file_name))
                print(image_name)
                image_path = f"{image_name}.jpg"
                print(f"Saving image to {image_path}")
                self.image_names.append(f"{image_path}")
                io.imsave(f"Images/{image_path}", image_array, quality=80)
                
                # Convert image to QImage and resize for display
                image = QImage(f"Images/{image_path}")
                image = image.scaled(int(self.width*0.44), int(self.height*0.6))
                images.append(image)
                


            # Update GUI to display images
            self.paint_widget.images = images
            self.paint_widget.set_images(self.index)
            self.paint_widget.update_pixmap(images[0].width(), images[0].height())
            self.displayed_name = self.image_names[self.index]
            # Update other GUI elements
            self.foreground_button.setVisible(True)
            self.background_button.setVisible(True)
            self.clear_button.setVisible(True)
            self.train_button.setVisible(True)
            self.reset_data_button.setVisible(True)
            self.prev_button.setVisible(True)
            self.next_button.setVisible(True)
            self.thickness_slider.setVisible(True)
            self.paint_widget.setVisible(True)
            self.resolution_spinbox.setVisible(True)
            self.predict_button.setVisible(True)




    def set_thickness(self, thickness):
        self.paint_widget.set_thickness(thickness)

    def add_foreground_pixels(self):
        l = self.paint_widget.temp_painted_pixels.shape[0]
        root = int(np.sqrt(l))
        # plt.figure()
        # plt.imshow(self.paint_widget.temp_painted_pixels[:root**2,:].reshape(root,root,3).astype(np.uint8))
        # plt.show()
        
        foreground = np.hstack((self.paint_widget.temp_painted_pixels, np.ones((self.paint_widget.temp_painted_pixels.shape[0], 1))))
        self.paint_widget.painted_pixels = np.vstack((self.paint_widget.painted_pixels, foreground))
        self.paint_widget.clear_temp_painted_pixels()
        print(f"{foreground.shape} pixels added to painted pixels array. {self.paint_widget.painted_pixels.shape}")

    def add_background_pixels(self):
        l = self.paint_widget.temp_painted_pixels.shape[0]
        root = int(np.sqrt(l))
        # plt.figure()
        # plt.imshow(self.paint_widget.temp_painted_pixels[:root**2,:].reshape(root,root,3).astype(np.uint8))
        # plt.show()

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

    def train_model(self, load_model=False):
        if len(self.paint_widget.painted_pixels) == 0:
            QMessageBox.warning(self, "Warning", "Please paint some pixels before training.")
            return

        self.image_array = io.imread(f"Images/{self.image_names[self.index]}")
        
        self.array_width = self.image_array.shape[1]
        self.array_height = self.image_array.shape[0]
        print(self.image_array.shape)

        self.image_array = transform.resize(self.image_array,
                                            (int(self.array_height*self.resolution),
                                              int(self.array_width*self.resolution)))
        
        self.array_width = self.image_array.shape[1]
        self.array_height = self.image_array.shape[0]
        
        self.image_array = np.reshape(self.image_array, (self.array_height*self.array_width, 3))
        
        #num_rounds, ok = QInputDialog.getInt(self, "Training Parameters", "Number of rounds:", value=100)
        self.train_thread = TrainModelThread(self.paint_widget.painted_pixels, 
                                             num_rounds=100, image_array=self.image_array, load_model=load_model)
        self.train_thread.training_finished.connect(self.training_finished)
        self.train_thread.start()
    
    
    def training_finished(self, accuracy):
        # Show a message box with the training accuracy
        #QMessageBox.information("Yeah!")
        # Get the prediction from the training thread
        prediction = self.train_thread.prediction
        # Reshape the prediction to the original image size
        prediction = np.reshape(prediction, (self.array_height, self.array_width))
        dpi = 300
        print(f"Saving prediction to predictions/{self.displayed_name}")
        plt.figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi)

        plt.axis('off')

        plt.imshow(prediction, cmap="Spectral")
        plt.savefig(f"predictions/{self.displayed_name}", bbox_inches='tight', pad_inches=0, dpi=dpi)
        np.save(f"predictions/{self.displayed_name}", prediction)
        preds_path = f"predictions/{self.displayed_name}"
        preds = QImage(preds_path)
        preds = preds.scaled(int(self.width*0.4), int(self.height*0.6))
        pixmap = QPixmap.fromImage(preds)
        pixmap = pixmap.scaled(int(self.width*0.4), int(self.height*0.6))
        self.image_label.setPixmap(pixmap)
        self.image_label.setVisible(True)

    def predict(self):
        print("hey")
        if os.path.isfile("model.txt"):
            self.image_array = io.imread(f"Images/{self.image_names[self.index]}")
        
            
            self.image_array = transform.resize(self.image_array,
                                                (int(self.image_array.shape[0]*self.resolution),
                                                int(self.image_array.shape[1]*self.resolution)))
            
            self.array_width = self.image_array.shape[1]
            self.array_height = self.image_array.shape[0]
            self.image_array = np.reshape(self.image_array, (self.image_array.shape[0]*self.image_array.shape[1], 3))*255
            model = lgb.Booster(model_file="model.txt")
            prediction = model.predict(self.image_array)
            prediction = np.reshape(prediction, (self.array_height, self.array_width))

            #save prediction to file with np.save
            np.save(f"predictions/{self.displayed_name}", prediction)



            dpi = 300
            plt.figure(figsize=(self.width/dpi, self.height/dpi), dpi=dpi)

            plt.axis('off')

            plt.imshow(prediction, cmap="Spectral")
            plt.savefig(f"predictions/{self.displayed_name}", bbox_inches='tight', pad_inches=0, dpi=dpi)
            preds_path = f"predictions/{self.displayed_name}"
            preds = QImage(preds_path)
            preds = preds.scaled(int(self.width*0.4), int(self.height*0.6))
            pixmap = QPixmap.fromImage(preds)
            pixmap = pixmap.scaled(int(self.width*0.4), int(self.height*0.6))
            self.image_label.setPixmap(pixmap)
            self.image_label.setVisible(True)
            
    def resizeEvent(self, event):
        # Get new dimensions of the widget
        new_size = event.size()
        new_width = new_size.width()
        new_height = new_size.height()
        print(f"New size: {new_width}x{new_height}")
        #self.open_image(self, predict=False)
        

    
    def prev_image(self):
        # Display the previous image in the list
        if self.index > 0:
            self.index -= 1
        elif self.index <= 0:
            self.index = self.max_index
        else: 
            return
        self.paint_widget.set_images(self.index)
        self.displayed_name = self.image_names[self.index]

    def next_image(self):
        # Display the next image in the list same way as prev_image
        if self.index >= self.max_index:
            self.index = 0
        elif self.index >= 0:
            self.index += 1
        else: 
            return
        self.paint_widget.set_images(self.index)
        self.displayed_name = self.image_names[self.index]

    def update_resolution(self, value):
     self.resolution = value

    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
