from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap, QColor, QPainterPath, QPainterPathStroker

from PyQt5.QtWidgets import QLabel
import numpy as np


class PaintWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setPixmap(QPixmap())
        self.images = []
        self.thickness = 5
        self.image = None
        self.image_preds = None
        self.painted_pixels = np.empty((0, 4))
        self.temp_painted_pixels = np.empty((0, 3))

    def mousePressEvent(self, event):
        self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        global_pos = self.mapToGlobal(event.pos())
        widget_pos = self.mapFromGlobal(global_pos)

        painter = QPainter(self.pixmap())
        pen = QPen(Qt.black, self.thickness, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(self.paint_path(widget_pos))
        painter.end()

        path = self.paint_path(widget_pos)

        # Create a path that outlines the original path with the given pen
        stroker = QPainterPathStroker(pen)
        stroke_path = stroker.createStroke(path)

        painted_pixels = [np.array([0, 0, 0])]
        for x in range(int(stroke_path.boundingRect().left()), int(stroke_path.boundingRect().right())):
            for y in range(int(stroke_path.boundingRect().top()), int(stroke_path.boundingRect().bottom())):
                point = QPointF(x, y)
                if stroke_path.contains(point):
                    color = QColor(self.image.pixel(int(point.x()), int(point.y())))
                    print(int(point.x()), int(point.y()))
                    rgb = np.array([color.red(), color.green(), color.blue()])
                    painted_pixels.append(rgb)
        painted_pixels = np.array(painted_pixels)
        self.temp_painted_pixels = np.vstack((self.temp_painted_pixels, painted_pixels[1:,:]))
        self.start_pos = widget_pos
        self.update()


    def print_pixel_info(self, pos):
        if self.image:
            color = QColor(self.image.pixel(int(pos.x()), int(pos.y())))
            print(f"Pixel color at ({pos.x()}, {pos.y()}): R:{color.red()} G:{color.green()} B:{color.blue()}")


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
        

    def set_images(self, index=0):
        print(f"Setting image {index}")
        self.image = self.images[index]
        self.update_pixmap(self.image.width(), self.image.height())

    def set_image2(self, image):
        self.image_preds = image
        self.update_pixmap2(self.image_preds.width(), self.image_preds.height())
       

    def set_thickness(self, thickness):
        self.thickness = thickness

    def update_pixmap(self, w, h):
        pixmap = QPixmap.fromImage(self.image)
        pixmap = pixmap.scaled(w, h, Qt.KeepAspectRatio)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, pixmap)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self.update()
        painter.end()

    def update_pixmap2(self, w, h):
        pixmap = QPixmap.fromImage(self.image_preds)
        pixmap = pixmap.scaled(w, h, Qt.KeepAspectRatio)
        painter = QPainter(pixmap)
        painter.drawPixmap(0, 0, pixmap)
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())
        self.update()
        painter.end()

    def paint_path(self, end_pos):
        path = QPainterPath()
        path.moveTo(self.start_pos)
        path.lineTo(end_pos)
        return path

    def clear_temp_painted_pixels(self):
        self.temp_painted_pixels = np.empty((0, 3))
        if self.image_preds is not None:
            self.update_pixmap2(self.image_preds.width(), self.image_preds.height())
        else:
            self.update_pixmap(self.image.width(), self.image.height())

    def reset_painted_pixels(self):
        self.painted_pixels = np.empty((0, 3))
        self.clear_temp_painted_pixels()