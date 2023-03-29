import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QSpinBox, QMessageBox, QApplication, QMainWindow, QFileDialog, QLabel, QPushButton, QVBoxLayout, QWidget, QSlider, QHBoxLayout

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class TrainModelThread(QThread):
    training_finished = pyqtSignal(float)

    def __init__(self, saved_array, num_rounds, image_array):
        super().__init__()
        self.saved_array = saved_array
        self.num_rounds = num_rounds
        self.image_array = image_array

    def run(self):
        print("run")
        if len(self.saved_array) == 0:
            return

        # Prepare the data
        X_train, X_test, y_train, y_test = train_test_split(self.saved_array[:,:-1], self.saved_array[:,-1],
                                                            test_size=0.3)
        scale_factor = y_train.sum()/(len(y_train)-y_train.sum())
        d_train = lgb.Dataset(data = X_train, label = y_train)
        d_valid = lgb.Dataset(data = X_test, label = y_test)
        params = {'objective'    : 'binary',
                'learning_rate': 0.02,
                'scale_pos_weight' : scale_factor,
                'boosting_type': 'goss',
                'metric'       : "binary_error",
                'num_leaves'   : 200,
                'max_depth'    : 10,
                'zero_as_missing' : 'true',
                'min_split_gain':0.01,
                'min_child_samples': 2,
                'n_jobs': 1}

        self.model = lgb.train(params, d_train, self.num_rounds, d_valid, early_stopping_rounds=None, verbose_eval=1)

        print("done")
        # Test the model
        y_pred = self.model.predict(X_test)
        y_pred = np.round(y_pred).astype(int)
        accuracy = accuracy_score(y_test, y_pred)

        # Emit a signal when the training is finished, passing the accuracy value
        self.prediction = self.model.predict(self.image_array)

        self.training_finished.emit(accuracy)
