import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PyQt5.QtCore import QThread, pyqtSignal


class TrainModelThread(QThread):
    training_finished = pyqtSignal(float)

    def __init__(self, saved_array, num_rounds, image_array, resolution=1.0, load_model=False):
        super().__init__()
        self.saved_array = saved_array
        self.num_rounds = num_rounds
        self.image_array = image_array*255
        self.resolution = resolution
        self.load_model = load_model
    

    def run(self):
        print("run")
        if len(self.saved_array) == 0 and not self.load_model:
            return

        if self.load_model:
            # Load a saved model from disk
            pass

        else:
            # Prepare the data
            X_train, X_test, y_train, y_test = train_test_split(self.saved_array[:,:-1], self.saved_array[:,-1],
                                                                test_size=0.5)
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

            self.model = lgb.train(params, d_train, self.num_rounds, d_valid, early_stopping_rounds=20, verbose_eval=10)
            self.model.save_model('model.txt', num_iteration=self.model.best_iteration)

            # Test the model
            y_pred = self.model.predict(X_test)
            y_pred = np.round(y_pred).astype(int)
            accuracy = accuracy_score(y_test, y_pred)

        # Make predictions on the input image
        self.prediction = self.model.predict(self.image_array)
        accuracy=0.5
        # Emit a signal when the training is finished, passing the accuracy value
        self.training_finished.emit(accuracy)
