import os
import pandas as pd
import rampwf as rw
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class Loss_ratio(ClassifierBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='loss_ratio', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        item_prices = np.asarray([2.50, 2.00, 1.50, 6.00, 3.00, 1.50, 3.50, 1.50, 3.00,2.0])
        real_price = item_prices[y_true_label_index]
        estimated_price = item_prices[y_pred_label_index]
        total_price = np.sum(real_price)
        l = 0
        for i in range (len(estimated_price)):
            if estimated_price[i] > real_price[i] :
                l += 0.2*real_price[i]
            elif estimated_price[i] < real_price[i] :
                l+= real_price[i] - estimated_price[i]
        loss = l/total_price
        return loss
    

problem_title =\
    'Fruits and vegetables classification (10 classes)'
_target_column_name = 'class'
_prediction_label_names = range(0, 10)
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.SimplifiedImageClassifier(
    n_classes=len(_prediction_label_names))

score_types = [
    rw.score_types.Accuracy(name='accuracy', precision=3),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
    Loss_ratio(name='loss_ratio', precision=2)
]



def get_cv(folder_X, y):
    _, X = folder_X
    cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df['id'].values
    y = df['class'].values
    folder = os.path.join(path, 'data', 'imgs')
    return (folder, X), y


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)
