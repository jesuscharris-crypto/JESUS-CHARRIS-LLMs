import numpy as np
from sklearn.metrics import f1_score


def evaluar_f1_promedio(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
