"""Multiclass metrics without a large evaluation dependency."""

import numpy as np

from . import CLASSES


def classification_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Expected equal, non-empty target and prediction arrays")
    n = len(CLASSES)
    matrix = np.bincount(actual * n + predicted, minlength=n * n).reshape(n, n)
    tp = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(n, dtype=float), where=support != 0)
    precision = np.divide(tp, predicted_count, out=np.zeros(n, dtype=float), where=predicted_count != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n, dtype=float), where=(precision + recall) != 0)
    present = support > 0
    return {"accuracy": float(tp.sum() / support.sum()),
            "macro_f1": float(f1[present].mean()),
            "balanced_accuracy": float(recall[present].mean()),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(CLASSES)},
            "confusion_matrix": matrix.tolist()}
