import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

from src.data.paths import get_output_path


def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels, class_labels,
                                 caption):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    metrics = _get_metrics(true_labels=test_labels,
                           predicted_labels=predictions)

    _confusion_matrix(predictions, test_labels, class_labels, "conf_matrix_" + caption + '.png')
    return predictions, metrics


def _get_metrics(true_labels, predicted_labels):
    results = list()
    results.append(['accuracy', np.round(metrics.accuracy_score(true_labels, predicted_labels), 4)])
    results.append(
        ['precision', np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 4)])
    results.append(['recall', np.round(metrics.recall_score(true_labels, predicted_labels, average='weighted'), 4)])
    results.append(['F1 score', np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 4)])
    return results


def _confusion_matrix(test_labels, predictions, target_names, filename):
    conf_mat = confusion_matrix(test_labels, predictions)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=target_names, yticklabels=target_names, cmap="Blues")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(get_output_path(filename))
