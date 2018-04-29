import time

import pandas as pd
from sklearn.linear_model import SGDClassifier

from src.models.bow_sklearn import get_bow_detail
from src.models.svm_common import train_predict_evaluate_model
from src.review.review import extract_corpus
from src.utils.time import timer


def _generate_svm_bow_data(train_reviews, test_reviews, text_element, pos_tag="", **kwargs):
    bow_vectorizer, bow_train_features, bow_train_labels = get_bow_detail(train_reviews, text_element, pos_tag,
                                                                          **kwargs)
    bow_test_corpus, bow_test_labels = extract_corpus(test_reviews, text_element, pos_tag)
    bow_test_features = bow_vectorizer.transform(bow_test_corpus)
    return bow_train_features, bow_train_labels, bow_test_features, bow_test_labels


def svm_bow_classification(train_reviews, test_reviews, text_element, class_labels, caption, pos_tag="", **kwargs):
    # get elements
    start_time = time.time()
    bow_train_features, bow_train_labels, bow_test_features, bow_test_labels = _generate_svm_bow_data(train_reviews,
                                                                                                      test_reviews,
                                                                                                      text_element,
                                                                                                      pos_tag, **kwargs)
    svm = SGDClassifier(loss='hinge', max_iter=50)

    svm_bow_predictions, metrics = train_predict_evaluate_model(classifier=svm,
                                                                train_features=bow_train_features,
                                                                train_labels=bow_train_labels,
                                                                test_features=bow_test_features,
                                                                test_labels=bow_test_labels, class_labels=class_labels,
                                                                caption=caption)
    # Assemble the data list for the model
    result = list()
    result.append(['Model', 'BOW'])
    result.append(['pos filter', pos_tag])
    for key, value in kwargs.items():
        result.append([key, value])
    result = result + metrics
    result.append(['exec time', timer(start_time, time.time())])
    results = pd.DataFrame(data=result, columns=['Data', 'Value'])
    print(results)
