import time

import pandas as pd
from sklearn.linear_model import SGDClassifier

from src.models.svm_common import train_predict_evaluate_model
from src.models.tfidf_sklearn import get_tfidf_detail
from src.review.review import extract_corpus
from src.utils.time import timer


def _generate_svm_tfidf_data(train_reviews, test_reviews, text_element, pos_tag="", **kwargs):
    tfidef_transformer, tfidf_vectorizer, tfidf_train_features, tfidf_train_labels = get_tfidf_detail(train_reviews,
                                                                                                      text_element,
                                                                                                      pos_tag,
                                                                                                      **kwargs)
    tfidf_test_corpus, tfidf_test_labels = extract_corpus(test_reviews, text_element, pos_tag)
    tfidf_test_features = tfidf_vectorizer.transform(tfidf_test_corpus)
    return tfidf_train_features, tfidf_train_labels, tfidf_test_features, tfidf_test_labels


def svm_tfidf_classification(train_reviews, test_reviews, text_element, class_labels, caption, pos_tag="", **kwargs):
    # get elements
    start_time = time.time()
    tfidf_train_features, tfidf_train_labels, tfidf_test_features, tfidf_test_labels = _generate_svm_tfidf_data(
        train_reviews,
        test_reviews,
        text_element,
        pos_tag, **kwargs)

    svm = SGDClassifier(loss='hinge', max_iter=50)

    svm_tfidf_predictions, metrics = train_predict_evaluate_model(classifier=svm,
                                                                  train_features=tfidf_train_features,
                                                                  train_labels=tfidf_train_labels,
                                                                  test_features=tfidf_test_features,
                                                                  test_labels=tfidf_test_labels,
                                                                  class_labels=class_labels,
                                                                  caption=caption)
    # Assemble the data list for the model
    result = list()
    result.append(['Model', 'TFIDF'])
    result.append(['pos filter', pos_tag])
    for key, value in kwargs.items():
        result.append([key, value])
    result = result + metrics
    result.append(['exec time', timer(start_time, time.time())])
    results = pd.DataFrame(data=result, columns=['Data', 'Value'])
    print(results)
