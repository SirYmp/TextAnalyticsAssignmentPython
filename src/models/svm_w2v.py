import time

import pandas as pd
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from src.models.svm_common import train_predict_evaluate_model
from src.models.w2v_gensim import get_averaged_word_vectorizer_detail, _average_word_vectors
from src.review.review import extract_corpus_as_is
from src.utils.time import timer


def _get_w2v_data(train_reviews, test_reviews, text_element, size=500, min_count=2, window=10, sample=1e-3):
    model, vocabulary, train_features, train_labels = get_averaged_word_vectorizer_detail(train_reviews,
                                                                                          text_element, size,
                                                                                          min_count, window,
                                                                                          sample)

    test_corpus, test_labels = extract_corpus_as_is(test_reviews, text_element)

    test_features = [_average_word_vectors(sentence, model, vocabulary, size) for sentence in
                     tqdm(test_corpus, desc='w2v - calculating test processing ')]

    return train_features, train_labels, test_features, test_labels


def svm_w2v_classification(train_reviews, test_reviews, text_element, class_labels, caption, size=500, min_count=2,
                           window=10, sample=1e-3):
    start_time = time.time()
    # get elements
    train_features, train_labels, test_features, test_labels = _get_w2v_data(train_reviews,
                                                                             test_reviews,
                                                                             text_element,
                                                                             size, min_count, window, sample)

    svm = SGDClassifier(loss='hinge', max_iter=100)

    svm_w2v_predictions, metrics = train_predict_evaluate_model(classifier=svm,
                                                                train_features=train_features,
                                                                train_labels=train_labels,
                                                                test_features=test_features,
                                                                test_labels=test_labels,
                                                                class_labels=class_labels,
                                                                caption=caption)
    result = list()
    result.append(['Model', 'W2V'])
    result.append(['size', size])
    result.append(['min_count', min_count])
    result.append(['window', window])
    result.append(['sample', sample])
    result = result + metrics
    result.append(['exec time', timer(start_time, time.time())])
    results = pd.DataFrame(data=result, columns=['Data', 'Value'])
    print(results)
