import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from src.models.tfidf_sklearn import get_tfidf_detail
from src.review.review import extract_corpus_as_is


def _compute_word2vec(corpus, size, min_count, window=10, sample=1e-3):
    model = Word2Vec(corpus, size=size, min_count=min_count, window=window, sample=sample)
    return model


def _average_word_vectors(words, w2v_model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0

    for word in words:
        # if type(word) == list: print('word is list' + word)
        if word in vocabulary:
            nwords += 1
            feature_vector = np.add(feature_vector, w2v_model[word])
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def _tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                   if tfidf_vocabulary.get(word)
                   else 0 for word in words]
    word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    feature_vector = np.zeros((num_features,), dtype="float64")
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)
    return feature_vector


def get_averaged_word_vectorizer_detail(reviews, tokenized_text_element, size, min_count, window=10, sample=1e-3):
    corpus, labels = extract_corpus_as_is(reviews, tokenized_text_element)
    model = _compute_word2vec(corpus, size, min_count, window, sample)
    vocabulary = set(model.wv.index2word)
    features = [_average_word_vectors(sentence, model, vocabulary, size) for sentence in
                tqdm(corpus, desc='w2v - calculating processing ')]
    return model, vocabulary, np.array(features), labels


def tfidf_wtd_averaged_word_vectorizer(reviews, text_element, tokenized_text_element, size, min_count,
                                       num_features=10, window=10, sample=1e-3):
    # gets the tfidf vector and vocabulary
    transformer, vectorizer, tfidf_vectors, labels = get_tfidf_detail(reviews, text_element)
    tfidf_vocabulary = vectorizer.vocabulary_
    # gets the tokenized corpus
    corpus, labels = extract_corpus_as_is(reviews, tokenized_text_element)
    # gets the model
    model = _compute_word2vec(corpus, size, min_count, window, sample)
    docs_tfidfs = [(doc, doc_tfidf)
                   for doc, doc_tfidf
                   in zip(corpus, tfidf_vectors)]
    features = [_tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                            model, num_features)
                for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)
