import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from src.review.review import extract_corpus


# TFIDF uses it's own bow extraction
def get_bow_detail_tfidf(reviews, text_element, pos_tag='', **kwargs):
    # get corpus based on text_element and pos
    corpus, labels = extract_corpus(reviews, text_element, pos_tag)
    vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True, **kwargs)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features, labels


def get_tfidf_detail(reviews, text_element, pos_tag="", **kwargs):
    vectorizer, bow_features, labels = get_bow_detail_tfidf(reviews, text_element, pos_tag, **kwargs)
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_features = transformer.fit_transform(bow_features)
    return transformer, vectorizer, tfidf_features, labels


def get_tfidf_to_df(reviews, text_element, pos_tag='', **kwargs):
    transformer, vectorizer, features, labels = get_tfidf_detail(reviews, text_element, pos_tag, **kwargs)
    tf = features.todense()
    tf = np.array(tf, dtype='float64')
    ttf = features.todense().sum(axis=0)
    ttf = np.array(ttf, dtype='float64')
    names = vectorizer.get_feature_names()
    df_features = pd.DataFrame(data=tf, columns=names)
    df_totalfrequency = pd.DataFrame(data=ttf, columns=names)
    return df_features, df_totalfrequency
