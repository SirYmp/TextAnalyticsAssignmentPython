import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer

from src.review.review import extract_corpus


# TFIDF uses it's own bow extraction
def get_bow_detail_tfidf(reviews, text_element, pos_tag='', **kwargs):
    # get corpus based on text_element and pos
    corpus = extract_corpus(reviews, text_element, pos_tag)
    vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, use_idf=True, **kwargs)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def get_tfidf_detail(reviews, text_element, pos_tag="", **kwargs):
    vectorizer, bow_features = get_bow_detail_tfidf(reviews, text_element, pos_tag, **kwargs)
    transformer = TfidfTransformer(norm='l2',
                                   smooth_idf=True,
                                   use_idf=True)
    tfidf_features = transformer.fit_transform(bow_features)

    return transformer, vectorizer, tfidf_features


def get_tfidf_to_df(reviews, text_element, pos_tag='', **kwargs):
    transformer, vectorizer, tfidf_features = get_tfidf_detail(reviews, text_element, pos_tag, **kwargs)
    names = vectorizer.get_feature_names()
    tfidf_features = np.round(tfidf_features.todense(), 2)
    df = pd.DataFrame(data=tfidf_features, columns=names)

    return df
