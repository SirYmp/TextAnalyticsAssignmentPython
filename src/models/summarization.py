import numpy as np
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

from src.review.review import extract_corpus_as_is


def _build_feature_matrix(sentences, feature_type='frequency'):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectorizer = CountVectorizer(binary=True, min_df=1, ngram_range=(1, 1))
    elif feature_type == 'frequency':
        vectorizer = CountVectorizer(binary=False, min_df=1, ngram_range=(1, 1))
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
    else:
        raise Exception("Wrong feature type entered. Possible values: 'binary', 'frequency', 'tfidf'")
    feature_matrix = vectorizer.fit_transform(sentences).astype(float)
    return vectorizer, feature_matrix


def _low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt


def lsa_text_summarizer(reviews, text_element, num_sentences=100, num_topics=2, feature_type='frequency',
                        sv_threshold=0.5):
    # Extract Sentences
    sentences = list()
    corpus, labels = extract_corpus_as_is(reviews, text_element)
    for c in tqdm(corpus, desc="extracting sentence"):
        for sentence in c:
            sentences.append(sentence)

    vec, dt_matrix = _build_feature_matrix(sentences, feature_type=feature_type)
    td_matrix = dt_matrix.transpose()
    td_matrix = td_matrix.multiply(td_matrix > 0)
    u, s, vt = _low_rank_svd(td_matrix, singular_count=num_topics)
    min_sigma_value = max(s) * sv_threshold
    s[s < min_sigma_value] = 0
    salience_scores = np.sqrt(np.dot(np.square(s), np.square(vt)))
    top_sentence_indices = salience_scores.argsort()[-num_sentences:][::-1]
    top_sentence_indices.sort()
    print("processed with: " + feature_type)
    for index in top_sentence_indices:
        print(sentences[index])
