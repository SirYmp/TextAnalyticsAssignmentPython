import random

from src.data.json_pkl import create_parsed_pkl_from_Json, load_cleaned_reviews_pkl, load_test_reviews_pkl, \
    load_train_reviews_pkl
from src.data.paths import AmazonCategory
from src.data.xlsx import append_df_to_excel
from src.features.process_review.process_review_files import create_processed_review_file, \
    assemble_training_test_subsets
from src.models.bow_sklearn import get_bow_to_df
from src.models.tfidf_sklearn import get_tfidf_to_df
from src.models.word2vec_gensim import averaged_word_vectorizer, tfidf_weighted_averaged_word_vectorizer
from src.review.review import Text_Element
from src.visualization.visualization import generate_bar_chart_from_totalfrequencies

_output_excel = False
_output_graph = False

_run_Parse = False
_run_Processing = False
_run_Assemble_Training_Test = False
_run_ReviewTests = True
_run_BoW = False
_run_TFIDF = False
_run_Word2Vec = False
_run_Word2Vec_weighted_TFIDF = False




def main():
    # create parsed pkl with a review set from the JSON Files
    # _______________________________________________________
    if _run_Parse:
        create_parsed_pkl_from_Json(AmazonCategory.amazon_kindle, 18745, 100000)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_moviesandtv, 98765, 100000)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_videogames, 49864, 100000)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_electronics, 51234, 100000)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_books, 81234, 100000)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_clothing_shoes_jewel, 91298, 100000)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_beauty, 74563, 100000)

    # create cleaned pkl (with text transformations) from initially parsed pkl
    # _______________________________________________________
    if _run_Processing:
        create_processed_review_file(AmazonCategory.amazon_kindle)
        create_processed_review_file(AmazonCategory.amazon_moviesandtv)
        create_processed_review_file(AmazonCategory.amazon_videogames)
        create_processed_review_file(AmazonCategory.amazon_electronics)
        create_processed_review_file(AmazonCategory.amazon_books)
        create_processed_review_file(AmazonCategory.amazon_clothing_shoes_jewel)
        create_processed_review_file(AmazonCategory.amazon_beauty)

    # load review into object
    # _______________________________________________________
    if _run_Assemble_Training_Test:
        assemble_training_test_subsets(AmazonCategory.amazon_kindle, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_moviesandtv, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_videogames, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_electronics, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_books, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_clothing_shoes_jewel, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_beauty, 0.7)

    reviews = load_train_reviews_pkl(AmazonCategory.amazon_kindle)
    new_set = load_train_reviews_pkl(AmazonCategory.amazon_moviesandtv)
    reviews += new_set
    new_set = load_train_reviews_pkl(AmazonCategory.amazon_videogames)
    reviews += new_set
    new_set = load_train_reviews_pkl(AmazonCategory.amazon_electronics)
    reviews += new_set
    new_set = None

    if _run_ReviewTests:
        for rev in random.sample(reviews, 100):
            print('Original:')
            print(rev.text_raw)
            print('Cleaned:')
            print(rev.text_cleaned)
            print('-----------------------------------------')

    # BOW parameters: set, text_element, fid POS tag, n-gram
    if _run_BoW:
        df, dft = get_bow_to_df(reviews, Text_Element.SF_finalized, pos_tag='', max_features=100, ngram_range=(1, 1))
        if _output_graph:
            generate_bar_chart_from_totalfrequencies(dft)
        if _output_excel:
            append_df_to_excel('bow_output.xlsx', df, 'bow_ngram_1_1 top 1000')
        df, dft = get_bow_to_df(reviews, Text_Element.SF_finalized, pos_tag='', max_features=100, ngram_range=(2, 2))
        if _output_graph:
            generate_bar_chart_from_totalfrequencies(dft)
        if _output_excel:
            append_df_to_excel('bow_output.xlsx', df, 'bow_ngram_2_2 top 1000')
        df, dft = get_bow_to_df(reviews, Text_Element.SF_finalized, pos_tag='NOUN', max_features=100,
                                ngram_range=(1, 1))
        if _output_excel:
            append_df_to_excel('bow_output.xlsx', df, 'bow_ngram_1_1 NOUN top 1000')
        if _output_graph:
            generate_bar_chart_from_totalfrequencies(dft)

    # TFIDF
    if _run_TFIDF:
        df = get_tfidf_to_df(reviews, Text_Element.SF_finalized, pos_tag='', max_features=500, ngram_range=(1, 1))
        if _output_excel:
            append_df_to_excel('tfidf_output.xlsx', df, 'tdidf_ngram_1_1')
        df = get_tfidf_to_df(reviews, Text_Element.SF_finalized, pos_tag='NOUN', max_features=500, ngram_range=(1, 1))
        if _output_excel:
            append_df_to_excel('tfidf_output.xlsx', df, 'tdidf_ngram_1_1 NOUN')

    # Word2Vec: REQUIRES SENTENCE & TEXT TOKENIZATION
    if _run_Word2Vec:
        df = averaged_word_vectorizer(reviews, Text_Element.S16_direct_word_tokenize, size=10, min_count=2,
                                      num_features=10, window=10)
        print(df)

    if _run_Word2Vec_weighted_TFIDF:
        df = tfidf_weighted_averaged_word_vectorizer(reviews, text_element=Text_Element.SF_finalized,
                                                     tokenized_text_element=Text_Element.S16_direct_word_tokenize,
                                                     size=10, min_count=2, num_features=10, window=10)


if __name__ == '__main__':
    main()
