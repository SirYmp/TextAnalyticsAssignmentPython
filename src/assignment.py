import random
import time

from tqdm import tqdm

from src.data.json_pkl import create_parsed_pkl_from_Json, load_test_reviews_pkl, load_train_reviews_pkl
from src.data.paths import AmazonCategory
from src.models.summarization import lsa_text_summarizer
from src.models.summary_stats import get_data_exploration_per_cat_bow, \
    get_data_exploration_all_cat_bow, get_data_exploration_base_statistics, get_data_exploration_all_cat_tfidf
from src.models.svm_bow import svm_bow_classification
from src.models.svm_tfidf import svm_tfidf_classification
from src.models.svm_w2v import svm_w2v_classification
from src.processing.process_review.process_review_files import create_processed_review_file, \
    assemble_training_test_subsets, create_processed_review_file_redux
from src.review.review import Text_Element

_run_Parse = False
_run_Processing = False
_run_Processing_Redux = False
_run_Assemble_Training_Test = False

_run_ReviewTests = True
_run_exploration_BoW = True
_run_exploration_TFIDF = True
_run_exploration_Word2Vec = True
_run_exploration_Word2Vec_weighted_TFIDF = True

_run_svm = True
_run_svm_extend = False
_run_lsa = True

_output_excel = False
_output_graph = False


def main():
    # create parsed pkl with a review set from the JSON Files
    # _______________________________________________________
    if _run_Parse:
        create_parsed_pkl_from_Json(AmazonCategory.amazon_kindle, 999000111, 100000, 1)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_moviesandtv, 222000999, 100000, 2)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_videogames, 555000777, 100000, 3)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_electronics, 444333000, 100000, 4)
        create_parsed_pkl_from_Json(AmazonCategory.amazon_books, 111222333, 100000, 5)

    # create cleaned pkl (with text transformations) from initially parsed pkl
    # _______________________________________________________
    if _run_Processing:
        if _run_Processing_Redux:
            create_processed_review_file_redux(AmazonCategory.amazon_kindle)
            create_processed_review_file_redux(AmazonCategory.amazon_moviesandtv)
            create_processed_review_file_redux(AmazonCategory.amazon_videogames)
            create_processed_review_file_redux(AmazonCategory.amazon_electronics)
            create_processed_review_file_redux(AmazonCategory.amazon_books)

        else:
            create_processed_review_file(AmazonCategory.amazon_kindle)
            create_processed_review_file(AmazonCategory.amazon_moviesandtv)
            create_processed_review_file(AmazonCategory.amazon_videogames)
            create_processed_review_file(AmazonCategory.amazon_electronics)
            create_processed_review_file(AmazonCategory.amazon_books)

    if _run_Assemble_Training_Test:

        assemble_training_test_subsets(AmazonCategory.amazon_kindle, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_moviesandtv, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_videogames, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_electronics, 0.7)
        assemble_training_test_subsets(AmazonCategory.amazon_books, 0.7)

    if _run_ReviewTests:
        reviews = load_train_reviews_pkl(AmazonCategory.amazon_kindle)
        reviews = reviews + load_train_reviews_pkl(AmazonCategory.amazon_moviesandtv)
        reviews = reviews + load_train_reviews_pkl(AmazonCategory.amazon_videogames)
        reviews = reviews + load_train_reviews_pkl(AmazonCategory.amazon_electronics)

        # check small lenght reviews
        for rev in reviews:
            if rev.count_number_of_words(Text_Element.SF_finalized) < 3:
                print(rev.get_text_element(Text_Element.SF_finalized))

        for rev in random.sample(reviews, 2):
            print('\n ')
            print(rev.get_full_description())

        for rev in random.sample(reviews, 100):
            print('Original:')
            print(rev.text_raw)
            print('S14 :')
            print(rev.get_text_element(Text_Element.S14_remove_stopwords))
            print('Cleaned:')
            print(rev.text_cleaned)
            print('-----------------------------------------')

    # BOW parameters: set, text_element, POS tag, n-gram
    if _run_exploration_BoW:
        all_categories = [AmazonCategory.amazon_books, AmazonCategory.amazon_videogames,
                          AmazonCategory.amazon_moviesandtv, AmazonCategory.amazon_kindle,
                          AmazonCategory.amazon_electronics]

        # get base statistics
        get_data_exploration_base_statistics(all_categories)

        # get per category bow data
        # get_data_exploration_per_cat_bow('bow raw', all_categories, Text_Element.S01_raw, max_features=100,
        #                                 ngram_range=(1, 1))
        get_data_exploration_per_cat_bow('bow_norm', all_categories, Text_Element.SF_finalized, max_features=20,
                                         ngram_range=(1, 1))
        # get_data_exploration_per_cat_bow('bow raw', all_categories, Text_Element.S01_raw, max_features=100,
        #                                 ngram_range=(2, 2))
        get_data_exploration_per_cat_bow('bow_norm', all_categories, Text_Element.SF_finalized, max_features=20,
                                         ngram_range=(2, 2))
        # get_data_exploration_per_cat_bow('bow raw', all_categories, Text_Element.S01_raw, pos_tag='NOUN',
        #                                 max_features=100, ngram_range=(1, 1))
        get_data_exploration_per_cat_bow('bow_norm', all_categories, Text_Element.SF_finalized, pos_tag='NOUN',
                                         max_features=20, ngram_range=(1, 1))

        # get full-set bow data
        model_categories = [AmazonCategory.amazon_videogames, AmazonCategory.amazon_moviesandtv,
                            AmazonCategory.amazon_kindle, AmazonCategory.amazon_electronics]

        # get_data_exploration_all_cat_bow('bow_raw_model', model_categories, Text_Element.S01_raw, max_features=20,
        #                                 ngram_range=(1, 1))
        get_data_exploration_all_cat_bow('bow_norm_model', model_categories, Text_Element.SF_finalized,
                                         max_features=20, ngram_range=(1, 1))
        # get_data_exploration_all_cat_bow('bow_raw_model', model_categories, Text_Element.S01_raw, max_features=100,
        #                                 ngram_range=(2, 2))
        get_data_exploration_all_cat_bow('bow_norm_model', model_categories, Text_Element.SF_finalized,
                                         max_features=20, ngram_range=(2, 2))
        # get_data_exploration_all_cat_bow('bow_raw_model', model_categories, Text_Element.S01_raw, pos_tag='NOUN',
        #                                 max_features=100, ngram_range=(1, 1))
        get_data_exploration_all_cat_bow('bow_norm_model', model_categories, Text_Element.SF_finalized, pos_tag='NOUN',
                                         max_features=20, ngram_range=(1, 1))

    # TFIDF
    if _run_exploration_TFIDF:
        model_categories = [AmazonCategory.amazon_videogames, AmazonCategory.amazon_moviesandtv,
                            AmazonCategory.amazon_kindle, AmazonCategory.amazon_electronics]

        get_data_exploration_all_cat_tfidf('tfidf_norm_model', model_categories, Text_Element.SF_finalized, pos_tag=''
                                           , max_features=20, ngram_range=(1, 1))
        get_data_exploration_all_cat_tfidf('tfidf_norm_model', model_categories, Text_Element.SF_finalized, pos_tag=''
                                           , max_features=20, ngram_range=(2, 2))

    if _run_svm:

        if _run_svm_extend:
            category_set = [AmazonCategory.amazon_books, AmazonCategory.amazon_videogames,
                            AmazonCategory.amazon_moviesandtv, AmazonCategory.amazon_kindle,
                            AmazonCategory.amazon_electronics]
            extension = '5set'
            class_labels = ['1', '2', '3', '4', '5']
            max_feature_set = [200, 1000]
        else:
            # define categories in the set
            category_set = [AmazonCategory.amazon_videogames,
                            AmazonCategory.amazon_moviesandtv, AmazonCategory.amazon_kindle,
                            AmazonCategory.amazon_electronics]
            class_labels = ['1', '2', '3', '4']
            extension = '4set'
            max_feature_set = [250, 500, 750, 1000]

        # assemble categories
        init = True
        for category in tqdm(category_set, desc="SVM: assembling training, test sets "):
            if init:
                train_reviews = load_train_reviews_pkl(category)
                test_reviews = load_test_reviews_pkl(category)
                init = False
            else:
                train_reviews = train_reviews + load_train_reviews_pkl(category)
                test_reviews = test_reviews + load_test_reviews_pkl(category)

        for mf in max_feature_set:
            # just to avoid overlaps with tqdm
            time.sleep(2)

            svm_bow_classification(train_reviews, test_reviews, Text_Element.SF_finalized, class_labels,
                                   'SVM_BOW_' + str(mf) + '_' + extension, max_features=mf)
            time.sleep(2)
            svm_bow_classification(train_reviews, test_reviews, Text_Element.SF_finalized, class_labels,
                                   'SVM_BOW_BIGRAM_' + str(mf) + '_' + extension, max_features=mf, ngram_range=(2, 2))
            time.sleep(2)
            svm_bow_classification(train_reviews, test_reviews, Text_Element.SF_finalized, class_labels,
                                   'SVM_BOW_NOUNS_' + str(mf) + '_' + extension, pos_tag='NOUN', max_features=mf)
            time.sleep(2)
            svm_tfidf_classification(train_reviews, test_reviews, Text_Element.SF_finalized, class_labels,
                                     'SVM_TFIDF_' + str(mf) + '_' + extension, max_features=mf)
            time.sleep(2)
            svm_w2v_classification(train_reviews, test_reviews, Text_Element.S16_direct_word_tokenize, class_labels,
                                   'SVM_W2V_' + str(mf) + '_' + extension, mf, min_count=2, window=100)

    if _run_lsa:
        category_set = [AmazonCategory.amazon_videogames,
                        AmazonCategory.amazon_moviesandtv, AmazonCategory.amazon_kindle,
                        AmazonCategory.amazon_electronics]

        for category in tqdm(category_set, desc="LSA: assembling training set"):
            train_reviews = load_train_reviews_pkl(category)
            print('LSA frequency (BOW) for: ' + str(category))

            lsa_text_summarizer(train_reviews, Text_Element.S11_sentence_tokenize, num_sentences=20, num_topics=2,
                                feature_type='frequency', sv_threshold=0.5)

            print('LSA tfidf for: ' + str(category))
            lsa_text_summarizer(train_reviews, Text_Element.S11_sentence_tokenize, num_sentences=20, num_topics=2,
                                feature_type='tfidf', sv_threshold=0.5)


if __name__ == '__main__':
    main()
