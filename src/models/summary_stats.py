import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.json_pkl import load_cleaned_reviews_pkl
from src.models.bow_sklearn import get_bow_to_df
from src.models.tfidf_sklearn import get_tfidf_to_df
from src.review.review import Text_Element
from src.visualization.visualization import generate_bar_chart_from_totalfrequencies


def calculate_average_words(reviews, text_element):
    total_word_count = 0
    total_reviews = 0
    for review in tqdm(reviews, desc='Stats: calculate average words'):
        if not review.get_text_element(text_element) is None:
            total_word_count += review.count_number_of_words(text_element)
            total_reviews += 1
    if total_reviews == 0:
        return 0
    else:
        return int(np.round((total_word_count / total_reviews), 0))


def get_data_exploration_base_statistics(categories):
    # Generate average word count
    stat_list = list()
    for c in categories:
        reviews = load_cleaned_reviews_pkl(c)
        raw_avg = calculate_average_words(reviews, Text_Element.S01_raw)
        final_avg = calculate_average_words(reviews, Text_Element.SF_finalized)
        compression = np.round(final_avg / raw_avg, 2)
        stat_list.append([c.value, raw_avg, final_avg, compression])
    stats = pd.DataFrame(data=stat_list, columns=['category', 'raw word avg count', 'norm count', 'compression'])
    print(stats)


def get_data_exploration_per_cat_bow(file_tag, categories, text_element, pos_tag='', **kwargs):
    for c in categories:
        reviews = load_cleaned_reviews_pkl(c)
        df, dft = get_bow_to_df(reviews, text_element, pos_tag, **kwargs)
        file_name = file_tag + '_categ_' + str(c.value) + '_' + pos_tag
        if 'ngram_range' in kwargs:
            file_name += "_ngram_" + str(kwargs.get('ngram_range'))
        if "max_features" in kwargs:
            file_name += '_top_' + str(kwargs.get('max_features'))
        generate_bar_chart_from_totalfrequencies(dft, file_name + '.png')


def get_data_exploration_all_cat_bow(file_tag, categories, text_element, pos_tag='', **kwargs):
    init = True
    for c in tqdm(categories, desc="Data Exploration - Assembling all categories "):
        if init:
            reviews = load_cleaned_reviews_pkl(c)
            init = False
        else:
            reviews = reviews + load_cleaned_reviews_pkl(c)

    df, dft = get_bow_to_df(reviews, text_element, pos_tag, **kwargs)
    file_name = file_tag + '_categ_all_' + pos_tag
    if 'ngram_range' in kwargs:
        file_name += "_ngram_" + str(kwargs.get('ngram_range'))
    if "max_features" in kwargs:
        file_name += '_top_' + str(kwargs.get('max_features'))
    generate_bar_chart_from_totalfrequencies(dft, file_name + '.png')


def get_data_exploration_all_cat_tfidf(file_tag, categories, text_element, pos_tag='', **kwargs):
    init = True
    for c in tqdm(categories, desc="Data Exploration - Assembling all categories "):
        if init:
            reviews = load_cleaned_reviews_pkl(c)
            init = False
        else:
            reviews = reviews + load_cleaned_reviews_pkl(c)

    df, dft = get_tfidf_to_df(reviews, text_element, pos_tag, **kwargs)
    file_name = file_tag + '_categ_all_' + pos_tag
    if 'ngram_range' in kwargs:
        file_name += "_ngram_" + str(kwargs.get('ngram_range'))
    if "max_features" in kwargs:
        file_name += '_top_' + str(kwargs.get('max_features'))
    generate_bar_chart_from_totalfrequencies(dft, file_name + '.png')
