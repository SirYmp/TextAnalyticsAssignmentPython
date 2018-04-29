import random

from tqdm import tqdm

from src.data.json_pkl import write_reviews_cleaned, load_parsed_reviews_pkl, load_cleaned_reviews_pkl, \
    write_reviews_cleaned_sets
from src.data.paths import get_converted_file_name
from src.processing.process_review.clean_review import clean_review, clean_review_reduced_dictionary


def create_processed_review_file(amazoncategory):
    reviews = load_parsed_reviews_pkl(amazoncategory)
    # clean samples
    for review in tqdm(reviews, desc="Cleaning & Normalizing " + get_converted_file_name(amazoncategory)):
        clean_review(review)
    # write processed file
    write_reviews_cleaned(amazoncategory, reviews)


def create_processed_review_file_redux(amazoncategory):
    reviews = load_parsed_reviews_pkl(amazoncategory)
    # clean samples
    for review in tqdm(reviews, desc="Cleaning & Normalizing (REDUX) " + get_converted_file_name(amazoncategory)):
        clean_review_reduced_dictionary(review)
    # write processed file
    write_reviews_cleaned(amazoncategory, reviews)

def assemble_training_test_subsets(amazoncategory, training_cut_percent):
    reviews = load_cleaned_reviews_pkl(amazoncategory)
    random.shuffle(reviews)
    cut = abs(len(reviews) * training_cut_percent)
    training_set = reviews[:int(cut)]
    test_set = reviews[int(cut):]
    write_reviews_cleaned_sets(amazoncategory, training_set, test_set)
