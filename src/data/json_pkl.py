import json
import os.path
import mmap
import ntpath
import random
from tqdm import tqdm
from pickle import dump, load
from src.review.review import Review
from src.data.paths import get_source_data_path, get_processed_data_path, get_converted_data_path, \
    get_processed_test_data_path, get_processed_training_data_path


def test_json_reading(category):
    file_path = get_source_data_path(category)
    jsonfile = open(file_path, 'r')
    while 1:
        line = jsonfile.readline()
        sample = json.loads(line)
        print(sample)  # Look at the different fields
        reviewText = sample['reviewText']
    jsonfile.close()


def create_parsed_pkl_from_Json(category, random_seed, sample_size):
    file_path = get_source_data_path(category)
    review_set = set()
    with open(file_path) as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc="Parsing / Sampling / Writing PKL"
                                                                    + ntpath.basename(file_path)):
            sample = json.loads(line)
            reviewer_id = sample['reviewerID']
            asin = sample['asin']
            helpful = sample['helpful']
            text_raw = sample['reviewText']
            overall = sample['overall']
            summary = sample['summary']
            # if text raw not empty add to set
            if (text_raw != '') and (not text_raw is None):
                # create Amazon Review object
                review = Review(category.value, reviewer_id, asin, helpful, text_raw, overall, summary)
                # add review object to set
                review_set.add(review)
        file.close()
    random.seed(random_seed)
    sampled_review_set = random.sample(review_set, sample_size)
    write_reviews_parsed(category, sampled_review_set)


def write_reviews_parsed(amazoncategory, reviews_parsed):
    """Writes cleaned reviews set into data/datasets/processed/*.pkl."""
    # Get file path to processed folder.
    file_path = get_converted_data_path(amazoncategory)
    # Write file into file path.
    with open(file_path, 'wb') as f:
        dump(reviews_parsed, f)


def write_reviews_cleaned(amazoncategory, reviews_cleaned):
    """Writes cleaned reviews set into data/datasets/processed/*.pkl."""
    # Get file path to processed folder.
    file_path = get_processed_data_path(amazoncategory)
    # Write file into file path.
    with open(file_path, 'wb') as f:
        dump(reviews_cleaned, f)


def write_reviews_cleaned_sets(amazoncategory, training_set, test_set):
    file_path = get_processed_training_data_path(amazoncategory)
    with open(file_path, 'wb') as f:
        dump(training_set, f)
    file_path = get_processed_test_data_path(amazoncategory)
    with open(file_path, 'wb') as f:
        dump(test_set, f)


def load_parsed_reviews_pkl(amazoncategory):
    """Load cleaned reviews into session."""
    file_path_ext = get_converted_data_path(amazoncategory)
    if os.path.exists(file_path_ext):
        with open(file_path_ext, 'rb') as f:
            return load(f)


def load_cleaned_reviews_pkl(amazoncategory):
    """Load cleaned reviews into session."""
    file_path_ext = get_processed_data_path(amazoncategory)
    if os.path.exists(file_path_ext):
        with open(file_path_ext, 'rb') as f:
            return load(f)


def load_train_reviews_pkl(amazoncategory):
    """Load cleaned train set reviews into session."""
    file_path_ext = get_processed_training_data_path(amazoncategory)
    if os.path.exists(file_path_ext):
        with open(file_path_ext, 'rb') as f:
            return load(f)


def load_test_reviews_pkl(amazoncategory):
    """Load cleaned test set reviews into session."""
    file_path_ext = get_processed_test_data_path(amazoncategory)
    if os.path.exists(file_path_ext):
        with open(file_path_ext, 'rb') as f:
            return load(f)



def get_num_lines(file_path):
    fp = open(file_path, "r+")
    lines = 0
    # mmap can fail with low memory
    try:
        buf = mmap.mmap(fp.fileno(), 0)
        while buf.readline():
            lines += 1
    finally:
        return lines
