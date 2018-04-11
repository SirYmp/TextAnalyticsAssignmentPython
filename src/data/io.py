import json
import os.path
import mmap
import sys
from tqdm import tqdm
from pickle import dump, load
from src.review.review import Review
from src.utils.paths import get_source_data_path, get_processed_data_path, AmazonCategory


def test_json_reading(category):
    file_path = get_source_data_path(category)
    print("Json file load: " + file_path)
    jsonfile = open(file_path, 'r')
    while 1:
        line = jsonfile.readline()
        sample = json.loads(line)
        print(sample)  # Look at the different fields
        reviewText = sample['reviewText']
    jsonfile.close()


def create_reviews_set(category):
    file_path = get_source_data_path(category)
    review_set = set()
    # flush tqdm channel to avoid conflicts and garbage prints
    sys.stderr.flush
    print("Json file: " + file_path)
    with open(file_path) as file:
        for line in tqdm(file, total=get_num_lines(file_path), desc="1) Parsing file "):
            sample = json.loads(line)
            reviewer_id = sample['reviewerID']
            asin = sample['asin']
            # in a case reviewname is not existing
            if 'reviewerName' in sample:
                reviewer_name = sample['reviewerName']
            else:
                reviewer_name = ''
            helpful = sample['helpful']
            text_raw = sample['reviewText']
            overall = sample['overall']
            summary = sample['summary']
            time = sample['reviewTime']
            # create Amazon Review object
            review = Review(category.value, reviewer_id, asin, reviewer_name, helpful, text_raw,
                            overall, summary, time)
            # add review object to set
            review_set.add(review)
        file.close()
    # Return review set.
    return review_set


def write_reviews_cleaned(amazoncategory, reviews_cleaned):
    """Writes cleaned reviews set into data/datasets/processed/*.pkl."""
    # Get file path to processed folder.
    file_path = get_processed_data_path(amazoncategory)
    # Write file into file path.
    with open(file_path, 'wb') as f:
        dump(reviews_cleaned, f)


def load_reviews_pkl(amazoncategory):
    """Load cleaned reviews into session."""
    file_path_ext = get_processed_data_path(amazoncategory)
    if os.path.exists(file_path_ext):
        with open(file_path_ext, 'rb') as f:
            return load(f)


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines
