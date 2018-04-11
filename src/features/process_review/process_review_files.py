import random
from tqdm import tqdm
import sys
from src.data.io import create_reviews_set, write_reviews_cleaned
from src.features.process_review.clean_review import clean_review

def process_review_file (amazoncategory, sampleseed, samplesize):
    reviews = create_reviews_set(amazoncategory)
    # generate sample
    random.seed(sampleseed)
    sampled_reviews = random.sample(reviews, samplesize)
    # clean sample
    for review in tqdm(sampled_reviews, total=samplesize, desc="3) Cleaning / Normalizing:"):
        clean_review(review)
    # write processed file
    write_reviews_cleaned(amazoncategory, sampled_reviews)
    # flush tqdm channel to avoid conflicts and garbage prints
    sys.stderr.flush
    return reviews