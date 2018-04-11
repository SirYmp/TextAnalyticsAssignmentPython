from src.utils.paths import AmazonCategory
from src.features.process_review.process_review_files import process_review_file
from src.data.io import test_json_reading, load_reviews_pkl

def main():

    # LOAD DATA; SAMPLE DATA; CLEAN; SAVE PICKLE FILE
    # Electronics
    reviewset1= process_review_file(AmazonCategory.amazon_electronics, 999, 100000)
    # Movies & TV
    reviewset2 = process_review_file(AmazonCategory.amazon_moviesandtv, 999, 100000)
    # Videogames
    reviewset3 = process_review_file(AmazonCategory.amazon_videogames, 999, 100000)
    # Kindle
    reviewset4 = process_review_file(AmazonCategory.amazon_kindle, 999, 100000)

    #actual_category = AmazonCategory.amazon_electronics
    #reviews = load_reviews_pkl(actual_category)
    #print(reviews[1].get_full_description())

if __name__ == '__main__':
    main()

