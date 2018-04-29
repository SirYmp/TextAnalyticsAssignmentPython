from src.processing.process_text.clean import clean_text, clean_text_reduced_dictionary


def clean_review(review):
    """Cleans text of review / logs changes to cleaning_log and final to text_cleaned."""
    review.cleaning_log, review.text_cleaned = clean_text(review.text_raw, wordtokenize=True)


def clean_review_reduced_dictionary(review):
    review.cleaning_log, review.text_cleaned = clean_text_reduced_dictionary(review.text_raw, wordtokenize=True)
