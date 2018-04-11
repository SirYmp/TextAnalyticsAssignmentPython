from src.features.process_text.clean import clean_text


def clean_review(review):
    """Cleans text of review / logs changes to cleaning_log and final to text_cleaned."""
    review.cleaning_log, review.text_cleaned = clean_text(review.text_raw, wordtokenize=True)
