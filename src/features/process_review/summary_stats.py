import src.review.review


def calculate_average_words(reviews, text_element):
    total_word_count = 0
    total_reviews = 0
    for review in reviews:
        total_word_count += review.count_number_of_words(text_element)
        total_reviews += 1
    if total_reviews == 0:
        return 0
    else:
        return total_word_count / total_reviews
