import itertools
import sys
from enum import Enum

from tqdm import tqdm

from src.processing.process_text.tokenization_nltk import filter_word_by_pos


class ListStream:
    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)


class Text_Element(Enum):
    S01_raw = 1
    S02_replace_whitespaces = 2
    S03_replace_multiple_stopwords = 3
    S04_replace_apostrophes = 4
    S05_expand_contractions = 5
    S06_remove_hyperlinks = 6
    S07_remove_special_characters = 7
    S08_remove_numbers = 8
    S09_convert_case = 9
    S10_expand_abbreviations = 10
    S11_sentence_tokenize = 11
    S12_remove_end_characters = 12
    S13_lemmatize = 13
    S14_remove_stopwords = 14
    S15_sentence_word_tokenize = 15
    SF_finalized = 16
    S16_direct_word_tokenize = 17


class Review(object):
    """Class represents an AMAZON REVIEW

    Attributes:
        reviewer_id: A unique identifier of the reviewer.
        asin: A a unique identifier fo the Product.
        helpful: ########################################################
        text_raw: The original raw text review.
        overall score: The review score (attributed by reviewer).
        summary: Review summary created by reviewer.
        time: timestamp.
    """

    def __init__(self, category, reviewer_id, asin, helpful, text_raw, overall, summary):
        """Initializes Amazon Review object with defined content."""
        self.category = category
        self.reviewer_id = reviewer_id
        self.asin = asin
        self.helpful = helpful
        self.text_raw = text_raw
        self.overall = overall
        self.summary = summary
        self.text_cleaned = None
        self.cleaning_log = dict()

    def __str__(self):
        """Creates user-friendly string representation of Amazon Review"""

        return "ReviewerID: {}\nProductID: {}\nHelpful: {}\n Overall: {}\nSummary: {}\nText:\n{}". \
            format(self.reviewer_id, self.asin, self.helpful,
                   self.overall, self.summary, self.text_raw)

    def __eq__(self, other):
        """
        Determines if review is equal to other review.

        Compares the review_ids values. Returns True if equal.
        """
        return (self.reviewer_id + self.asin) == (other.reviewer_id + other.asin)

    def __hash__(self):
        """
        Returns a unique hash value composed by revierID and ProductID.
        """
        return hash(self.reviewer_id + self.asin)

    def get_text_element(self, text_element):
        """Returns specified text element."""
        # If text_element is 1, return 01_raw.
        switcher = {1: '01_raw',
                    2: 'replace_whitespaces',
                    3: 'replace_multiple_stopwords',
                    4: 'replace_apostrophes',
                    5: 'expand_contractions',
                    6: 'remove_hyperlinks',
                    7: 'remove_special_characters',
                    8: 'remove_numbers',
                    9: 'convert_case',
                    10: 'expand_abbreviations',
                    11: 'sentence_tokenize',
                    12: 'remove_end_characters',
                    13: 'lemmatize',
                    14: 'remove_stopwords',
                    15: 'sentence_word_tokenize',
                    16: 'finalized',
                    17: 'direct_word_tokenize'}
        element_string = switcher.get(text_element.value, 'text element not found')
        # print(text_element.value)
        # print(element_string)
        # print('end switcher')
        if element_string == '01_raw':
            return self.text_raw
        elif element_string == 'finalized':
            return self.text_cleaned
        # If text_id is in keys of text_processed, return value from text_processes.
        if element_string in self.cleaning_log.keys():
            return self.cleaning_log[element_string]

    def count_number_of_words(self, text_id):
        """
        Counts the words in the Amazon Review.
        Retrieves the text from the Amazon Review, splits it by spaces and counts
        the length of the list.
        Returns:
            An integer value, corresponding to the number of words in the Amazon Review.
        """
        text_element = self.get_text_element(text_id)
        word_list = text_element.split()
        return len(word_list)

    def get_full_description(self):

        desc = "ReviewerID: {}\nProductID: {}\Helpful: {}\n Overall: {}\nSummary: {}". \
            format(self.reviewer_id, self.asin, self.helpful, self.overall, self.summary)
        desc += "Original Text :\n" + self.text_raw

        # note that clean text has been tokenized and hence is composed from lists)
        # redirect printer to string (use console print capability to depict list)
        sys.stdout = x = ListStream()
        print("\nFinal Text :")
        print(self.text_cleaned)
        for item in self.cleaning_log:
            print("\n" + item + ":")
            print(self.cleaning_log[item])
        # redirect printer to normal
        sys.stdout = sys.__stdout__
        # collect data
        desc += "".join(list(itertools.chain.from_iterable(x.data)))
        return desc


def create_review_from_dict(amazoncategory, review_dict):
    reviewer_id = review_dict.get('reviewerID')
    asin = review_dict.get('asin')
    helpful = review_dict.get('helpful')
    text_raw = review_dict.get('reviewText')
    overall = review_dict.get('overall')
    summary = review_dict.get('summary')
    # create Amazon Review object
    review = Review(amazoncategory.value, reviewer_id, asin, helpful, text_raw, overall, summary)
    # add review object to set
    return review


def extract_corpus(reviews, text_element, pos_tag):
    corpus = list()
    labels = list()
    for review in tqdm(reviews, desc="building corpus"):
        # We need to check if text elements are empty as per the cleaning process
        if type(review.get_text_element(text_element)) == list:
            print('cannot extract from tokenized texts, use other function instead, or choose other text_element')
            return None
        if not review.get_text_element(text_element) is None:
            if pos_tag == '':
                corpus.append(review.get_text_element(text_element))
                labels.append(review.category)
            else:
                temp_corpus = filter_word_by_pos(review.get_text_element(text_element), pos_tag, tokenize=False)
                if not temp_corpus is None:
                    corpus.append(temp_corpus)
                    labels.append(review.category)
    return corpus, labels


def extract_corpus_as_is(reviews, text_element):
    corpus = list()
    labels = list()
    for rev in tqdm(reviews, desc="building corpus (as is)"):
        if not rev.get_text_element(text_element) is None:
            corpus.append(rev.get_text_element(text_element))
            labels.append(rev.category)

    # corpus = [rev.get_text_element(text_element) for rev in tqdm(reviews, desc="building corpus (as is)") if
    # not rev.get_text_element(text_element) is None]
    return corpus, labels
