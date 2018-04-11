import itertools
import sys


class ListStream:
    def __init__(self):
        self.data = []

    def write(self, s):
        self.data.append(s)


class Review(object):
    """Class represents an AMAZON REVIEW

    Attributes:
        reviewer_id: A unique identifier of the reviewer.
        asin: A a unique identifier fo the Product.
        reviewer_name: The name of the reviewer.
        helpful_rating: ########################################################
        text_raw: The original raw text review.
        overall score: The review score (attributed by reviewer).
        summary: Review summary created by reviewer.
        ux_review_time: unix timestamp.
        review_time: timestamp.
    """

    def __init__(self, category, reviewer_id, asin, reviewer_name, helpful, text_raw, overall, summary,
                 time):
        """Initializes Amazon Review object with defined content."""
        self.category = category
        self.reviewer_id = reviewer_id
        self.asin = asin
        self.reviewer_name = reviewer_name
        self.helpful = helpful
        self.text_raw = text_raw
        self.overall = overall
        self.summary = summary
        self.time = time
        self.text_cleaned = None
        self.cleaning_log = dict()

    def __str__(self):
        """Creates user-friendly string representation of Amazon Review"""

        return "ReviewerID: {}\nProductID: {}\nReviewerName: {}\nCreated: {}\nHelpful: {}\n Overall: {}\nSummary: {}\nText:\n{}". \
            format(self.reviewer_id, self.asin, self.reviewer_name, self.time, self.helpful,
                   self.overall, self.summary, self.text_raw)

    def __eq__(self, other):
        """
        Determines if review is equal to other review.

        Compares the review_ids values. Returns True if equal.
        """
        return (self.reviewer_id + self.asin) == (other.revier_id + other.asin)

    def __hash__(self):
        """
        Returns a unique hash value composed by revierID and ProductID.
        """
        return hash(self.reviewer_id + self.asin)

    def get_text_element(self, text_id):
        """Returns specified text element."""
        # If text_id is 01_raw, return text.
        if text_id == '01_raw':
            return self.text_raw
        # If text_id is in keys of text_processed, return value from text_processes.
        if text_id in self.cleaning_log.keys():
            return self.cleaning_log[text_id]

    def count_number_of_words(self):
        """
        Counts the words in the Amazon Review.

        Retrieves the text from the Amazon Review, splits it by spaces and counts
        the length of the list.

        Returns:
            An integer value, corresponding to the number of words in the Amazon Review.
        """
        word_list = self.text_raw.split()
        return len(word_list)

    def get_full_description(self):

        desc = "ReviewerID: {}\nProductID: {}\nReviewerName: {}\nCreated: {}\nHelpful: {}\n Overall: {}\nSummary: {}\n". \
            format(self.reviewer_id, self.asin, self.reviewer_name, self.time, self.helpful,
                   self.overall, self.summary)
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
    reviewer_name = review_dict.get('reviewerName', '')
    helpful = review_dict.get('helpful')
    text_raw = review_dict.get('reviewText')
    overall = review_dict.get('overall')
    summary = review_dict.get('summary')
    time = review_dict.get('reviewTime')
    # create Amazon Review object
    review = Review(amazoncategory.value, reviewer_id, asin, reviewer_name, helpful, text_raw,
                    overall, summary, time)
    # add review object to set
    return review
