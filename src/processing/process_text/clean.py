from src.processing.process_text.lemmatize_nltk import lemmatize_text
from src.processing.process_text.normalize import expand_contractions, remove_special_characters, \
    remove_stopwords, remove_end_characters, convert_case, remove_hyperlinks, replace_whitespaces, \
    replace_apostrophes, replace_multiple_stopwords, remove_numbers, expand_abbreviations, correct_pontuation
from src.processing.process_text.tokenization_nltk import sentence_tokenize, word_tokenize, \
    convert_tokens_to_string_of_words


def clean_text(text, wordtokenize=False):
    clean_dict = {}
    # Replace multiple whitespaces.
    clean_dict['replace_whitespaces'] = replace_whitespaces(text)
    # Replace multiple stopwords.
    clean_dict['replace_multiple_stopwords'] = replace_multiple_stopwords(clean_dict['replace_whitespaces'])
    # Replace apostrophes.
    clean_dict['replace_apostrophes'] = replace_apostrophes(clean_dict['replace_multiple_stopwords'])
    # Expand contractions.
    clean_dict['expand_contractions'] = expand_contractions(clean_dict['replace_apostrophes'])
    # Remove hyperlinks.
    clean_dict['remove_hyperlinks'] = remove_hyperlinks(clean_dict['expand_contractions'])
    # Remove special characters.
    clean_dict['remove_special_characters'] = remove_special_characters(clean_dict['remove_hyperlinks'])
    # Remove numbers.
    clean_dict['remove_numbers'] = remove_numbers(clean_dict['remove_special_characters'])
    # Convert to lower case.
    clean_dict['convert_case'] = convert_case(clean_dict['remove_numbers'])
    # Expand abbreviations.
    clean_dict['expand_abbreviations'] = expand_abbreviations(clean_dict['convert_case'])
    # Tokenize sentences.
    temp_sentence = correct_pontuation(clean_dict['expand_abbreviations'])
    temp_sentence = replace_whitespaces(temp_sentence)
    clean_dict['sentence_tokenize'] = sentence_tokenize(temp_sentence)
    # If sentence tokenize is empty, return None.
    if not clean_dict['sentence_tokenize']:
        return clean_dict, None
    else:
        # Remove end characters.
        clean_dict['remove_end_characters'] = [remove_end_characters(item) for item in clean_dict['sentence_tokenize'] if len(item) > 1]
        # Lemmatize words.
        clean_dict['lemmatize'] = [lemmatize_text(item) for item in clean_dict['remove_end_characters'] if len(item) > 1]
        # Remove stopwords.
        clean_dict['remove_stopwords'] = [remove_stopwords(item) for item in clean_dict['lemmatize'] if len(item) > 1]
        # If tokenize, tokenize words.
        if wordtokenize:
            clean_dict['sentence_word_tokenize'] = [word_tokenize(item, 'whitespace') for item in
                                                    clean_dict['remove_stopwords'] if len(item) > 1]
        # Return dictionary and cleaned text.
        non_tokenized_result = convert_tokens_to_string_of_words(clean_dict['sentence_word_tokenize'])
        clean_dict['direct_word_tokenize'] = word_tokenize(non_tokenized_result, 'whitespace')
        return clean_dict, non_tokenized_result


def clean_text_reduced_dictionary(text, wordtokenize=False):
    clean_dict, non_tokenized_result = clean_text(text, wordtokenize)
    del clean_dict['replace_whitespaces']
    del clean_dict['replace_multiple_stopwords']
    del clean_dict['replace_apostrophes']
    del clean_dict['expand_contractions']
    del clean_dict['remove_hyperlinks']
    del clean_dict['remove_special_characters']
    del clean_dict['remove_numbers']
    del clean_dict['convert_case']
    del clean_dict['expand_abbreviations']
    return clean_dict, non_tokenized_result
