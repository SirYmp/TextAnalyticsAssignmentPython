from os import pardir
from os.path import dirname, join, exists
from enum import Enum

# Relative path to data folder.
_SOURCE_DATA_FOLDER_PATH = join(dirname(__file__), pardir, pardir, "data\\sourcedata\\")
_PROCESSED_DATA_FOLDER_PATH = join(dirname(__file__), pardir, pardir, "data\\processed\\")
_DICTIONARY_DATA_FOLDER_PATH = join(dirname(__file__), pardir, pardir, "data\\sourcedata\\dictionaries")


class AmazonCategory(Enum):
    amazon_books = 1
    amazon_digitalmusic = 2
    amazon_kindle = 3
    amazon_videogames = 4
    amazon_cdsandvinyl = 5
    amazon_moviesandtv = 6
    amazon_electronics = 7
    amazon_beauty = 8
    amazon_clothing_shoes_jewel = 9


def get_source_file_name(amazoncategory):
    switcher = {1: 'Books_5.json',
                2: 'Digital_Music_5.json',
                3: 'Kindle_Store_5.json',
                4: 'Video_Games_5.json',
                5: 'CDs_and_Vinyl_5.json',
                6: 'Movies_and_TV_5.json',
                7: 'Electronics_5.json',
                8: 'Beauty_5.json',
                9: 'Clothing_Shoes_and_Jewelry_5.json'}
    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_processed_file_name(amazoncategory):
    switcher = {1: 'Books_5_Cleaned.pkl',
                2: 'Digital_Music_5_Cleaned.pkl',
                3: 'Kindle_Store_5_Cleaned.pkl',
                4: 'Video_Games_5_Cleaned.pkl',
                5: 'CDs_and_Vinyl_5_Cleaned.pkl',
                6: 'Movies_and_TV_5_Cleaned.pkl',
                7: 'Electronics_5_Cleaned.pkl',
                8: 'Beauty_5_Cleaned.pkl',
                9: 'Clothing_Shoes_and_Jewelry_5_Cleaned.pkl'}
    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_processed_training_file_name(amazoncategory):
    switcher = {1: 'Books_5_Cleaned_Training.pkl',
                2: 'Digital_Music_5_Cleaned_Training.pkl',
                3: 'Kindle_Store_5_Cleaned_Training.pkl',
                4: 'Video_Games_5_Cleaned_Training.pkl',
                5: 'CDs_and_Vinyl_5_Cleaned_Training.pkl',
                6: 'Movies_and_TV_5_Cleaned_Training.pkl',
                7: 'Electronics_5_Cleaned_Training.pkl',
                8: 'Beauty_5_Cleaned_Training.pkl',
                9: 'Clothing_Shoes_and_Jewelry_5_Cleaned_Training.pkl'}
    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_processed_test_file_name(amazoncategory):
    switcher = {1: 'Books_5_Cleaned_Test.pkl',
                2: 'Digital_Music_5_Cleaned_Test.pkl',
                3: 'Kindle_Store_5_Cleaned_Test.pkl',
                4: 'Video_Games_5_Cleaned_Test.pkl',
                5: 'CDs_and_Vinyl_5_Cleaned_Test.pkl',
                6: 'Movies_and_TV_5_Cleaned_Test.pkl',
                7: 'Electronics_5_Cleaned_Test.pkl',
                8: 'Beauty_5_Cleaned_Test.pkl',
                9: 'Clothing_Shoes_and_Jewelry_5_Cleaned_Test.pkl'}
    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_converted_file_name(amazoncategory):
    switcher = {1: 'Books_5_Parsed.pkl',
                2: 'Digital_Music_5_Parsed.pkl',
                3: 'Kindle_Store_5_Parsed.pkl',
                4: 'Video_Games_5_Parsed.pkl',
                5: 'CDs_and_Vinyl_5_Parsed.pkl',
                6: 'Movies_and_TV_5_Parsed.pkl',
                7: 'Electronics_5_Parsed.pkl',
                8: 'Beauty_5_Parsed.pkl',
                9: 'Clothing_Shoes_and_Jewelry_5_Parsed.pkl'}
    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_source_data_path(amazoncategory):
    return join(_SOURCE_DATA_FOLDER_PATH, get_source_file_name(amazoncategory))


def get_converted_data_path(amazoncategory):
    return join(_PROCESSED_DATA_FOLDER_PATH, get_converted_file_name(amazoncategory))


def get_processed_data_path(amazoncategory):
    return join(_PROCESSED_DATA_FOLDER_PATH, get_processed_file_name(amazoncategory))


def get_processed_training_data_path(amazoncategory):
    return join(_PROCESSED_DATA_FOLDER_PATH, get_processed_training_file_name(amazoncategory))


def get_processed_test_data_path(amazoncategory):
    return join(_PROCESSED_DATA_FOLDER_PATH, get_processed_test_file_name(amazoncategory))


def get_dictionaty_data_path(sub_path):
    """Returns path to file in data folder."""
    return join(_DICTIONARY_DATA_FOLDER_PATH, sub_path)
