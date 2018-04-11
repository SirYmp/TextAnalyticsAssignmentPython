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
    amazon_electronics =7


def get_source_file_name(amazoncategory):

    switcher={1:'books\\Video_Games_5.json',
              2:'digitalmusic\\Digital_Music_5.json',
              3: 'kindle\\Kindle_Score_5.json',
              4: 'videogames\\Video_Games_5.json',
              5: 'cdsandvinyl\\CDs_and_Vinyl_5.json',
              6: 'moviesandtv\\Movies_and_TV_5.json',
              7: 'electronics\\Electronics_5.json'}

    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_processed_file_name(amazoncategory):

    switcher={1:'books\\Video_Games_5_Cleaned.pkl',
              2:'digitalmusic\\Digital_Music_5_Cleaned.pkl',
              3: 'kindle\\Kindle_Score_5_Cleaned.pkl',
              4: 'videogames\\Video_Games_5_Cleaned.pkl',
              5: 'cdsandvinyl\\VCDs_and_Vinyl_5_Cleaned.pkl',
              6: 'moviesandtv\\Movies_and_TV_5_Cleaned.pkl',
              7: 'electronics\\Electronics_5_Cleaned.pkl'}

    return switcher.get(amazoncategory.value, 'wrongly defined')


def get_source_data_path(amazoncategory):
    return join(_SOURCE_DATA_FOLDER_PATH, get_source_file_name(amazoncategory))


def get_processed_data_path(amazoncategory):
    return join(_PROCESSED_DATA_FOLDER_PATH, get_processed_file_name(amazoncategory))


def get_dictionaty_data_path(sub_path):
    """Returns path to file in data folder."""
    return join(_DICTIONARY_DATA_FOLDER_PATH, sub_path)

