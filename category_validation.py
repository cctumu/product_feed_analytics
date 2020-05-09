#!/usr/bin/env python
# coding: utf-8
import os

import inflect
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
import numpy as np
import logging

align_product_category_hb = {
    'Home & Garden > Household Supplies > Waste Containment > Rubbish Bins & Waste Paper Baskets': 'Home & Garden > Household Supplies > Waste Containment > Trash Cans & Wastebaskets',
    'Home & Garden > Household Supplies > Storage & Organisation': 'Home & Garden > Household Supplies > Storage & Organization',
    'Home & Garden > Linens & Bedding > Towels > Bath Towels & Flannels': 'Home & Garden > Linens & Bedding > Towels > Bath Towels & Washcloths',
    'Clothing & Accessories > Clothing > Nightwear & Loungewear > Robes': 'Apparel & Accessories > Clothing > Sleepwear & Loungewear > Robes',
    'Home & Garden > Kitchen & Dining > Tableware > Coffee & Tea Pots': 'Home & Garden > Kitchen & Dining > Tableware > Coffee Servers & Tea Pots',
    'Home & Garden > Linens & Bedding > Table Linen': 'Home & Garden > Linens & Bedding > Table Linens',
    'Home & Garden > Kitchen & Dining > Tableware > Cutlery': 'Home & Garden > Kitchen & Dining > Tableware > Flatware',
    'Home & Garden > Linen > Table Linen > Placemats': 'Home & Garden > Linens & Bedding > Table Linens > Placemats',
    'Furniture > Cabinets & Storage > Cupboards & Wardrobes': 'Furniture > Cabinets & Storage > Armoires & Wardrobes',
    'Home & Garden > Decor > Artwork > Posters, Prints & Visual Artwork': 'Home & Garden > Decor > Artwork > Posters, Prints, & Visual Artwork',
    'Furniture > Benches > Storage Benches': 'Furniture > Benches > Storage & Entryway Benches',
    'Home & Garden > Linens & Bedding > Bedding > Quilts & Duvets': 'Home & Garden > Linens & Bedding > Bedding > Quilts & Comforters'
}


def set_logger():
    # create logger
    log = logging.getLogger('my_app')
    log.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    log.addHandler(ch)

    # 'application' code
    # log.debug('debug message')
    # log.info('info message')
    # log.warning('warn message')
    # log.error('error message')
    # log.critical('critical message')
    return log


logger = set_logger()

# Correctly generate plurals, singular nouns, ordinals, indefinite articles; convert numbers to words
inflect_engine = inflect.engine()


def partial_match(x, y):
    """
    Fuzzy partial matching function for 2 strings.

    :param x: string 1 to be matched
    :param y: string 2 to be matched
    :return: matching rate
    """
    return fuzz.partial_ratio(x, y)


partial_match_vector = np.vectorize(partial_match)


def _data_loading():
    data_url = 'http://files-as.intelligentreach.com/feedexports/25ebb945-5a4c-4a3c-99e5-3abea8ce47c0' \
               '/Habitat_UK_Particular_Audience_V2.xml '
    return pd.read_csv(data_url, delimiter='\t', dtype=str, header=0, error_bad_lines=False)


def _data_extraction(df):
    """
    Extracting data.

    :param df: input dataframe
    :return: extracted dataframe
    """
    df_extracted = df[['mpn', 'link', 'google_product_category', 'title', 'product_type']].copy()
    df_extracted.columns = ['sku_id', 'link', 'product_category', 'title', 'product_type']
    df_extracted.dropna(axis=0, how='any', inplace=True)

    df_extracted = _join_google_category(df_extracted)
    return df_extracted


def __read_google_product_category():
    file_path = f'data/Google_Product_Category.csv'

    df = pd.read_csv(file_path, delimiter=',', dtype=str, header=None,
                     names=['product_category_id', "col1", "col2", "col3", "col4", "col5", "col6", "col7"])

    df["product_category"] = df.apply(
        lambda row:
        ' > '.join(
            [v for v in [row['col1'], row['col2'], row['col3'], row['col4'], row['col5'], row['col6'], row['col7']] if
             v is not np.nan]), axis=1)

    # print(df)

    return df[['product_category_id', 'product_category']]


def _join_google_category(df_converted):
    for k, v in align_product_category_hb.items():
        df_converted["product_category"] = df_converted.apply(
            lambda row: v if (row["product_category"] == k) else row["product_category"], axis=1)

    df_converted = df_converted.merge(__read_google_product_category(),
                                      on=['product_category'], how='left')

    return df_converted


def filter_nouns(row: str):
    """
    Filter nouns from a string.

    :param row: string to be filtered
    :return: filtered string
    """
    punctuations = '?:!.,;>&^%$#'
    row_words = nltk.word_tokenize(row)
    return ' '.join(
        set([inflect_engine.singular_noun(word) if inflect_engine.singular_noun(word) else word for word in row_words if
             word not in punctuations and (not word.isdigit()) and wn.synsets(word, pos='n')])).lower()


def convert_cat(row: str):
    """
    Filter nouns from a string without first cat.

    :param row: string to be filtered
    :return: filtered string
    """
    punctuations = '?:!.,;>&^%$#'
    row_words = nltk.word_tokenize(row)
    return ' '.join(
        set([inflect_engine.singular_noun(word) if inflect_engine.singular_noun(word) else word for word in row_words if
             word not in punctuations and (not word.isdigit())][1:])).lower()


def _data_transformation(df_extracted):
    """
    Generating fuzzy partial matching rates for (category vs title) & (category vs title).

    :param df_extracted: extracted dataframe
    :return: transformed dataframe
    """
    df_transformed = df_extracted
    df_transformed['title_nouns'] = df_transformed['title'].copy().apply(filter_nouns)
    df_transformed['product_category_convert'] = \
        df_transformed['product_category'].copy().apply(convert_cat)
    df_transformed['product_type_convert'] = df_transformed['product_type'].copy().apply(convert_cat)
    df_transformed.dropna(axis=0, how='any', inplace=True)
    df_transformed['diff_rate'] = df_transformed.apply(
        lambda x: partial_match_vector(x['product_category_convert'], x['title_nouns']), axis=1)
    df_transformed['diff_rate_cat'] = df_transformed.apply(
        lambda x: partial_match_vector(x['product_category_convert'], x['product_type_convert']), axis=1)
    return df_transformed


def _data_output(df_transformed):
    """
    Sort dataframe by partial matching rates and output to csv.

    :param df_transformed: dataframe transformed
    """
    output_dir = 'output_data'
    output_file_title = f"{output_dir}/data_analysis_hb_title_rate.csv"
    output_file_cat = f"{output_dir}/data_analysis_hb_cat_rate.csv"
    output_path = os.getcwd()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df_transformed = df_transformed.sort_values(['diff_rate_cat'], ascending=True)
    df_transformed.to_csv(output_file_cat, index=False)

    df_transformed = df_transformed.sort_values(['diff_rate'], ascending=True)
    df_transformed.to_csv(output_file_title, index=False)

    logger.info(f'Output to {output_path}\\{output_file_cat} and {output_path}\\{output_file_title}.')


def main():
    df_loaded = _data_loading()
    df_extracted = _data_extraction(df_loaded)
    df_transformed = _data_transformation(df_extracted)
    _data_output(df_transformed)


if __name__ == "__main__":
    main()
