#!/usr/bin/env python
# coding: utf-8

import inflect
import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
import numpy as np

# Correctly generate plurals, singular nouns, ordinals, indefinite articles; convert numbers to words
inflect_engine = inflect.engine()


# partial matching function
def partial_match(x, y):
    return fuzz.partial_ratio(x, y)


partial_match_vector = np.vectorize(partial_match)


def _data_loading():
    data_url = 'http://files-as.intelligentreach.com/feedexports/25ebb945-5a4c-4a3c-99e5-3abea8ce47c0' \
               '/Habitat_UK_Particular_Audience_V2.xml '
    return pd.read_csv(data_url, delimiter='\t', dtype=str, header=0, error_bad_lines=False)


def _data_extraction(df):
    df_extracted = df[['link', 'google_product_category', 'title', 'product_type']].copy()
    df_extracted.dropna(axis=0, how='any', inplace=True)
    return df_extracted


def filter_nouns(row: str):
    punctuations = '?:!.,;>&^%$#'
    row_words = nltk.word_tokenize(row)
    return ' '.join(
        set([inflect_engine.singular_noun(word) if inflect_engine.singular_noun(word) else word for word in row_words if
             word not in punctuations and (not word.isdigit()) and wn.synsets(word, pos='n')])).lower()


def convert_cat(row: str):
    punctuations = '?:!.,;>&^%$#'
    row_words = nltk.word_tokenize(row)
    return ' '.join(
        set([inflect_engine.singular_noun(word) if inflect_engine.singular_noun(word) else word for word in row_words if
             word not in punctuations and (not word.isdigit())][1:])).lower()


def _data_transformation(df_extracted):
    df_transformed = df_extracted
    df_transformed['title_nouns'] = df_transformed['title'].copy().apply(filter_nouns)
    df_transformed['google_product_category_convert'] = \
        df_transformed['google_product_category'].copy().apply(convert_cat)
    df_transformed['product_type_convert'] = df_transformed['product_type'].copy().apply(convert_cat)
    df_transformed.dropna(axis=0, how='any', inplace=True)
    df_transformed['diff_rate'] = df_transformed.apply(
        lambda x: partial_match_vector(x['google_product_category_convert'], x['title_nouns']), axis=1)
    df_transformed['diff_rate_cat'] = df_transformed.apply(
        lambda x: partial_match_vector(x['google_product_category_convert'], x['product_type_convert']), axis=1)
    return df_transformed


def _data_output(df_transformed):
    df_transformed = df_transformed.sort_values(['diff_rate_cat'], ascending=True)
    df_transformed.to_csv("data_analysis_hb_cat_rate.csv")

    df_transformed = df_transformed.sort_values(['diff_rate'], ascending=True)
    df_transformed.to_csv("data_analysis_hb_title_rate.csv")


def main():
    df_loaded = _data_loading()
    df_extracted = _data_extraction(df_loaded)
    df_transformed = _data_transformation(df_extracted)
    _data_output(df_transformed)


if __name__ == "__main__":
    main()
