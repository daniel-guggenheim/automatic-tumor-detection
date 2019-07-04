# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
import string
from typing import List

import fr_core_news_sm as spacy_lang
import pandas
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin, BaseEstimator

from src.utils.utilities import now_time


class TextPreprocessing(TransformerMixin, BaseEstimator):
    """
    Clean french textual data using the spacy library.

    This is a preprocessing step that is useful for many machine learning models.
    """

    def __init__(self):
        self.nlp = spacy_lang.load()
        self.symbols = set(" ".join(string.punctuation + '0123456789' + '°').split(" "))
        self.stopwords = set(stopwords.words('french'))
        self.accepted_words = set(['pas'])
        self.pos_to_remove = ['PUNCT', 'SPACE', 'NUM', 'DET', 'PROPN']

    def transform(self, texts: List[str], **transform_params) -> List[str]:
        """
        Take an array of text, clean each text and return the result.

        :param pdf_texts: Array of text.
        :param transform_params: Other parameters (to be configured)
        :return:
        """
        clean_texts = []
        for txt in texts:
            clean_texts.append(self.__normalize_text(txt))
        return clean_texts

    def fit(self, X, y=None, **fit_params):
        return self

    def __normalize_text(self, text: str):
        """
        Use spacy to clean a french text and return all meaningful tokens joined by a space.

        Example:
            Input: "J'ai acheté une magnifique maison avec un grand jardin."
            Output: "acheter magnifique maison grand jardin"
        """
        doc = self.nlp(text, disable=['parser', 'ner'])
        tokens = []
        for w in doc:
            # Add token only if correct POS, not a stopword, not punctuation, OR in accepted words
            if w.pos_ not in self.pos_to_remove and not w.is_stop and w.lower_ not in self.stopwords and not any(
                    [c in self.symbols for c in w.lower_]) or w.lower_ in self.accepted_words:
                tokens.append(w.lemma_.lower())
        return ' '.join(tokens)

    def transform_df(self, df_texts: pandas.DataFrame, **transform_params) -> pandas.DataFrame:
        """
        Apply the transform method to a dataframe. Apply the transform method of this class
        to the text column of the df, and add the column "clean_text" with the result.

        :param df_texts: The dataframe to apply the transform method to.
        :param transform_params: Other parameters (to be configured)
        :return: A dataframe with the text cleaned.
        """
        df2 = df_texts.copy()
        df2['clean_text'] = self.transform(df_texts.text.tolist(), **transform_params)
        return df2
