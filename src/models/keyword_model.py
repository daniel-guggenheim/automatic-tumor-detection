# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
import re
from typing import List, Dict

import pandas
from sklearn.base import TransformerMixin, BaseEstimator


class KeywordModel(BaseEstimator, TransformerMixin):
    """
    Model to detect positive and negative pdfs with keywords.
    """

    def __init__(self, keyword_filename='data/keywords/terms_2016-2017.txt'):
        self.keywords = self.__load_keywords(keyword_filename)

    def _compute_keywords_prediction(self, pdf_texts: List[str]) -> List[int]:
        """
        If a clean pdf text contains a keyword coming from the keyword list, then
        the document is classified as positive (1), otherwise as negative (0).
        :param pdf_texts: The cleaned texts to classify (clean means they must have been through
            the KeywordPreprocessing model.
        :return: A list of 0 or 1, classifying each text in pdf_texts as positive or negative.
        """
        y_pred = []
        for txt in pdf_texts:
            pred = 0
            for kw in self.keywords:
                if kw in txt:
                    pred = 1
            y_pred.append(pred)
        return y_pred

    def fit(self, X, y=None, **fit_params):
        """
        Does nothing, as this model does not need to be trained.
        """
        return self

    def predict(self, cleaned_pdf_texts: List[str]) -> List[int]:
        return self._compute_keywords_prediction(cleaned_pdf_texts)

    @staticmethod
    def __load_keywords(filename: str) -> List[str]:
        """
        Load the list of keyword from a file, with 1 keyword per line.

        :param filename: Keyword filename
        :return: List of keyword
        """
        with open(filename) as f:
            return f.read().splitlines()


class KeywordPreprocessing(BaseEstimator, TransformerMixin):
    """
    Clean the data according to the keyword classification technique used by the tumor registry before.
    """

    CHAR_REPLACEMENT = {
        'e': ['é', 'è', 'ê', 'ë'],
        'a': ['à', 'â', 'ä'],
        'u': ['ù', 'û', 'ü'],
        'i': ['î', 'ï'],
        'o': ['ô', 'ö'],
        'c': ['ç'],
        'oe': ['œ'],
        'ae': ['æ'],
        ' ': ['&#', '\t', '\r', '\n', ',', '(', ')', '.', ';', '-', '.', '/'],
    }

    def __init__(self, replacements=CHAR_REPLACEMENT):
        self.replacements = replacements

    def transform(self, pdf_texts: List[str], **transform_params) -> List[str]:
        return [self.__normalize_text(txt) for txt in pdf_texts]

    def transform_df(self, df_texts: pandas.DataFrame, **transform_params) -> pandas.DataFrame:
        """
        Apply the transform method to a dataframe. Apply the transform method of this class
        to the text column of the df, and add the column "clean_text" with the result.

        :param df_texts: The dataframe to apply the transform method to.
        :param transform_params: Other parameters (to be configured)
        :return: A dataframe with the text cleaned.
        """
        df2 = df_texts.copy()
        df2['kw_clean_text'] = self.transform(df_texts.text.tolist(), **transform_params)
        return df2

    def fit(self, X, y=None, **fit_params):
        return self

    def __normalize_text(self, txt: str) -> str:
        txt = txt.lower()
        txt = self.__replace_multiple(self.replacements, txt)
        txt = " ".join(txt.split())
        return txt

    @staticmethod
    def __replace_multiple(replacement_expressions: Dict[str, List[str]], text: str) -> str:
        """
        replacement_expressions: is the desired replacements. It must be in  the
        form: {"replacement": ["condition1", "condition2"], "otherRepl": ["condition3"]}

        Example:
            ````
                >>> replacement_expressions = {'e': ['é', 'è', 'ê', 'ë'], 'a': ['à', 'â', 'ä']}
                >>> text = "Le château était à l'héritier."
                >>> replace_multiple(replacement_expressions, text)
                "Le chateau etait a l'heritier.""
            ```
        """
        rep = {}
        for replacement, cond_list in replacement_expressions.items():
            for cond in cond_list:
                rep[cond] = replacement
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
