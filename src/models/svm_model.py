# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.ml_utilities import print_all_metrics
from src.utils.utilities import save_pickle


class SvmModel(BaseEstimator, TransformerMixin):
    """
    This class contains the SVM Model with all its parameters.
    It allows to train on the 2016 and poster datasets, and to predict using a split position
    parameter.
    """

    def __init__(self, pred_split_position: int = 0, pipeline: Pipeline = None, random_state: int = 43):
        self.pred_split_position = pred_split_position
        self.random_state = random_state
        if pipeline is None:
            self.pipeline = Pipeline([
                ('vectorizer', CountVectorizer(ngram_range=(1, 1), max_features=5000, binary=True)),
                ('classifier', LinearSVC(C=0.1, max_iter=50000))])

    def fit(self, X: np.ndarray, y: np.ndarray = None, **fit_params):
        """
        Train the model with the given X (data) and y (labels).

        :param X: np.array of preprocessed text to train the model on
        :param y: np.array of true label (1 or 0) corresponding to the given text
        :param fit_params: Optionnal parameters to give the the vectorizer and classifer
        :return: self : Pipeline
            This estimator
        """
        return self.pipeline.fit(X, y, **fit_params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the labels of an array of text.

        Note: use the "split position" parameter which allows to make some sensitivity/specificity
        tradeoffs.

        :param X: np.array of string of preprocessed text for which to predict the label
        :return: np.array of predicted label (1 or 0 integer) corresponding to the given text.
        """
        y_test_proba = self.pipeline.decision_function(X)
        return np.array([1 if p > self.pred_split_position else 0 for p in y_test_proba])

    def fit_on_2016_and_poster_data(self, df_2016: pd.DataFrame, df_poster: pd.DataFrame):
        """
        Train the model combining the 2016 data + the poster data that have been preprocessed.
        This is the command that should be used if this model is to be used in production.

        Note that you should not test this model on the 2016 data or the poster data after executing this function
        (as this has overfitted the data).

        :param df_2016: The 2016 data to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        :param df_poster: The poster data to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        """
        concat = pd.concat([df_2016, df_poster], sort=False)
        concat = concat.sample(frac=1, random_state=self.random_state)
        self.fit(concat.clean_text.values, concat.y.values)

    def train_and_test_on_2016_and_poster_data(self, df_2016: pd.DataFrame, df_poster: pd.DataFrame):
        """
        Train the model on part of the 2016 and poster data, and test on the other part of this data.
        Print metrics of the results.

        Note: This function use a simple split, for more robust results cross-validation should be used.

        :param df_2016: The 2016 data to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        :param df_poster: The poster data to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        :return: test_2016_y_true, test_2016_y_pred, test_poster_y_true, test_poster_y_pred
            The results of the test on both the 2016 and poster dataset.
        """
        train_2016, test_2016 = train_test_split(df_2016, test_size=0.15, random_state=self.random_state)
        train_poster, test_poster = train_test_split(df_poster, test_size=0.25, random_state=self.random_state,
                                                     stratify=df_poster.y)

        self.fit_on_2016_and_poster_data(train_2016, train_poster)
        test_2016_pred = self.predict(test_2016.clean_text.values)
        test_poster_pred = self.predict(test_poster.clean_text.values)
        print('---- Metrics for 2016 data: ----\n')
        print_all_metrics(test_2016.y.values, test_2016_pred)
        print('\n' + '*' * 70 + '\n\n')
        print('----- Metrics for poster data: ----\n')
        print_all_metrics(test_poster.y.values, test_poster_pred)
        return test_2016.y.values, test_2016_pred, test_poster.y.values, test_poster_pred

    def save_model(self, file_path: str):
        """
        Saves the instance (self) to a pickle file.

        :param file_path:
        :return:
        """
        save_pickle(self, file_path)
