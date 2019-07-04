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
    This class contains the SVM Model.

    A "split" class parameter allows to move the hyperplane of the results to accomodate for the
    sensitivity/specificity trade-off.
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

        Note: use the pred_split_position class attribute which allows to make some sensitivity/specificity
        tradeoffs.

        :param X: np.array of string of preprocessed text for which to predict the label
        :return: np.array of predicted label (1 or 0 integer) corresponding to the given text.
        """
        y_test_proba = self.pipeline.decision_function(X)
        return np.array([1 if p > self.pred_split_position else 0 for p in y_test_proba])

    def fit_on_2_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Train the model combining two datasets that have been preprocessed.

        :param df1: A dataset to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        :param df2: A second dataset to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        """
        concat = pd.concat([df1, df2], sort=False)
        concat = concat.sample(frac=1, random_state=self.random_state)
        self.fit(concat.clean_text.values, concat.y.values)

    def train_and_test_on_2_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame):
        """
        Train the model on part of df1 and df2, and test on the other part of this data. Print metrics of the results.

        Note: This function use a simple split, for more robust results cross-validation should be used.

        :param df1: A dataset to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        :param df2: A second dataset to use to train the model. Must have been preprocessed before, so the dataframe
            must contain the columns clean_text, and the column y.
        :return: test_df1_y_true, test_df1_y_pred, test_df2_y_true, test_df2_y_pred
            The results of the test on both the df1 and df2 dataset.
        """
        train_df1, test_df1 = train_test_split(df1, test_size=0.15, random_state=self.random_state)
        train_df2, test_df2 = train_test_split(df2, test_size=0.25, random_state=self.random_state,
                                                     stratify=df2.y)

        self.fit_on_2_datasets(train_df1, train_df2)
        test_df1_pred = self.predict(test_df1.clean_text.values)
        test_df2_pred = self.predict(test_df2.clean_text.values)

        print('---- Metrics for df1 data: ----\n')
        print_all_metrics(test_df1.y.values, test_df1_pred)
        print('\n' + '*' * 70 + '\n\n')
        print('----- Metrics for df2 data: ----\n')
        print_all_metrics(test_df2.y.values, test_df2_pred)
        return test_df1.y.values, test_df1_pred, test_df2.y.values, test_df2_pred

    def save_model(self, file_path: str):
        """
        Saves the instance (self) to a pickle file.

        :param file_path:
        :return:
        """
        save_pickle(self, file_path)
