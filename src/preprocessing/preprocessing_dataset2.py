# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
from src.preprocessing import standard_preprocessing
from src.preprocessing.text_preprocessing import TextPreprocessing
from src.utils.utilities import save_pickle


def preprocess_dataset2(dataset2_folder: str, df2_filename: str):
    """
    Preprocess the dataset2 folder, and returns a dataframe of cleaned text, so that it is ready to be
    used by the keyword classifier or the svm classifier.

    :param dataset2_folder: The path to the folder containing the secondary dataset
    :param df2_filename: The name of the output path where the
    :return: A dataframe with one row per pdf in the dataset2 folder, each row containing the path,
        the filename, the folder, the labo, the text and the cleaned text.
    """
    df2 = standard_preprocessing.tumor_pdf_folder_to_df(dataset2_folder)
    df2['y'] = df2.apply(lambda r: 1 if 'pos' in r.filename else 0, axis=1)
    df2 = standard_preprocessing.add_labo_to_df(df2)
    text_preprocessing = TextPreprocessing()
    df2 = text_preprocessing.transform_df(df2)
    save_pickle(df2, df2_filename)
    return df2


if __name__ == '__main__':
    inp = 'data/dataset2/'
    out = 'data/data_preprocessed/df2.p'

    preprocess_dataset2(dataset2_folder=inp, df2_filename=out)
