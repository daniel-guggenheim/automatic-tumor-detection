# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
from src.preprocessing import data_preprocessing
from src.preprocessing.text_preprocessing import TextPreprocessing
from src.utils.utilities import save_pickle


def preprocess_poster_dataset(poster_folder: str, df_poster_filename: str):
    """
    Preprocess the poster folder, and returns a dataframe of cleaned text, so that it is ready to be
    used by the keyword classifier or the svm classifier.

    :param poster_folder: The path to the folder containing the poster dataset
    :param df_poster_filename: The name of the output path where the
    :return: A dataframe with one row per pdf in the poster folder, each row containing the path,
        the filename, the folder, the labo, the text and the cleaned text.
    """
    print('Converting all pdfs in poster folder into text')
    df_poster = data_preprocessing.tumor_pdf_folder_to_df(poster_folder)
    df_poster['y'] = df_poster.apply(lambda r: 1 if 'pos' in r.filename else 0, axis=1)
    print('Adding labo name to dataframe')
    df_poster = data_preprocessing.add_labo_to_df(df_poster)
    print('Preprocessing all poster texts')
    text_preprocessing = TextPreprocessing()
    df_poster = text_preprocessing.transform_df(df_poster)
    print('Finished. Saving dataframe at location: ' + df_poster_filename)
    save_pickle(df_poster, df_poster_filename)
    return df_poster


if __name__ == '__main__':
    inp = 'data/poster_data/'
    out = 'data/data_preprocessed/poster_df.p'

    preprocess_poster_dataset(poster_folder=inp, df_poster_filename=out)
