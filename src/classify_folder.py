# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
"""
Methods to classify entire folders of PDF with the machine learning methods.
"""
from src.models.keyword_model import KeywordPreprocessing, KeywordModel
from src.preprocessing.data_preprocessing import tumor_pdf_folder_to_df
from src.preprocessing.text_preprocessing import TextPreprocessing
from src.utils.utilities import load_pickle


def classify_folder_of_pdfs_with_svm(data_folder: str, model_filename: str):
    """
    Classify a folder of tumor pdf files into positive or negative categories. Return a dataframe with the
    intermediate steps (text, cleaned text, etc) and the result.

    :param data_folder: Path to a folder containing only pdf files
    :param model_filename: The location of a pickle of a trained SvmModel instance.
    :return: a dataframe with one row per pdf in the given folder, each row containing the path, the predicted
        label (0 or 1 for positive or negative), the filename, the folder, the text and the cleaned text.
    """
    print('Loading model')
    svm_model = load_pickle(model_filename)
    print('Converting all pdfs in folder into text')
    df = tumor_pdf_folder_to_df(data_folder)
    print('Preprocessing all texts')
    text_preprocessing = TextPreprocessing()
    df = text_preprocessing.transform_df(df)
    print('Classifying all text with the svm model.')
    df['y_pred'] = svm_model.predict(df.clean_text.values)
    print('Finished. Returning dataframe of results')
    return df


def classify_folder_of_pdfs_with_keywords(data_folder: str,
                                          keyword_filename: str = 'data/keywords/terms_2016-2017.txt'):
    """
    Classify a folder of tumor pdf files into positive or negative categories using the keyword algorithm.
    Return a dataframe with the intermediate steps (text, cleaned text, etc) and the result.

    :param data_folder: Path to a folder containing only pdf files
    :param keyword_filename: The location of the list of keywords.
    :return: a dataframe with one row per pdf in the given folder, each row containing the path, the predicted
        label (0 or 1 for positive or negative), the filename, the folder, the text and the cleaned text.
    """
    print('Loading model')
    kw_model = KeywordModel(keyword_filename)
    print('Converting all pdfs in folder into text')
    df = tumor_pdf_folder_to_df(data_folder)
    print('Preprocessing all texts')
    text_preprocessing = KeywordPreprocessing()
    df = text_preprocessing.transform_df(df)
    print('Classifying all text with the svm model.')
    df['y_pred'] = kw_model.predict(df.kw_clean_text.values)
    print('Finished. Returning dataframe of results')
    return df
