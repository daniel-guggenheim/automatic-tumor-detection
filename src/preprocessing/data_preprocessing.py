# Author: Daniel Guggenheim
# Company: unisante, iumsp
# Date: 03/2019
# ===================================================
"""
This file contain utility functions for the first step of preprocessing: pdf text extraction into a dataframe.

It contains also function to add information (find the lab name in the file).
"""

import os
import pathlib
from typing import Optional, Tuple, List

import pandas as pd

from src.preprocessing.text_preprocessing import TextPreprocessing
from src.utils.text_utilities import pdf_to_text_with_tika, pdf_to_text
from src.utils.utilities import now_time, load_pickle, save_pickle


def tumor_pdf_folder_to_df(data_folder: str, incremental_save_file=None, pdf_password: str = "tvr16") -> pd.DataFrame:
    """
    Go through each pdfs in a folder recursively and extract the text. Return a dataframe with the text
    corresponding to each pdf, the path to the file, the filename and the folder.

    Use as pdf parser the tika library first, and in case of failure the PyPDF2 parser. (This choice was made because
    it seems that PyPDF2 doesn't always detect spaces.)

    :param data_folder: The folder which contains the pdfs.
    :param incremental_save_file: A file where to make an incremental save of the text, if the command has to be launched multiple times
    :param pdf_password: Password to use for protected pdfs.
    :return: Dataframe with 1 pdf per line, with as column: [text, path, filename, folder]
    """
    # List[Dict] each element is a pdf file, with the future column elements of the dataframe in a dict.
    df_list = []
    if incremental_save_file is not None:
        try:
            df_list = load_pickle(incremental_save_file)
        except FileNotFoundError:
            pass

    def is_pdf_file(f):
        return f[-4:] == '.pdf' and f[0] != '.'

    nb_files = sum([len([f for f in files if is_pdf_file(f)]) for r, d, files in os.walk(data_folder)])
    file_count = 0

    # Go through each pdf in all subfolders, and extract the text
    for i, (path, subfolders, files) in list(enumerate(os.walk(data_folder))):
        for pdf_filename in files:
            if is_pdf_file(pdf_filename):
                file_count += 1
                if file_count % max(1, int(nb_files / 20)) == 0:
                    print(f'{now_time()}  {file_count / nb_files:.0%}')
                    if incremental_save_file is not None:
                        save_pickle(df_list, incremental_save_file)
                pdf_path = pathlib.Path(path, pdf_filename)
                pdf_text = pdf_to_text_with_tika(pdf_path.as_posix())
                if pdf_text is None:  # parsing failed with tika
                    pdf_text = pdf_to_text(pdf_path, password=pdf_password)
                if pdf_text is None:  # (because the parsing can fail with tika AND pypdf2)
                    pdf_text = ''
                df_list.append(
                    {'text': pdf_text, 'path': pdf_path, 'filename': pdf_path.name, 'folder': pdf_path.parent})

    if incremental_save_file is not None:
        save_pickle(df_list, incremental_save_file)
    return pd.DataFrame(df_list)


def pdf_folder_to_clean_df(data_folder: str) -> pd.DataFrame:
    """
    Go through each pdfs in a folder recursively, extract the text and clean it. Return a dataframe with the text
    corresponding to each pdf, the clean text, the path to the file, the filename and the folder.

    :param data_folder: The folder which contains the pdfs.
    :return: Dataframe with 1 pdf per line, with as column: [text, clean_text, path, filename, folder]
    """
    df = tumor_pdf_folder_to_df(data_folder)
    text_preprocessing = TextPreprocessing()
    df = text_preprocessing.transform_df(df)
    return df


def find_labo_from_pdf_text(pdf_text: str, labs_pattern: Optional[List[Tuple[str, str]]] = None) -> List[str]:
    """
    Find the lab a tumor analysis pdf text is coming from.

    :param pdf_text: The text of the pdf file
    :param labs_pattern: tuple(lab_name, pattern): How to recognize a lab (multiple elements per lab allowed)
    :return: List lab that was detected with the pattern
    """
    if labs_pattern is None:
        labs_pattern = [('a', 'anonymous1'),
                        ('b', 'anonymous2'),
                        ('c', 'anonymous3')
                        ]
    found_labo = set()

    for labo_name, pattern in labs_pattern:
        if pattern in pdf_text:
            found_labo.add(labo_name)
    return list(found_labo)


def add_labo_to_df(df: pd.DataFrame, labs_pattern: Optional[List[Tuple[str, str]]] = None):
    """
    Find all lab in a dataframe and add a column to it with the name of the lab.
    """

    def find_labo(row):
        found_labo = find_labo_from_pdf_text(row.text, labs_pattern)
        if len(found_labo) == 0:
            return None
        if len(found_labo) > 1:
            print(f"Found too many labo at index {row.name}: {found_labo}.")
        return found_labo[0]

    df = df.copy()
    df['labo'] = df.apply(find_labo, axis=1)
    return df
