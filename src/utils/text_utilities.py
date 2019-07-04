import pathlib
from typing import Optional

import PyPDF2
from PyPDF2 import PdfFileReader
from tika import parser


def pdf_to_text(pdf_path: pathlib.Path, password: str = None) -> Optional[str]:
    """
    Extract text from a PDF and return it, using the PyPDF2 package.

    Optionally, a password can be given.

    :param pdf_path: The path to the pdf file from which to extract the text.
    :param password: The password to use to decrypt the pdf.
    :return: The text extracted from the pdf.
        Can return None in case of error.
    """
    with open(pdf_path, 'rb') as f:
        try:
            pdf = PdfFileReader(f)
        except PyPDF2.utils.PdfReadError:
            return None
        if password and pdf.isEncrypted:
            pdf.decrypt(password)
        txt = ""
        for page in range(pdf.getNumPages()):
            txt += pdf.getPage(page).extractText() + "\n"
    return txt


def pdf_to_text_with_tika(pdf_location: str) -> Optional[str]:
    """
    Extract text from a PDF and return it, using the tika package.

    Note:
        - Works well with PDF created with tesseract.
        - On the first call to this function, it takes some time.

    :param pdf_location: The path to the pdf file from which to extract the text.
    :return: The text extracted from the pdf.
        Can return None in case of an internal error of the pdf conversion system.
        Can raise FileNotFoundError.
    """
    response = parser.from_file(pdf_location)

    if 'status' in response and response['status'] == 200:
        return response['content']
    else:
        return None
