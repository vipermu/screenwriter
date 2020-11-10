import re
from typing import *

punctuation = ['?', '!', '...', '-']


def remove_pagination(text: str,) -> str:
    """
    Removes pagination if exist from the input text.

    Args:
        text (str): text to processs.

    Returns:
        str: processed text.
    """
    processed_text = re.sub("^\d+\.", "", text)
    processed_text = re.sub("\d+\.$", "", processed_text)
    processed_text = re.sub("^\d+\ \ +", "", processed_text)
    processed_text = re.sub("\ \ +\d+", "", processed_text)

    return processed_text


def is_dialog(text: str,) -> str:
    """
    Recognizes wether the text is the character name or the start 
    of a dialog.

    Args:
        text (str): text to processs

    Returns:
        bool: True if the text is in fact the start of a dialog.
    """
    # TODO: improve me, I'm sketchy af.
    if text.isupper() and not text.endswith(":") \
            and not any([p in text for p in punctuation]):
        return True

    return False
