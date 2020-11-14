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

def remove_manual_conds(text: str,) -> str:
    """
    Removes sentences that match the ones hardcoded in this function.

    Args:
        text (str): text to processs.

    Returns:
        str: processed text.
    """
    blacklist_list = [
        "continue", 
        "continue:", 
        "(continue)", 
        "continued", 
        "continued:", 
        "(continued)", 
        "cut to:",
        ]

    blacklist_cond = text.lower() in blacklist_list

    if blacklist_cond:
        return ""
    else:
        return text

def is_dialog(text: str,) -> str:
    """
    Recognizes wether the text is the character name or the start 
    of a dialog.

    Args:
        text (str): text to processs

    Returns:
        bool: True if the text is in fact the start of a dialog.
    """
    parenthesis_cond = text.startswith("(") and text.endswith(")")

    # TODO: improve me, I'm sketchy af.
    if text.isupper() \
            and not text.endswith(":") \
            and not any([p in text for p in punctuation]) \
            or parenthesis_cond:
        return True

    return False
