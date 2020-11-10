import os
import glob
import pickle
import logging
from typing import *

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer

from screenwriter.data_utils import remove_pagination, is_dialog

logger = logging.getLogger("screenwriter.datasets")


class ScreenwriterData(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        data_dir: Optional[str] = "./scraper/extracted-data",
        out_dir: Optional[str] = "./screenwriter/data",
        batch_size: Optional[int] = 1,
        block_size: Optional[int] = 512,
        recompute: Optional[bool] = False,
    ) -> None:
        txt_file_path_list = glob.glob(os.path.join(data_dir, "*.txt"))

        self.block_token_list = []
        for txt_file_path in txt_file_path_list:
            if 'test' in txt_file_path:
                continue

            pkl_filename = txt_file_path.split("/")[-1].split(".")[-2] + "-tokens.pkl"
            cache_file_path = os.path.join(out_dir, pkl_filename)

            if os.path.exists(cache_file_path) and not recompute:
                logger.info(f"Loading cached data from {cache_file_path}...")
                with open(cache_file_path, "rb") as cache_file:
                    tokenized_lines = pickle.load(cache_file)

            else:
                logger.info(f"Processing {txt_file_path}...")
                with open(txt_file_path) as txt_file:
                    text_lines = txt_file.readlines()
                    _processed_lines, tokenized_lines = self.process_lines(
                        text_lines=text_lines,
                        tokenizer=tokenizer,
                    )

                logger.info(
                    f"Storing processed tokens in {cache_file_path}...")
                with open(cache_file_path, "wb") as cache_file:
                    pickle.dump(tokenized_lines, cache_file)

            logger.info(f"Creating blocks of {block_size} tokens...")

            cached_tokenized_line = []
            for tokenized_line in tokenized_lines:
                cached_tokenized_line.extend(tokenized_line)

                if len(cached_tokenized_line) >= block_size:
                    block_tokenized_line = cached_tokenized_line[0:block_size]
                    self.block_token_list.append(block_tokenized_line)

                    cached_tokenized_line = tokenized_line

    def process_lines(
        self,
        text_lines: List[str],
        tokenizer: GPT2Tokenizer,
    ) -> Tuple[List[str], List[int]]:
        """
        Process a list of lines with the following rules:
        - If line is blank: pass, do nothing.
        - If line contain text:
            - If text is not dialog: append the current 
                line and continue.
            - Else:
                1. Iterate and append to the current sentence until 
                    finding a blank line.

        Args:
            text_lines (List[str]): list containing the lines to
                be processed.

        Returns:
            Tuple[List[str], List[int]]: tuple containing a list 
                of the processed lines and a list with the tokens 
                computed from these lines.
        """
        processed_lines = []
        tokenized_lines = []
        max_stacked_sentences = 12

        text_lines_gen = (l for l in text_lines)
        for line in text_lines_gen:
            processed_line = self.process_line(line)

            if processed_line == "":
                continue

            if is_dialog(processed_line):
                processed_line += " \n"

            else:
                for _ in range(max_stacked_sentences):
                    try:
                        extra_line = next(text_lines_gen)
                    except StopIteration:
                        break

                    processed_extra_line = self.process_line(extra_line)

                    if processed_extra_line != "":
                        processed_line += f" {processed_extra_line}"
                    else:
                        break

                processed_line += " \n\n"

            processed_lines.append(processed_line)

            tokenized_line = tokenizer.encode(processed_line)
            tokenized_lines.append(tokenized_line)

        return processed_lines, tokenized_lines

    def process_line(self, line: str) -> str:
        """
        Process a raw line extracted from the dataset.

        Args:
            line (str): raw input line.

        Returns:
            str: processed line.
        """
        processed_line = line.strip()
        processed_line = remove_pagination(processed_line)

        return processed_line

    def __len__(self):
        return len(self.block_token_list)

    def __getitem__(self, item_id):
        return torch.tensor(self.block_token_list[item_id], dtype=torch.long)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    screenwriter_data = ScreenwriterData(tokenizer)

    for data in screenwriter_data:
        logger.info(data)
