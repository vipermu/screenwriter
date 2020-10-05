import os
import glob
import pickle
import re
from typing import *

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer


class ScreenwriterData(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: GPT2Tokenizer,
        data_dir: Optional[str] = "./data",
        batch_size: Optional[int] = 1,
        block_size: Optional[int] = 512,
        recompute: Optional[bool] = False,
    ) -> None:
        txt_file_path_list = glob.glob(os.path.join(data_dir, "*.txt"))

        self.block_token_list = []
        for txt_file_path in txt_file_path_list:
            cache_file_path = txt_file_path[:-4] + "-tokens.pkl"

            if os.path.exists(cache_file_path) and not recompute:
                print(f"Loading cached data from {cache_file_path}...")
                with open(cache_file_path, "rb") as cache_file:
                    tokenized_lines = pickle.load(cache_file)

            else:
                print(f"Processing {txt_file_path}...")
                with open(txt_file_path) as txt_file:
                    text_lines = txt_file.readlines()
                    _processed_lines, tokenized_lines = self.process_lines(
                        text_lines=text_lines,
                        tokenizer=tokenizer,
                    )

                print(f"Storing processed tokens in {cache_file_path}...")
                with open(cache_file_path, "wb") as cache_file:
                    pickle.dump(tokenized_lines, cache_file)

            print(f"Creating blocks of {block_size} tokens...")

            cached_tokenized_line = []
            for tokenized_line in tokenized_lines: 
                cached_tokenized_line.extend(tokenized_line)

                if len(cached_tokenized_line) >= block_size:
                    block_tokenized_line = cached_tokenized_line[0:block_size]
                    self.block_token_list.append(block_tokenized_line)

                    cached_tokenized_line = tokenized_line


    @staticmethod
    def process_lines(
        text_lines: List[str],
        tokenizer: GPT2Tokenizer,
    ) -> Tuple[List[str], List[int]]:
        """
        Process a list of lines removing extra spaces, pagination
        lines and organizing line jumps (i.e. `\n`) to be at the 
        end of each line.
        
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

        cached_line = ""
        line_ended = False
        for line in text_lines:
            line = line.strip()
            
            #NOTE: remove pagination
            if re.search("^\d+\.", line) is not None:
                continue

            if line == "":
                if cached_line:
                    cached_line += " \n"
                line_ended = True

            else:
                if line_ended and cached_line:
                    processed_lines.append(cached_line)

                    tokenized_line = tokenizer.encode(cached_line)
                    tokenized_lines.append(tokenized_line)

                    cached_line = ""
                    line_ended = False
                    
                if not cached_line:
                    cached_line = line
                else:
                    cached_line += f" {line}"
        
        return processed_lines, tokenized_lines

    def __len__(self):
        return len(self.block_token_list)

    def __getitem__(self, item_id):
        return torch.tensor(self.block_token_list[item_id], dtype=torch.long)

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    screenwriter_data = ScreenwriterData(tokenizer) 

    for data in screenwriter_data:
        print(data)
