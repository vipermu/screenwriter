import os
import glob
import pickle
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
                    token_id_list = pickle.load(cache_file)

            else:
                print(f"Processing {txt_file_path}...")
                with open(txt_file_path) as txt_file:
                    text = txt_file.read()
                    token_id_list = tokenizer.encode(text)
                    token_id_list = tokenizer.build_inputs_with_special_tokens(
                        token_id_list)

                print(f"Storing processed tokens in {cache_file_path}...")
                with open(cache_file_path, "wb") as cache_file:
                    pickle.dump(token_id_list, cache_file)

            print(f"Creating blocks of {block_size} tokens...")
            for block_id in tqdm(range(0, len(token_id_list), block_size)):
                if block_id + block_size < len(token_id_list):
                    self.block_token_list.append(
                        token_id_list[block_id:block_id+block_size])

    def __len__(self):
        return len(self.block_token_list)

    def __getitem__(self, item_id):
        return torch.tensor(self.block_token_list[item_id], dtype=torch.long)

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    screenwriter_data = ScreenwriterData(tokenizer) 

    for data in screenwriter_data:
        print(data)