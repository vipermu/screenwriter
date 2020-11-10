import random
import torch
from typing import *

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def generate_sentences(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    top_k: int = 50,
    top_p: float = 0.95,
    max_length: int = 512,
    num_return_sequences: int = 1,
    prompt_tokens: torch.Tensor = None
) -> List[str]:
    if prompt_tokens is None:
        prompt_tokens =  random.randint(1,30000)

    sample_outputs = model.generate(
        input_ids=prompt_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        max_length=len(prompt_tokens) + max_length,
        num_return_sequences=num_return_sequences
    )

    sentence_list = []
    for idx, sample_output in enumerate(sample_outputs):
        sentence = tokenizer.decode(
            sample_output,
            skip_special_tokens=True,
        )

        sentence = f"\n{sentence}"
        sentence_list.append(sentence)

    return sentence_list
