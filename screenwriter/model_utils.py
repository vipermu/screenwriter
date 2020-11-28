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
        prompt_tokens =  torch.tensor(random.randint(1,30000))[None, None]
    
    if prompt_tokens.shape[1] > max_length:
        prompt_tokens = prompt_tokens[:, 0:max_length]

    
    sample_outputs = model.generate(
        input_ids=prompt_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        max_length=max_length*2,
        num_return_sequences=num_return_sequences
    )

    sentence_list = []
    for idx, sample_output in enumerate(sample_outputs):
        input_sentence = tokenizer.decode(
            prompt_tokens[0],
            skip_special_tokens=True,
        )
        generated_sentence = tokenizer.decode(
            sample_output[prompt_tokens.shape[1]:],
            skip_special_tokens=True,
        )

        sentence = (
            f"\n\n{' # ' * 32}\n\n"
            f"\n\n{' # ' * 32}\n\n"
            f"INPUT:\n\n{input_sentence}"
            f"\n\n{' # ' * 32}\n\n"
            f"GENERATION:\n\n{generated_sentence}"
        )
        sentence_list.append(sentence)

    return sentence_list
