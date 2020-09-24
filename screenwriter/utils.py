import random
from typing import *

from transformers import GPT2LMHeadModel


def generate_sentences(
    model: GPT2LMHeadModel,
    top_k: int = 50,
    top_p: float = 0.95,
    max_length: int = 200,
    num_return_sequences: int = 1,
) -> List[str]:
    model.eval()

    sample_outputs = model.generate(
        bos_token_id=random.randint(1, 30000),
        do_sample=True,
        top_k=50,
        top_p=0.95,
        max_length=max_length,
        num_return_sequences=1
    )

    sentence_list = []
    for idx, sample_output in enumerate(sample_outputs):
        sentence = tokenizer.decode(
            sample_output,
            skip_special_tokens=True,
        )
        sentence_list.append(sentence)

    model.train()

    return sentence_list
