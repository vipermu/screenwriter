import logging
import json
from typing import *

import torch
import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm

from screenwriter.generate_utils import predict_token


@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "./screenwriter/checkpoints/9_65000"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("screenwriter.trainer")

    if torch.cuda.is_available():
        device = 'cuda'
        num_gpu = torch.cuda.device_count()
    else:
        device = 'cpu'
        num_gpu = 0

    logger.info((
        f"Generating in '{device}' "
        f"with {num_gpu} GPU{'s'*(num_gpu > 1)} "
    ))

    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    logger.info(f"Loading model from {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        pad_token_id=tokenizer.eos_token_id,
        gradient_checkpointing=False,
    )

    model = model.to(device)
    model.eval()
    # TODO: remove this if autocast can fit the gpu
    model.half()

    return model, tokenizer


def main():
    model_name = "./screenwriter/checkpoints/9_65000"

    st.title(" Screenplay Generator ")
    st.write(f""" Using {model_name} """)

    max_generation_len = st.sidebar.slider(
        "Generation Lenght",
        32,
        2048,
    )

    context = st.sidebar.text_area("Context")

    model, tokenizer = load_model()

    if st.sidebar.button("Generate"):
        if not context:
            context = " "

        sentence = generate(
            input_text=context,
            model=model,
            tokenizer=tokenizer,
            max_generation_len=max_generation_len,
            max_context_len=256,
        )

    else:
        pass


def generate(
    input_text: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    max_generation_len: int = 200,
    max_context_len: int = 256,
):
    generated_sentence = input_text
    prompt_tokens = torch.tensor(tokenizer.encode(
        generated_sentence)).to("cuda").unsqueeze(0)
    text_form = st.empty()

    for _ in tqdm(range(max_generation_len)):
        # NOTE: uncomment this and remove `model.half()` if it fits into the GPU
        # with torch.cuda.amp.autocast():
        #     outputs = model(prompt_tokens)

        context_len = prompt_tokens.shape[1]
        if context_len > max_context_len:
            prompt_tokens = prompt_tokens[:, (context_len - max_context_len):]

        outputs = model(prompt_tokens)

        last_scores = outputs[0][:, -1, :]
        probs = torch.softmax(last_scores, dim=-1)

        predicted_token = predict_token(probs)
        predicted_token = predicted_token.to("cuda")

        prompt_tokens = torch.cat([prompt_tokens, predicted_token], dim=1)

        predicted_word = tokenizer.decode(
            predicted_token,
            skip_special_tokens=True,
        )

        generated_sentence += predicted_word
        text_form.empty()
        text_form.text(generated_sentence)

    return generated_sentence


if __name__ == "__main__":
    main()
