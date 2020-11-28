import logging
import json
from typing import *

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from screenwriter.datasets import ScreenwriterData
from screenwriter.generate_utils import predict_token
from screenwriter.args import get_generate_args


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("screenwriter.generate")

if torch.cuda.is_available():
    device = 'cuda'
    num_gpu = torch.cuda.device_count()
else:
    device = 'cpu'
    num_gpu = 0

args = get_generate_args()

logger.info(json.dumps(args.__dict__, indent=2))
logger.info((
    f"Generating in '{device}' "
    f"with {num_gpu} GPU{'s'*(num_gpu > 1)} "
))

logger.info(f"Loading tokenizer from {args.model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

logger.info(f"Loading model from {args.model_name}...")
model = GPT2LMHeadModel.from_pretrained(
    args.model_name,
    pad_token_id=tokenizer.eos_token_id,
    gradient_checkpointing=False,
)
model = model.to(device)
model.eval()
# TODO: remove this and uncomment autocast if it can fit into the GPU.
model.half()

dataset = ScreenwriterData(
    tokenizer,
    block_size=args.block_size,
    recompute=args.recompute_data,
)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)

if args.prompt != "":
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors="pt")
    prompt_tokens = prompt_tokens.to(device)
else:
    prompt_tokens = next(iter(data_loader)).to(device)

generated_sentence = tokenizer.decode(
    prompt_tokens[0, :],
    skip_special_tokens=True,
)


for _ in range(args.max_generation_len):
    # NOTE: uncomment this and remove `model.half()` if it fits into GPU.
    # with torch.cuda.amp.autocast():
    #     outputs = model(prompt_tokens)

    outputs = model(prompt_tokens)

    last_scores = outputs[0][:, -1, :]
    probs = torch.softmax(last_scores, dim=-1)

    predicted_token = predict_token(
        probs=probs,
        k=args.k,
        p=args.p,
    )
    predicted_token = predicted_token.to(device)

    prompt_tokens = torch.cat([prompt_tokens, predicted_token], dim=1)

    context_len = prompt_tokens.shape[1]
    if context_len > args.max_context_len:
        prompt_tokens = prompt_tokens[:, (context_len - args.max_context_len):]

    predicted_word = tokenizer.decode(
        predicted_token,
        skip_special_tokens=True,
    )

    generated_sentence += predicted_word

    logger.info(generated_sentence)

    if args.output_file != "":
        with open(args.output_file, "w") as file:
            file.write(generated_sentence)
