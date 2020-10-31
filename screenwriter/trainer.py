import os
import logging
import json
from typing import *

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from quoters import Quote

from screenwriter.datasets import ScreenwriterData
from screenwriter.model_utils import generate_sentences
from screenwriter.args import get_args


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

args = get_args()

logger.info(json.dumps(args.__dict__, indent=2))
logger.info((
    f"Training in '{device}' "
    f"with {num_gpu} GPUs "
    f"and mix precision is set to '{args.use_fp16}'"
))

writer = SummaryWriter(log_dir=args.log_dir)

logger.info(f"Loading tokenizer from {args.model_name}...")
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

logger.info(f"Loading model from {args.model_name}...")
model = GPT2LMHeadModel.from_pretrained(
    args.model_name,
    pad_token_id=tokenizer.eos_token_id,
)
model = model.to(device)
model.train()
if args.use_fp16 and device == "cuda":
    model.half()

dataset = ScreenwriterData(
    tokenizer,
    block_size=args.block_size,
    recompute=args.recompute_data,
)

data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=-1,
)

prompt_tokens = tokenizer.encode(args.prompt, return_tensors="pt")
prompt_tokens.to(device)

iteration = 1
for epoch in range(args.num_epochs):
    logger.info(f"EPOCH {epoch} started -- {Quote.print()}")

    for idx, data in enumerate(data_loader):
        data = data.to(device)
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]

        loss.backward()

        if iteration % args.num_grad_accum == 0:
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            model.zero_grad()

        if iteration % args.metrics_freq == 0:
            logger.info(f"ITERATION: {iteration}")

            loss_float = float(loss.data.cpu())
            logger.info(f"LOSS: {loss_float}")
            writer.add_scalar("Loss/train", loss_float, iteration)

            model.eval()

            sentence_list = generate_sentences(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
            )

            for sentence_idx, sentence in enumerate(sentence_list):
                logger.info(f"SENTENCE {sentence_idx}: {sentence}")

            model.train()

        iteration = iteration + 1

    if (epoch + 1) % args.saving_freq == 0:
        model_save_dir = os.path.join(args.save_dir, str(epoch))
        model.save_pretrained(model_save_dir)
        logger.info("Model saved!")

writer.close()
