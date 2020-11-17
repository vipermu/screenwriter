import os
import logging
import json
from typing import *

import torch
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
    f"with {num_gpu} GPU{'s'*(num_gpu > 1)} "
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

scaler = torch.cuda.amp.GradScaler()

prompt_tokens = None
if args.prompt != "":
    prompt_tokens = tokenizer.encode(args.prompt, return_tensors="pt")
    prompt_tokens = prompt_tokens.to(device)

iteration = 1
for epoch in range(args.num_epochs):
    logger.info(f"EPOCH {epoch} started -- {Quote.print()}")

    for data_batch in data_loader:
        data_batch = data_batch.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(data_batch, labels=data_batch)

            loss, _logits = outputs[:2]

            loss /= args.num_grad_accum
        
        scaler.scale(loss).backward()
        # loss.backward()

        if iteration % args.num_grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if iteration % args.generation_freq == 0:
            logger.info(f"ITERATION: {iteration}")

            model.eval()

            if args.prompt == "":
                prompt_tokens = data_batch[0][None]

            sentence_list = generate_sentences(
                model=model,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_length=args.generation_limit,
            )

            for sentence_idx, sentence in enumerate(sentence_list):
                logger.info(f"SENTENCE {sentence_idx}: {sentence}")
                
                with open(args.gen_log_file, 'a') as f:
                    f.write(sentence)

            model.train()

        if iteration % args.metrics_freq == 0:
            logger.info(f"ITERATION: {iteration}")
            loss_float = float(loss.data.cpu()) * args.num_grad_accum
            logger.info(f"LOSS: {loss_float}")
            writer.add_scalar("Loss/train", loss_float, iteration)

        if iteration % args.saving_freq == 0:
            out_model_dir = os.path.join(args.save_dir, f"{epoch}_{iteration}")
            model.save_pretrained(out_model_dir)
            tokenizer.save_pretrained(out_model_dir)
            logger.info(f"Model saved in {out_model_dir}")

        iteration = iteration + 1

writer.close()
