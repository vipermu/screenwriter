from typing import *

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

from screenwriter.datasets import ScreenwriterData
from screenwriter.utils import generate_sentences

MODEL_DIR = "gpt2-medium"
BLOCK_SIZE = 128

LEARNING_RATE = 3e-5
NUM_WARMUP_STEPS = 10000
BATCH_SIZE = 4

NUM_EPOCHS = 100
METRICS_FREQ = 200
GLOBAL_COUNTER = 0

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'

train_loss_results = []
train_accuracy_results = []

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

model = GPT2LMHeadModel.from_pretrained(
    MODEL_DIR,
    pad_token_id=tokenizer.eos_token_id,
)
model = model.to(DEVICE)
model.train()

dataset = ScreenwriterData(
    tokenizer,
    block_size=BLOCK_SIZE,
)
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=NUM_WARMUP_STEPS,
    num_training_steps=-1,
)

iteration = 1
for epoch in range(NUM_EPOCHS):
    print(f"EPOCH {epoch} started" + '=' * 30)

    for idx, data in enumerate(data_loader):
        data = data.to(DEVICE)
        outputs = model(data, labels=data)
        loss, logits = outputs[:2]

        loss.backward()

        if iteration % BATCH_SIZE == 0:
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            model.zero_grad()

        if iteration % METRICS_FREQ == 0:
            model.eval()

            sentence_list = generate_sentences(model, tokenizer)
            for sentence_idx, sentence in enumerate(sentence_list):
                print(f"Sentence {sentence_idx}: {sentence}")

            model.train()

        iteration = iteration + 1
