# %%
from typing import List
import random

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from IPython.display import display

import circuitsvis as cv


model = HookedTransformer.from_pretrained('attn-only-2l', device='cpu')
# model = HookedTransformer.from_pretrained('gpt2-small', device='cpu')

tokenizer = model.tokenizer
ct = ColoredTokenizer(tokenizer)

# %%

def doubled_patterns(text, insert_bos=False):
    tokenizer = model.tokenizer
    mid_id = [tokenizer.bos_token_id] if insert_bos else tokenizer.encode('\n')
    tokens = tokenizer.encode(text)
    tokens = [tokenizer.bos_token_id] + tokens + mid_id + tokens
    tok_tens = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tok_tens, remove_batch_dim=True, return_type='logits')

    mid_tok = [tokenizer.bos_token] if insert_bos else ["\n"]
    str_toks = model.to_str_tokens(text, prepend_bos=False)
    str_toks = [tokenizer.bos_token] + str_toks + mid_tok + str_toks

    for layer in range(model.cfg.n_layers):
        attn_pattern = cache['pattern', layer]
        display(cv.attention.attention_patterns(
            tokens=str_toks,
            attention=attn_pattern,
        ))


def losses(tokens: List[int], insert_bos=False):
    r1 = random.randint(0, model.cfg.d_vocab)
    bos = tokenizer.bos_token_id
    tokens = [bos] + tokens + ([bos] if insert_bos else [r1]) + tokens
    tok_tens = torch.tensor(tokens).unsqueeze(0)

    with torch.no_grad():
        logits = model.forward(tok_tens, return_type='logits', prepend_bos=False)

    losses = F.cross_entropy(
        logits[0, :-1, :],
        tok_tens[0, 1:],
        reduction='none',
    )
    return losses

# %%

text = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
doubled_patterns(text)
doubled_patterns(text, insert_bos=True)

# %%
# compute average loss over many samples
n_toks = 20
losses_no_bos = torch.zeros(n_toks*2 + 1)
losses_yes_bos = torch.zeros(n_toks*2 + 1)
n_samples = 100

for _ in range(n_samples):
    toks_for_loss = [random.randint(0, model.cfg.d_vocab) for _ in range(20)]
    losses_no_bos += losses(toks_for_loss)
    losses_yes_bos += losses(toks_for_loss, insert_bos=True)

losses_no_bos /= n_samples
losses_yes_bos /= n_samples

length_each_repetition = losses_no_bos.shape[0] // 2
# Splitting the losses into first and second repetitions using calculated length
losses_no_bos_first = losses_no_bos[:length_each_repetition]
losses_no_bos_second = losses_no_bos[length_each_repetition:]
losses_yes_bos_first = losses_yes_bos[:length_each_repetition]
losses_yes_bos_second = losses_yes_bos[length_each_repetition:]

# Plotting
plt.figure(figsize=(15, 6))
plt.plot(losses_no_bos_first, label='First Repetition No BOS', marker='o')
plt.plot(losses_no_bos_second, label='Second Repetition No BOS', marker='s')
plt.plot(losses_yes_bos_first, label='First Repetition With BOS', marker='x')
plt.plot(losses_yes_bos_second, label='Second Repetition With BOS', marker='d')
plt.xlabel('Token Position')
plt.ylabel('Per-Token Loss')
plt.title('Per-Token Loss Comparison Across Repetitions')
plt.legend()
plt.show()

# %%
