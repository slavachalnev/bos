# %%
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from IPython.display import display

import circuitsvis as cv


model = HookedTransformer.from_pretrained('attn-only-2l', device='cpu')
tokenizer = model.tokenizer
ct = ColoredTokenizer(tokenizer)

# %%

def doubled_patterns(text, insert_bos=False):
    tokenizer = model.tokenizer
    mid_id = [tokenizer.bos_token_id] if insert_bos else tokenizer.encode('.')
    tokens = tokenizer.encode(text)
    tokens = [tokenizer.bos_token_id] + tokens + mid_id + tokens
    tok_tens = torch.tensor(tokens).unsqueeze(0)
    print(tokens)
    ct(tokens)

    with torch.no_grad():
        logits, cache = model.run_with_cache(tok_tens, remove_batch_dim=True, return_type='logits')

    mid_tok = [tokenizer.bos_token] if insert_bos else ["."]
    str_toks = model.to_str_tokens(text, prepend_bos=False)
    str_toks = [tokenizer.bos_token] + str_toks + mid_tok + str_toks

    for layer in range(model.cfg.n_layers):
        attn_pattern = cache['pattern', layer]
        display(cv.attention.attention_patterns(
            tokens=str_toks,
            attention=attn_pattern,
        ))
    losses = F.cross_entropy(
        logits[0, :-1, :],
        tok_tens[0, 1:],
        reduction='none',
    )
    return losses

# %%

text = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
losses_no_bos = doubled_patterns(text)
losses_yes_bos = doubled_patterns(text, insert_bos=True)

# %%

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
