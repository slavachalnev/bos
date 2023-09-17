# %%
import torch
from transformer_lens import HookedTransformer
from token_print import ColoredTokenizer
from IPython.display import display

import circuitsvis as cv


model = HookedTransformer.from_pretrained('attn-only-2l', device='cpu')
tokenizer = model.tokenizer
ct = ColoredTokenizer(tokenizer)

# %%

def doubled_patterns(text, insert_bos=False):
    mid_id = [model.tokenizer.bos_token_id] if insert_bos else []
    tokens = tokenizer.encode(text)
    tokens = [tokenizer.bos_token_id] + tokens + mid_id + tokens
    tok_tens = torch.tensor(tokens).unsqueeze(0)
    print(tokens)
    ct(tokens)

    with torch.no_grad():
        loss, cache = model.run_with_cache(tok_tens, remove_batch_dim=True, return_type='loss')
    print('loss is ', loss)

    mid_tok = [model.tokenizer.bos_token] if insert_bos else []
    str_toks = model.to_str_tokens(text, prepend_bos=False)
    str_toks = [model.tokenizer.bos_token] + str_toks + mid_tok + str_toks

    for layer in range(model.cfg.n_layers):
        attn_pattern = cache['pattern', layer]
        display(cv.attention.attention_patterns(
            tokens=str_toks,
            attention=attn_pattern,
        ))

# %%

text = "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."
doubled_patterns(text)
doubled_patterns(text, insert_bos=True)

# %%
