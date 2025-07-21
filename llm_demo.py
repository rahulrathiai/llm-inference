import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import random

# 1. Tokenizer and dynamic vocabulary
TOKENIZER_NAME = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(TOKENIZER_NAME)
VOCAB_SIZE = tokenizer.vocab_size

# 2. Hyperparameters
EMBED_DIM = 8
HIDDEN_DIM = 16
NUM_LAYERS = 6  # 6 layers for distil GPT-2

# 3. Model definition
class SimpleLLM(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        
        # Create multiple layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.ModuleDict({
                    'W_q': nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
                    'W_k': nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
                    'W_v': nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
                    'W_o': nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
                }),
                'ffn': nn.Sequential(
                    nn.Linear(EMBED_DIM, HIDDEN_DIM),
                    nn.ReLU(),
                    nn.Linear(HIDDEN_DIM, EMBED_DIM)
                ),
                'ln1': nn.LayerNorm(EMBED_DIM),
                'ln2': nn.LayerNorm(EMBED_DIM),
            })
            self.layers.append(layer)
        
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def _layer_forward(self, layer, hidden_states, layer_idx, mode="prefill"):
        # Self-attention
        q = layer['attention']['W_q'](hidden_states) if mode == "prefill" else layer['attention']['W_q'](hidden_states[-1:])
        k = layer['attention']['W_k'](hidden_states)
        v = layer['attention']['W_v'](hidden_states)
        
        if mode == "prefill":
            print(f"Q:\n", q)
            print(f"K:\n", k)
            print(f"V:\n", v)
        else:
            print(f"[Decode Layer {layer_idx + 1}] Q for last token:", q)
        
        # Attention computation
        attn_scores = torch.matmul(q, k.T) / (EMBED_DIM ** 0.5)
        if mode == "prefill":
            print(f"Attention scores:\n", attn_scores)
        else:
            print(f"[Decode Layer {layer_idx + 1}] Attention scores:", attn_scores)
        attn_weights = F.softmax(attn_scores, dim=-1)
        if mode == "prefill":
            print(f"Attention weights:\n", attn_weights)
        else:
            print(f"[Decode Layer {layer_idx + 1}] Attention weights:", attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = layer['attention']['W_o'](attn_output)
        if mode == "decode":
            print(f"[Decode Layer {layer_idx + 1}] Attention output:", attn_output)
        
        # Residual connection and layer norm
        hidden_states = layer['ln1'](hidden_states + attn_output)
        
        # Feedforward
        ff_output = layer['ffn'](hidden_states)
        hidden_states = layer['ln2'](hidden_states + ff_output)
        
        if mode == "prefill":
            print(f"Layer {layer_idx + 1} output:\n", hidden_states)
        else:
            print(f"[Decode Layer {layer_idx + 1}] Output hidden states:", hidden_states)
        
        return hidden_states, k, v

    def prefill(self, input_ids):
        embeds = self.embedding(input_ids)  # [seq_len, embed_dim]
        print("Embeddings:\n", embeds)
        hidden_states = embeds
        K_cache = []
        V_cache = []
        for layer_idx, layer in enumerate(self.layers):
            print(f"\n--- Layer {layer_idx + 1} ---")
            hidden_states, k, v = self._layer_forward(layer, hidden_states, layer_idx, mode="prefill")
            K_cache.append(k)
            V_cache.append(v)
        return K_cache, V_cache

    def decode(self, input_ids, K_cache, V_cache):
        embeds = self.embedding(input_ids)  # [seq_len, embed_dim]
        hidden_states = embeds
        print(f"\n--- Decode Phase (Processing through {self.num_layers} layers) ---")
        for layer_idx, layer in enumerate(self.layers):
            print(f"\n[Decode Layer {layer_idx + 1}]")
            hidden_states, k, v = self._layer_forward(layer, hidden_states, layer_idx, mode="decode")
            # Append the new K, V for the decode token
            K_cache[layer_idx] = torch.cat([K_cache[layer_idx], k[-1:]], dim=0)
            V_cache[layer_idx] = torch.cat([V_cache[layer_idx], v[-1:]], dim=0)
        logits = self.lm_head(hidden_states[-1:])  # [1, vocab_size]
        print("[Decode] Final logits:", logits)
        probs = F.softmax(logits, dim=-1)
        print("[Decode] Final probabilities:", probs)
        return probs

# 4. Utility functions
def encode(sentence):
    # Use the Hugging Face tokenizer
    return torch.tensor(tokenizer.encode(sentence, add_special_tokens=False), dtype=torch.long)

def decode(idx):
    # Decode a single token id to string
    return tokenizer.decode([idx])

# 5. Main demo
def main():
    model = SimpleLLM(num_layers=NUM_LAYERS)
    model.eval()
    sentence = input(f"Enter a sentence (any English text):\n> ")
    input_ids = encode(sentence)
    print("\nInput token indices:", input_ids)
    print("Input tokens:", [decode(idx.item()) for idx in input_ids])
    print(f"\nModel: SimpleLLM with {NUM_LAYERS} layers")
    
    # Prefill: process all tokens so far
    K_cache, V_cache = model.prefill(input_ids)
    
    # Decode: generate one new token
    print("\n--- Decoding next token ---")
    probs = model.decode(input_ids, K_cache, V_cache)
    
    # Sample next word
    next_token = torch.multinomial(probs[0], num_samples=1).item()
    print(f"\nPredicted next token: {decode(next_token)} (token {next_token})")
    
    # Print top 10 tokens and their probabilities
    topk = torch.topk(probs[0], 10)
    print("\nTop 10 next tokens and probabilities:")
    for rank, (idx, prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
        print(f"{rank:2d}. {decode(idx):15s} (token {idx:5d}): {prob:.5f}")

if __name__ == "__main__":
    main() 