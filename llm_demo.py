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

# 3. Model definition
class SimpleLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.W_q = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.W_k = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.W_v = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, VOCAB_SIZE)
        )

    def prefill(self, input_ids):
        # input_ids: [seq_len]
        embeds = self.embedding(input_ids)  # [seq_len, embed_dim]
        print("Embeddings:\n", embeds)
        Q = self.W_q(embeds)  # [seq_len, embed_dim]
        K = self.W_k(embeds)  # [seq_len, embed_dim]
        V = self.W_v(embeds)  # [seq_len, embed_dim]
        print("Q:\n", Q)
        print("K:\n", K)
        print("V:\n", V)
        return K, V, embeds

    def decode(self, input_ids, K, V):
        # input_ids: [seq_len]
        embeds = self.embedding(input_ids)  # [seq_len, embed_dim]
        Q = self.W_q(embeds[-1:])  # [1, embed_dim] (last token)
        print("\n[Decode] Embedding for last token:", embeds[-1:])
        print("[Decode] Q for last token:", Q)
        # Attention: Q [1, d], K [seq, d], V [seq, d]
        attn_scores = torch.matmul(Q, K.T) / (EMBED_DIM ** 0.5)  # [1, seq_len]
        print("[Decode] Attention scores:", attn_scores)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [1, seq_len]
        print("[Decode] Attention weights:", attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # [1, embed_dim]
        print("[Decode] Attention output:", attn_output)
        # Feedforward
        logits = self.ffn(attn_output)  # [1, vocab_size]
        print("[Decode] Logits:", logits)
        probs = F.softmax(logits, dim=-1)  # [1, vocab_size]
        print("[Decode] Probabilities:", probs)
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
    model = SimpleLLM()
    model.eval()
    sentence = input(f"Enter a sentence (any English text):\n> ")
    input_ids = encode(sentence)
    print("\nInput token indices:", input_ids)
    print("Input tokens:", [decode(idx.item()) for idx in input_ids])
    # Prefill: process all tokens so far
    K, V, embeds = model.prefill(input_ids)
    # Decode: generate one new token
    print("\n--- Decoding next token ---")
    probs = model.decode(input_ids, K, V)
    # Sample next word
    next_token = torch.multinomial(probs[0], num_samples=1).item()
    print(f"\nPredicted next token: {decode(next_token)} (token {next_token})")

if __name__ == "__main__":
    main() 