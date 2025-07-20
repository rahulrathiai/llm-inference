import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 1. Toy vocabulary and encoding/decoding
VOCAB = ["hello", "world", "I", "am", "a", "bot", "how", "are", "you", "?"]
word2idx = {w: i for i, w in enumerate(VOCAB)}
idx2word = {i: w for i, w in enumerate(VOCAB)}

# 2. Hyperparameters
EMBED_DIM = 8
HIDDEN_DIM = 16
VOCAB_SIZE = len(VOCAB)

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
    # Simple whitespace split, map to indices, unknowns to 0
    return torch.tensor([word2idx.get(w, 0) for w in sentence.strip().split()], dtype=torch.long)

def decode(idx):
    return idx2word.get(idx, "<unk>")

# 5. Main demo
def main():
    model = SimpleLLM()
    model.eval()
    sentence = input(f"Enter a sentence using words from the vocab {VOCAB}:\n> ")
    input_ids = encode(sentence)
    print("\nInput token indices:", input_ids)
    print("Input words:", [decode(idx.item()) for idx in input_ids])
    # Prefill: process all tokens so far
    K, V, embeds = model.prefill(input_ids)
    # Decode: generate one new token
    print("\n--- Decoding next token ---")
    probs = model.decode(input_ids, K, V)
    # Sample next word
    next_token = torch.multinomial(probs[0], num_samples=1).item()
    print(f"\nPredicted next word: {decode(next_token)} (token {next_token})")

if __name__ == "__main__":
    main() 