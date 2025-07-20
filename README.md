# Simple LLM Inference Demo

A minimal PyTorch implementation demonstrating the core components of Large Language Model (LLM) inference, including prefill and decode phases.

## Overview

This project simulates LLM inference with:
- **Prefill Phase**: Computes and stores Key (K) and Value (V) matrices for the input sequence
- **Decode Phase**: Runs attention mechanism and feedforward network to predict the next token
- **Educational Output**: Prints all intermediate matrices and computations for learning purposes

## Features

- Single-head self-attention mechanism
- Simple feedforward neural network
- **Unrestricted vocabulary:** Uses the Hugging Face GPT-2 tokenizer, so you can input any English text
- End-to-end inference simulation
- Detailed step-by-step output showing all intermediate computations

## Architecture

```
Input → Embedding → Q/K/V Projection → Attention → Feedforward → Output Distribution
```

- **Embedding Dimension**: 8
- **Hidden Dimension**: 16
- **Vocabulary**: GPT-2 tokenizer (50,257 tokens)
- **Attention**: Single head with scaled dot-product attention

## Installation

1. Clone the repository:
```bash
git clone https://github.com/rahulrathiai/llm-inference.git
cd llm-inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the demo:
```bash
python llm_demo.py
```

When prompted, enter any English sentence. The model will tokenize your input using the GPT-2 tokenizer and process it through the demo LLM.

Example input: `hello how are you?`

The program will:
1. Encode your input into token indices using the GPT-2 tokenizer
2. Run the prefill phase (compute embeddings, Q, K, V matrices)
3. Run the decode phase (attention, feedforward, output probabilities)
4. Sample and display the predicted next token

## Output Example

```
Input token indices: tensor([31373, 703, 389, 345, 30])
Input tokens: ['hello', ' how', ' are', ' you', '?']
Embeddings:
 tensor([...])
Q:
 tensor([...])
K:
 tensor([...])
V:
 tensor([...])

--- Decoding next token ---
[Decode] Embedding for last token: tensor([...])
[Decode] Q for last token: tensor([...])
[Decode] Attention scores: tensor([...])
[Decode] Attention weights: tensor([...])
[Decode] Attention output: tensor([...])
[Decode] Logits: tensor([...])
[Decode] Probabilities: tensor([...])

Predicted next token:  (token 220)
```

## Technical Details

### Prefill Phase
- Processes the entire input sequence at once
- Computes embeddings for all tokens
- Projects embeddings to Q, K, V matrices
- Stores K and V for use in decode phase

### Decode Phase
- Takes the last token as query
- Computes attention scores with all stored K values
- Applies softmax to get attention weights
- Computes weighted sum of V values
- Passes through feedforward network
- Outputs probability distribution over vocabulary
- Samples next token from distribution

### Attention Mechanism
- Scaled dot-product attention: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Single attention head
- No positional encoding (for simplicity)

## Future Enhancements

- Use real LLM weights from a small open-source model (e.g., GPT-2, TinyLlama)
- Multi-head attention
- Positional encoding
- Layer normalization
- Multiple transformer layers
- Larger embedding dimensions
- Training capabilities
- Beam search decoding

## License

MIT License - feel free to use this code for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 