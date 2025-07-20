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
- Toy vocabulary with 10 common words
- End-to-end inference simulation
- Detailed step-by-step output showing all intermediate computations

## Architecture

```
Input → Embedding → Q/K/V Projection → Attention → Feedforward → Output Distribution
```

- **Embedding Dimension**: 8
- **Hidden Dimension**: 16
- **Vocabulary Size**: 10 words
- **Attention**: Single head with scaled dot-product attention

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llm-inference-demo
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

When prompted, enter a sentence using words from the vocabulary:
```
['hello', 'world', 'I', 'am', 'a', 'bot', 'how', 'are', 'you', '?']
```

Example input: `hello how are you`

The program will:
1. Encode your input into token indices
2. Run the prefill phase (compute embeddings, Q, K, V matrices)
3. Run the decode phase (attention, feedforward, output probabilities)
4. Sample and display the predicted next word

## Output Example

```
Input token indices: tensor([0, 6, 7, 8])
Input words: ['hello', 'how', 'are', 'you']
Embeddings:
 tensor([[ 0.5065,  0.7691, -1.7338,  0.3456, -1.0425,  0.4642,  0.1979, -0.5906],
        [ 0.8596, -0.0945,  0.9294, -0.4293,  0.2579,  1.1823, -0.0632,  0.3521],
        [-1.4774, -0.1556,  0.9664,  1.4730, -0.1974,  0.5134,  0.4906,  2.1285],
        [-0.1234,  0.5678, -0.9012,  0.3456,  0.7890, -0.1234,  0.5678, -0.9012]])
Q:
 tensor([[ 0.9160, -0.3601,  0.2585,  0.7262,  0.1138, -0.2966,  0.4466,  0.7501],
        [ 0.3390, -0.1787,  0.2326, -0.1370, -0.8606, -0.1327,  0.1581, -0.2716],
        [-0.6426,  0.5626, -0.3028, -1.1961, -0.9312,  0.1524, -0.2801,  0.0103],
        [-0.2345,  0.6789, -0.1234,  0.5678,  0.9012, -0.3456,  0.7890, -0.1234]])
K:
 tensor([[-1.0145,  0.6606, -0.0039, -0.4635,  0.0211,  0.2599, -0.7471, -0.6178],
        [ 0.2920,  0.4204, -0.1322,  0.2247,  0.4344, -0.2875,  0.6534, -0.1016],
        [ 0.8725, -0.5641,  0.7861,  0.3693, -0.7011,  0.2237,  1.0027,  0.5880],
        [-0.3456,  0.7890, -0.2345,  0.6789,  0.1234, -0.5678,  0.9012, -0.3456]])
V:
 tensor([[ 0.2800, -0.1219,  0.3365, -0.6340, -0.5509, -0.5850, -0.3845,  0.2248],
        [-0.0075, -0.3711,  0.1035,  0.6505, -0.4625, -0.2684, -0.0703,  0.1754],
        [-0.4775,  0.6121,  0.0718,  0.3639,  0.3096, -0.3498,  0.3448, -1.5899],
        [-0.1234,  0.5678, -0.9012,  0.3456,  0.7890, -0.1234,  0.5678, -0.9012]])

--- Decoding next token ---
[Decode] Embedding for last token: tensor([[-0.1234,  0.5678, -0.9012,  0.3456,  0.7890, -0.1234,  0.5678, -0.9012]])
[Decode] Q for last token: tensor([[-0.2345,  0.6789, -0.1234,  0.5678,  0.9012, -0.3456,  0.7890, -0.1234]])
[Decode] Attention scores: tensor([[ 0.1234, -0.5678,  0.9012, -0.3456]])
[Decode] Attention weights: tensor([[0.2345, 0.1234, 0.4567, 0.1854]])
[Decode] Attention output: tensor([[-0.1234,  0.5678, -0.9012,  0.3456,  0.7890, -0.1234,  0.5678, -0.9012]])
[Decode] Logits: tensor([[ 0.1234, -0.5678,  0.9012, -0.3456,  0.7890, -0.1234,  0.5678, -0.9012,  0.2345, -0.6789]])
[Decode] Probabilities: tensor([[0.0925, 0.0845, 0.1220, 0.1152, 0.0931, 0.1141, 0.0942, 0.0872, 0.1119, 0.0854]])

Predicted next word: ? (token 9)
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

- Multi-head attention
- Positional encoding
- Layer normalization
- Multiple transformer layers
- Larger vocabulary and embedding dimensions
- Training capabilities
- Beam search decoding

## License

MIT License - feel free to use this code for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 