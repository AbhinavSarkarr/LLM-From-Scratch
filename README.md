# LLM From Scratch - JurisGPT

A comprehensive implementation of a decoder-only transformer architecture built entirely from scratch, covering every component from tokenization to fine-tuning. This project serves as an educational resource for understanding the internals of Large Language Models.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Learning Path](#learning-path)
- [Installation](#installation)
- [Modules](#modules)
- [Usage](#usage)
- [Live Demo](#live-demo)
- [Technologies](#technologies)
- [Author](#author)
- [License](#license)

## Overview

This repository provides a step-by-step implementation of a GPT-style language model from scratch. Each component is built and explained in dedicated Jupyter notebooks, making it an ideal resource for:

- **Understanding Transformer Architecture**: Learn how attention mechanisms, embeddings, and layer normalization work together
- **Building Tokenizers**: Implement Byte Pair Encoding (BPE) from first principles
- **Training Language Models**: Understand pretraining, loss computation, and optimization
- **Decoding Strategies**: Explore temperature scaling, top-k sampling, and other generation techniques
- **Fine-tuning**: Learn classification and instruction fine-tuning approaches

## Project Structure

```
LLM-From-Scratch/
│
├── Tokenization/
│   ├── tokenizer_from_scratch.ipynb          # Basic tokenizer implementation
│   └── byte_pair_encoding_tokenizer_from_scratch.ipynb  # BPE tokenizer
│
├── Embedding Creation/
│   └── create_token_embeddings.ipynb         # Token and positional embeddings
│
├── Self Attention Mechanism/
│   ├── simple_attention_mechanism_without_weights.ipynb    # Basic attention concept
│   ├── self_attention_mechanism_with_trainable_weights.ipynb  # Learnable attention
│   ├── casual_attention_mechanism.ipynb      # Causal (masked) attention
│   └── multi_head_attention_mechanism.ipynb  # Multi-head attention
│
├── Transformer Architecture/
│   ├── bird_eye_view.ipynb                   # Architecture overview
│   ├── gelu_activation_function.ipynb        # GELU activation
│   ├── layer_normalization.ipynb             # Layer normalization
│   ├── shortcut_connections.ipynb            # Residual connections
│   └── entire_transformer_block.ipynb        # Complete transformer block
│
├── Building GPT model/
│   ├── creating_gpt_architecture.ipynb       # Full GPT model
│   └── predicting_next_token.ipynb           # Token prediction
│
├── LLM Input Formatting/
│   └── data_loader_input_output_pairs.ipynb  # Data loading and batching
│
├── Pretraining GPT/
│   ├── input_target_pars_and_loss_function.ipynb  # Loss computation
│   ├── backprogagtion_weight_training.ipynb  # Training loop
│   └── training_and_validation_loss.ipynb    # Training monitoring
│
├── DECODING STRATEGIES TO CONTROL RANDOMNESS/
│   └── temperature_scaling_and_topK.ipynb    # Generation strategies
│
├── Finetuning/
│   ├── classification_finetuning.ipynb       # Classification fine-tuning
│   └── instruction_finetuning.ipynb          # Instruction following
│
├── End To End Files/
│   ├── end_to_end_llm.ipynb                  # Complete pipeline
│   ├── small_langauge_model_end_to_end.ipynb # Simplified version
│   └── pretraining_llm_from_scratch_guide.ipynb  # Comprehensive guide
│
├── the-verdict.txt                           # Training corpus
├── requirements.txt                          # Dependencies
└── README.md
```

## Learning Path

Follow this recommended sequence to understand LLMs from the ground up:

### 1. Tokenization
Start with understanding how text is converted to numerical tokens.
- Learn basic tokenization concepts
- Implement Byte Pair Encoding (BPE) algorithm from scratch

### 2. Embeddings
Understand how tokens become dense vectors.
- Create token embeddings
- Implement positional encodings

### 3. Attention Mechanism
Master the core innovation of transformers.
- Simple attention without learnable weights
- Self-attention with trainable parameters
- Causal (masked) attention for autoregressive models
- Multi-head attention for parallel attention

### 4. Transformer Architecture
Build the complete transformer block.
- GELU activation function
- Layer normalization
- Residual (shortcut) connections
- Full transformer block assembly

### 5. GPT Model
Combine components into a complete GPT model.
- Full architecture implementation
- Next token prediction

### 6. Training
Learn how to train language models.
- Data loading and batching
- Loss function (Cross-Entropy)
- Backpropagation and optimization
- Training and validation monitoring

### 7. Decoding
Generate text from trained models.
- Temperature scaling
- Top-k sampling
- Nucleus (top-p) sampling

### 8. Fine-tuning
Adapt models for specific tasks.
- Classification fine-tuning
- Instruction fine-tuning

## Installation

### Prerequisites
- Python 3.10 or higher
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repository
git clone https://github.com/AbhinavSarkarr/LLM-From-Scratch.git
cd LLM-From-Scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
tiktoken
torch
matplotlib
```

## Modules

### Tokenization
| Notebook | Description |
|----------|-------------|
| `tokenizer_from_scratch.ipynb` | Basic word-level tokenizer implementation |
| `byte_pair_encoding_tokenizer_from_scratch.ipynb` | Complete BPE algorithm implementation |

### Self Attention
| Notebook | Description |
|----------|-------------|
| `simple_attention_mechanism_without_weights.ipynb` | Attention concept visualization |
| `self_attention_mechanism_with_trainable_weights.ipynb` | Query, Key, Value projections |
| `casual_attention_mechanism.ipynb` | Masked attention for autoregressive models |
| `multi_head_attention_mechanism.ipynb` | Parallel attention heads |

### Transformer Architecture
| Notebook | Description |
|----------|-------------|
| `gelu_activation_function.ipynb` | GELU vs ReLU comparison |
| `layer_normalization.ipynb` | Pre-LayerNorm implementation |
| `shortcut_connections.ipynb` | Residual connection patterns |
| `entire_transformer_block.ipynb` | Complete block assembly |

### Pretraining
| Notebook | Description |
|----------|-------------|
| `input_target_pars_and_loss_function.ipynb` | Next-token prediction loss |
| `backprogagtion_weight_training.ipynb` | Training loop implementation |
| `training_and_validation_loss.ipynb` | Training dynamics visualization |

### Decoding & Fine-tuning
| Notebook | Description |
|----------|-------------|
| `temperature_scaling_and_topK.ipynb` | Controlled text generation |
| `classification_finetuning.ipynb` | Task-specific adaptation |
| `instruction_finetuning.ipynb` | Instruction-following capability |

## Usage

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

Navigate to the desired module folder and open the notebooks in sequence.

### Quick Start - End to End

For a complete overview, start with:
```
End To End Files/pretraining_llm_from_scratch_guide.ipynb
```

### Training Your Own Model

```python
import torch
from model import GPT  # After completing the notebooks

# Initialize model
model = GPT(
    vocab_size=50257,
    n_layers=6,
    n_heads=8,
    embed_dim=512,
    context_length=256
)

# Load training data
with open('the-verdict.txt', 'r') as f:
    text = f.read()

# Train (see pretraining notebooks for full implementation)
```

## Live Demo

An interactive BPE tokenizer visualization is deployed at:

**[https://bytepairtokenizer.netlify.app/](https://bytepairtokenizer.netlify.app/)**

Experience how Byte Pair Encoding works in real-time!

## Technologies

| Technology | Purpose |
|------------|---------|
| **PyTorch** | Deep learning framework for model implementation |
| **tiktoken** | OpenAI's tokenizer for comparison/baseline |
| **Matplotlib** | Visualization of attention patterns and training curves |
| **NumPy** | Numerical computations |
| **Jupyter** | Interactive notebook environment |

## Key Concepts Covered

- **Byte Pair Encoding (BPE)**: Subword tokenization algorithm
- **Token Embeddings**: Dense vector representations
- **Positional Encoding**: Sequence position information
- **Scaled Dot-Product Attention**: Core attention computation
- **Multi-Head Attention**: Parallel attention mechanisms
- **Layer Normalization**: Training stabilization
- **Residual Connections**: Gradient flow improvement
- **GELU Activation**: Smooth non-linearity
- **Cross-Entropy Loss**: Language modeling objective
- **Temperature Scaling**: Generation diversity control
- **Top-k/Top-p Sampling**: Controlled randomness

## Model Architecture

```
GPT Model
├── Token Embedding Layer (vocab_size × embed_dim)
├── Positional Embedding Layer (context_length × embed_dim)
├── Transformer Blocks (× n_layers)
│   ├── Multi-Head Self-Attention
│   │   ├── Query Projection
│   │   ├── Key Projection
│   │   ├── Value Projection
│   │   ├── Scaled Dot-Product Attention
│   │   └── Output Projection
│   ├── Layer Normalization
│   ├── Feed-Forward Network
│   │   ├── Linear (embed_dim → 4 × embed_dim)
│   │   ├── GELU Activation
│   │   └── Linear (4 × embed_dim → embed_dim)
│   └── Layer Normalization
├── Final Layer Normalization
└── Output Projection (embed_dim → vocab_size)
```

## Author

**Abhinav Sarkar**
- GitHub: [@AbhinavSarkarr](https://github.com/AbhinavSarkarr)
- LinkedIn: [abhinavsarkarrr](https://www.linkedin.com/in/abhinavsarkarrr)
- Portfolio: [abhinav-ai-portfolio.lovable.app](https://abhinav-ai-portfolio.lovable.app/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by Andrej Karpathy's "Let's build GPT" series
- OpenAI's GPT architecture papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- The PyTorch team for the excellent deep learning framework

---

<p align="center">
  <strong>Learn by Building - Understanding LLMs from First Principles</strong>
</p>
