# Dialogue Summarization MVP 🤖💬

**AI-Powered Dialogue Summarization using Transformer Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Project Overview

This project implements and compares two state-of-the-art approaches for automatic dialogue summarization using the SAMSum dataset. It demonstrates the feasibility of deploying AI-powered summarization to address information overload in group conversations.

### Key Features

- ✅ **Dual Architecture Comparison**: BERT-GPT2 Encoder-Decoder vs. GPT-2 Auto-regressive
- ✅ **Comprehensive Data Analysis**: Statistical analysis, vocabulary profiling, and visualization
- ✅ **Production-Ready Optimizations**: Mixed precision (FP16), gradient checkpointing, parallel data loading
- ✅ **Rigorous Evaluation**: ROUGE metrics with validation-based model selection
- ✅ **Complete Pipeline**: From raw data to trained models with sample generation

---

## 🎯 Problem Statement

Acme Communications faces critical challenges with information overload in group conversations. This proof-of-concept demonstrates the technical feasibility and business value of implementing automated summarization using state-of-the-art transformer models to deliver concise, accurate summaries that capture essential information.

---

## 📊 Dataset

**SAMSum Corpus** - Messenger-style conversation dataset

- **Training**: 14,732 dialogues
- **Validation**: 818 dialogues  
- **Test**: 819 dialogues
- **Average Dialogue Length**: ~350 words
- **Average Summary Length**: ~40 words
- **Compression Ratio**: 10:1

---

## 🏗️ Architecture

### Part 1: BERT-GPT2 Encoder-Decoder
```
Input Dialogue → BERT Encoder → Hidden States → GPT-2 Decoder → Summary
```
- **Encoder**: `bert-base-uncased` (110M parameters)
- **Decoder**: `gpt2` (124M parameters)
- **Total**: ~234M parameters

### Part 2: GPT-2 Auto-regressive
```
Prompt: "Dialogue: [text]\n\nSummary:" → GPT-2 → Generated Summary
```
- **Model**: `gpt2` (124M parameters)
- **Approach**: Causal language modeling (ChatGPT-style)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
16GB+ RAM
```

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd dialogue-summarization-mvp

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate rouge_score pandas matplotlib seaborn nltk tqdm
```

### Running the Project
```bash
# Open in Jupyter/Colab
jupyter notebook Dialogue_Summarize_MVP__5_.ipynb

# Or run in Google Colab (recommended for GPU access)
# Upload the notebook and run cells sequentially
```

---

## 📈 Results

### Model Performance (ROUGE Scores on Test Set)

| Metric | BERT-GPT2 | GPT-2 Auto-regressive | Winner |
|--------|-----------|----------------------|--------|
| **ROUGE-1** | 0.1419 | **0.1900** ✅ | GPT-2 AR |
| **ROUGE-2** | 0.0379 | **0.0766** ✅ | GPT-2 AR |
| **ROUGE-L** | 0.1096 | **0.1413** ✅ | GPT-2 AR |
| **ROUGE-Lsum** | 0.1098 | **0.1425** ✅ | GPT-2 AR |

### Training Performance

| Model | Training Loss | Validation Loss | Training Time/Epoch |
|-------|--------------|----------------|-------------------|
| BERT-GPT2 | 0.9226 | 0.9610 | ~15 min (GPU) |
| GPT-2 AR | 2.3828 | 2.4980 | ~45 min (GPU) |

**Key Finding**: The simpler GPT-2 auto-regressive approach outperformed the encoder-decoder architecture, demonstrating that architectural complexity doesn't always translate to better performance with limited training time.

---

## 🔧 Optimizations Implemented

- **Mixed Precision Training (FP16)**: 2-3x speedup, reduced memory usage
- **Gradient Checkpointing**: 30-40% memory savings
- **Parallel Data Loading**: 4 workers with persistent workers and pinned memory
- **Learning Rate Scheduling**: Linear warmup over 10% of training steps
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Validation-Based Model Selection**: Save best model based on validation loss

**Expected Speedup**: 4-6x faster than baseline implementation

---

## 📁 Project Structure

```
dialogue-summarization-mvp/
│
├── Dialogue_Summarize_MVP__5_.ipynb    # Main project notebook
├── README.md                            # This file
├── project_reflection.md                # Written reflection
├── mvp_comparison_analysis.md           # Detailed version comparison
│
├── checkpoints/                         # Model checkpoints (generated)
│   ├── best_bert_gpt2_summarization.pth
│   └── best_gpt2_ar_summarization.pth
│
└── outputs/                             # Generated summaries and analysis
```

---

## 💡 Key Insights

### Data Analysis
- **Vocabulary Size**: Dialogues (35,409 words) vs. Summaries (18,675 words)
- **Speaking Turns**: Average of 10-11 turns per dialogue
- **Sentence Distribution**: Dialogues (~15-16 sentences) → Summaries (~3-4 sentences)

### Model Insights
1. **Auto-regressive models** may be more suitable for dialogue summarization with limited training
2. **Validation during training** is critical for proper model selection
3. **EOS token configuration** significantly impacts generation quality in encoder-decoder models
4. **Compression challenge**: 10:1 ratio requires substantial abstraction, not just extraction

---

## 🔄 Usage Examples

### Generate a Summary
```python
# Using the trained GPT-2 AR model
dialogue = """
Hannah: Hey, did you see the email about the meeting?
Mike: Yes! It's tomorrow at 2 PM.
Hannah: Perfect. Can you prepare the slides?
Mike: Already done. I'll send them over tonight.
"""

summary = generate_summary_ar(gpt2_model_ar, gpt2_tokenizer_ar, dialogue, device='cuda')
print(f"Summary: {summary}")
# Output: "Hannah and Mike discuss tomorrow's meeting at 2 PM. Mike has prepared the slides."
```

---

## 🛠️ Technologies Used

- **PyTorch** 2.0+ - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained models and tokenizers
- **Datasets** (Hugging Face) - Dataset loading and processing
- **ROUGE Score** - Evaluation metrics
- **NLTK** - Text processing
- **Matplotlib/Seaborn** - Visualization
- **tqdm** - Progress bars

---

## 📊 Evaluation Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level longest common subsequence

All metrics use stemming for improved matching.

---

## 🚧 Future Improvements

- [ ] Implement **BART** or **T5** models (designed specifically for summarization)
- [ ] Increase training to **5-10 epochs** for better convergence
- [ ] Add **early stopping** based on validation ROUGE scores
- [ ] Implement **beam search optimization** with different beam sizes
- [ ] Add **perplexity** as an additional evaluation metric
- [ ] Deploy as **REST API** service with FastAPI
- [ ] Create **web interface** for interactive summarization
- [ ] Experiment with **few-shot learning** approaches
- [ ] Add **human evaluation** framework
- [ ] Implement **multi-lingual** summarization support

---

## 🤝 Contributing

This is an educational MVP project. Feedback and suggestions are welcome!

**Areas for Collaboration:**
- Alternative architecture experiments (BART, T5, Pegasus)
- Hyperparameter tuning strategies
- Additional evaluation metrics
- Production deployment approaches

---

**Built with ❤️ for advancing dialogue understanding through AI**

*Last Updated: November 2025*
