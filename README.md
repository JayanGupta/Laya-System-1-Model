# Laya — System 1 Decision Model

<p align="center">
  <img src="https://img.shields.io/badge/Model-convaiinnovations%2Flaya-blue?logo=huggingface&logoColor=white" alt="HuggingFace"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green?logo=apache&logoColor=white" alt="Apache 2.0"/>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-4.40%2B-yellow?logo=huggingface&logoColor=white" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Latency-%2B%7E33ms-brightgreen" alt="33ms latency"/>
</p>

<p align="center">
  <strong>A clean, open-source showcase of <a href="https://huggingface.co/convaiinnovations/laya">convaiinnovations/laya</a> — the non-autoregressive System 1 decision model that classifies text, routes tickets, and scores urgency in a single forward pass at ~33 ms.</strong>
</p>

---

## What Is Laya?

Most AI models (ChatGPT, Claude, Llama) are **System 2** — they generate text token-by-token in a loop. That takes 1–3 seconds, costs tokens, and can hallucinate.

**Laya is System 1** — it works like human intuition:

| | Generative LLMs | Laya (System 1) |
|---|---|---|
| **Output** | Text stream | Structured JSON |
| **Inference** | Token-by-token generation loop | Single forward pass |
| **Latency** | 1,000 – 3,000 ms | ~33 ms |
| **Hallucinations** | Yes (generates text) | No (never generates) |
| **Probabilities** | Uncalibrated | Calibrated via RLCD |
| **Cost** | Per token API pricing | Free, self-hosted |
| **License** | Varies | Apache 2.0 |

---

## Project Structure

```
Laya/
├── experiment/
│   └── laya_experiment.ipynb   # Step-by-step notebook (Colab ready)
├── data/
│   ├── tickets_dataset.json    # Sample support tickets (JSON)
│   └── tickets_dataset.csv     # Same data in CSV format
├── laya_engine.py              # Wrapper around the laya Python package
├── laya_engine_hf.py           # Pure Hugging Face transformers demo
├── laya_demo.py                # CLI script — classify any text in one command
├── requirements.txt            # All Python dependencies
└── README.md
```

---

## Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/JayanGupta/Laya-System-1-Model.git
cd Laya-System-1-Model

# 2. Install all dependencies
pip install -r requirements.txt
```

> **First run:** The model weights (~300 MB) are downloaded automatically from Hugging Face Hub and cached locally. All subsequent runs are instant.

---

## Scripts

### `laya_demo.py` — Classify any text from the terminal

The fastest way to see Laya in action. Pass any text as an argument.

```bash
# Default example (database outage)
python laya_demo.py

# Billing issue
python laya_demo.py "Double charged on invoice #INV-88912"

# Security breach
python laya_demo.py "Unrecognized login from foreign IP 203.0.113.42"

# Critical outage
python laya_demo.py "CRITICAL: Full production downtime, all services offline"

# General question
python laya_demo.py "How do I export my data to CSV?"
```

**Example output:**

```
--- LAYA DECISION ---
Input:       CRITICAL: Database connection pool exhausted causing HTTP 500 errors
Latency:     33.0 ms
Department:  technical_support  (97.4%)
Urgency:     CRITICAL
Escalation:  True
---------------------

Probability distribution:
  billing               1.0%  
  technical_support    97.4%  ########################################
  security              0.8%  
  general_inquiry       0.8%  
```

---

### `laya_engine.py` — The model wrapper (importable module)

Use this in your own Python code.

```python
from laya_engine import LayaClassifier, get_default_questions

model     = LayaClassifier()            # loads convaiinnovations/laya
questions = get_default_questions()     # department + urgency + escalation

state = {
    "subject": "Double charged on invoice #INV-88912",
    "body":    "Our finance team found a duplicate charge for April. Please refund."
}

result  = model.predict(state, questions)
answers = result["answers"]

print(answers["department"]["choice"])      # billing
print(answers["department"]["confidence"])  # 0.962
print(answers["urgency"]["score"])          # low
print(answers["needs_human_escalation"]["value"])  # False
print(result["latency_ms"])                 # 33.0
```

---

### `laya_engine_hf.py` — Raw Hugging Face transformers usage

Shows the lower-level Hugging Face API: tokenization, forward pass, and embedding extraction — no `laya` package needed.

```bash
python laya_engine_hf.py
```

Or import it in your own code:

```python
from laya_engine_hf import load_model, tokenize, forward_pass, cls_embedding

tokenizer, model = load_model()

text    = "Server is returning HTTP 500 errors"
inputs  = tokenize(tokenizer, text)
hidden  = forward_pass(model, inputs)
cls_vec = cls_embedding(hidden)          # [768-dim] summary vector of the text

print(cls_vec[:8].tolist())
```

**Example output:**

```
====================================================
LAYA  —  Hugging Face Transformers Demo
====================================================

Loading model: convaiinnovations/laya
Model ready.

Input text:
  CRITICAL: Database connection pool exhausted causing HTTP 500 errors across all services

Tokens (17):
  ['[CLS]', 'critical', ':', 'database', 'connection', 'pool', 'exhausted', ...]

Hidden state shape: torch.Size([1, 17, 768])
  (batch=1, tokens=17, embedding_dim=768)

[CLS] embedding (first 8 of 768 dimensions):
  [0.1823, -0.3241, 0.5102, -0.1876, 0.4320, -0.2198, 0.3871, -0.0912]
```

---

### `experiment/laya_experiment.ipynb` — Step-by-step Jupyter Notebook

A fully annotated notebook covering all 9 steps end-to-end.
Open it in VS Code, Jupyter Lab, or upload directly to Google Colab.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JayanGupta/Laya-System-1-Model/blob/main/experiment/laya_experiment.ipynb)

Steps covered:

| # | Step | What it shows |
|---|---|---|
| 1 | Install | `pip install transformers torch laya` |
| 2 | Load model | `AutoTokenizer` + `AutoModel.from_pretrained()` |
| 3 | Tokenize | Text → token IDs → tensor |
| 4 | Forward pass | Single pass through 12 transformer layers |
| 5 | Laya structured output | `laya.load()` → typed JSON decisions |
| 6 | Define state & questions | Input schema + typed question schema |
| 7 | Run inference | `agent.predict()` → department, urgency, escalation |
| 8 | Visualize | Calibrated probability bar chart |
| 9 | Batch testing | 5 tickets → pandas DataFrame |

---

## Two Ways to Use Laya

### Option A — High-level (laya package)

```python
import laya

agent = laya.load("convaiinnovations/laya")
result = agent.predict(
    {"subject": "Server down", "body": "All services unreachable"},
    {
        "department": {
            "type": "choice",
            "instructions": "Which team handles this?",
            "criteria": {
                "technical_support": "Server, database, API issues",
                "billing":           "Invoices and payments",
            }
        }
    }
)
print(result["answers"]["department"]["choice"])  # technical_support
```

### Option B — Low-level (transformers only)

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_id  = "convaiinnovations/laya"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModel.from_pretrained(model_id, device_map="auto")

inputs  = tokenizer("Server is down", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# last_hidden_state shape: [1, num_tokens, 768]
print(outputs.last_hidden_state.shape)
```

---

## References

- **Model on Hugging Face:** [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- **Paper / Blog:** [Convai Innovations](https://convai.com)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
