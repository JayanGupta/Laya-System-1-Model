<div align="center">

<!-- Animated Typing Header -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=34&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&width=900&height=95&lines=Laya+%E2%80%94+System+1+Decision+Model;Sub-35ms+Non-Autoregressive+Text+Classification;Instant+Ticket+Routing+%E2%80%A2+Calibrated+Confidence" alt="Typing SVG" /></a>

<br/>

<!-- Animated Wave Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=220&section=header&text=Laya%20%E2%80%94%20System%201%20Model&fontSize=44&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Non-Autoregressive%20Decision%20Engine%20%7C%20~33ms%20Latency%20%E2%80%A2%20Zero%20Hallucinations&descSize=16&descAlignY=55&descColor=58a6ff" width="100%"/>

<br/>

<!-- Badges Row 1: Repository Status -->
![GitHub last commit](https://img.shields.io/github/last-commit/JayanGupta/Laya-System-1-Model?style=for-the-badge&color=00d9ff&labelColor=0d1117)
![GitHub repo size](https://img.shields.io/github/repo-size/JayanGupta/Laya-System-1-Model?style=for-the-badge&color=7c3aed&labelColor=0d1117)
![GitHub stars](https://img.shields.io/github/stars/JayanGupta/Laya-System-1-Model?style=for-the-badge&color=fbbf24&labelColor=0d1117)
![GitHub forks](https://img.shields.io/github/forks/JayanGupta/Laya-System-1-Model?style=for-the-badge&color=34d399&labelColor=0d1117)

<br/>

<!-- Badges Row 2: Core Technologies -->
[![Model](https://img.shields.io/badge/Model-convaiinnovations%2Flaya-blue?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/convaiinnovations/laya)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-4.40%2B-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-Apache%202.0-22c55e?style=for-the-badge&logo=apache&logoColor=white)
![Latency](https://img.shields.io/badge/Latency-%7E33ms-00D9FF?style=for-the-badge&logo=speedtest&logoColor=white)

<br/>

<!-- Badges Row 3: Developer & Deployment Tooling -->
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

<br/><br/>

<!-- Profile Views Counter -->
![Profile Views](https://komarev.com/ghpvc/?username=JayanGupta&label=Repository+Views&color=00d9ff&style=for-the-badge)

</div>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## About This Repository

> *A clean, production-grade showcase and practical experimentation harness for [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) — the open-source non-autoregressive System 1 decision model that classifies raw text, routes enterprise support tickets, and scores urgency in a single forward pass in ~33 ms.*

Most modern AI solutions (ChatGPT, Claude, LLaMA) rely on **System 2** reasoning: they generate text token-by-token in sequential loops. This is computationally expensive (1,000 to 3,000 ms), costly per token, and prone to hallucinations. 

**Laya embodies System 1 intuition**: fast, non-generative, deterministic, calibrated, and instant.

### Core Architectural Differentiators

| Dimension | Generative LLMs (System 2) | Laya (System 1) |
| :--- | :--- | :--- |
| **Output Type** | Unstructured text stream | Structured JSON payload |
| **Inference Mechanism** | Multi-step autoregressive generation | Single forward pass through network |
| **End-to-End Latency** | 1,000 – 3,000 ms | **~33 ms** (up to 100x faster) |
| **Hallucination Risk** | High (stochastically synthesizes tokens) | **Zero** (purely classifies & scores) |
| **Confidence Scoring** | Uncalibrated token logits | **Calibrated probabilities** via RLCD |
| **Operational Cost** | Variable per-token API billing | **Free, lightweight, fully self-hosted** |
| **Model Footprint** | 7B – 70B+ parameters | **~300 MB compact checkpoint** |
| **Licensing** | Proprietary or restrictive open weights | **Apache 2.0** |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Architectural Pipeline

```mermaid
flowchart TD
    subgraph InputStage["1. Raw Input Ingestion"]
        A["Incoming User Query / Ticket<br/>'CRITICAL: DB pool exhausted with HTTP 500'"]
    end

    subgraph DecisionCore["2. Laya System 1 Neural Engine (~33ms)"]
        B["HuggingFace Tokenizer"] --> C["Non-Autoregressive Transformer Backbone"]
        C --> D["Multi-Head Classification & RLCD Confidence Calibration"]
    end

    subgraph StructuredOutputs["3. Zero-Hallucination Structured Output"]
        D --> E["Department Classification<br/>technical_support (97.4%)"]
        D --> F["Urgency Scoring<br/>CRITICAL"]
        D --> G["Escalation Flag<br/>True"]
    end

    subgraph Downstream["4. Downstream Orchestration"]
        E & F & G --> H["Automated Ticket Routing & Incident Response"]
    end

    A --> B

    style InputStage fill:#0d1117,stroke:#00d9ff,stroke-width:2px,color:#ffffff
    style DecisionCore fill:#0d1117,stroke:#7c3aed,stroke-width:2px,color:#ffffff
    style StructuredOutputs fill:#0d1117,stroke:#34d399,stroke-width:2px,color:#ffffff
    style Downstream fill:#0d1117,stroke:#fbbf24,stroke-width:2px,color:#ffffff
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Quick Start & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JayanGupta/Laya-System-1-Model.git
cd Laya-System-1-Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> On the first execution, model weights (~300 MB) are automatically fetched from Hugging Face Hub and cached locally. All subsequent runs execute instantly offline.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Live CLI Demonstrations

Test any raw text or support ticket in real-time straight from your terminal using `laya_demo.py`:

```bash
# Outage detection
python laya_demo.py "CRITICAL: Database connection pool exhausted causing HTTP 500 errors"

# Billing dispute
python laya_demo.py "Double charged on invoice #INV-88912"

# Security telemetry
python laya_demo.py "Unrecognized root login detected from foreign IP 203.0.113.42"

# General inquiry
python laya_demo.py "How do I export my dashboard analytics to CSV format?"
```

### Sample Real-Time Terminal Output
```text
--- LAYA DECISION ---
Input:       CRITICAL: Database connection pool exhausted causing HTTP 500 errors
Latency:     33.0 ms
Department:  technical_support  (97.4%)
Urgency:     CRITICAL
Escalation:  True
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Programmatic Python Usage

### Option A: Using the High-Level `laya_engine.py` Wrapper
```python
from laya_engine import LayaEngine

engine = LayaEngine()
result = engine.predict("Client portal is rendering blank white screens on mobile devices")

print(f"Department: {result.department} (Confidence: {result.confidence:.1%})")
print(f"Urgency:    {result.urgency}")
print(f"Latency:    {result.latency_ms:.1f} ms")
```

### Option B: Pure Hugging Face Transformers (`laya_engine_hf.py`)
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "convaiinnovations/laya"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

inputs = tokenizer("Refund requested for subscription cancelled last week", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

predicted_class = torch.argmax(probabilities, dim=-1).item()
print("Predicted class index:", predicted_class)
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Repository File Structure

```
Laya-System-1-Model/
|-- README.md                              # Master repository documentation
|-- requirements.txt                       # Project dependencies (torch, transformers)
|-- .gitignore                             # Git ignore specifications
|-- explanation.pdf                        # System 1 architecture & theoretical foundations
|
|-- laya_demo.py                           # Ready-to-run CLI demonstration tool
|-- laya_engine.py                         # Production wrapper with latency telemetry
|-- laya_engine_hf.py                      # Pure Hugging Face transformers implementation
|
|-- experiment/                            # Research & evaluation sandbox
|   `-- laya_notebook.ipynb                # Step-by-step Colab-ready interactive notebook
|
`-- data/                                  # Benchmark test datasets
    |-- tickets_dataset.csv                # Enterprise support tickets (CSV)
    `-- tickets_dataset.json               # Enterprise support tickets (JSON)
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Contributing

Contributions, issues, and feature requests are welcome:

1. Fork the Project (`https://github.com/JayanGupta/Laya-System-1-Model/fork`)
2. Create your Feature Branch (`git checkout -b feature/NewBenchmark`)
3. Commit your Changes (`git commit -m "feat: add multi-label classification benchmark"`)
4. Push to the Branch (`git push origin feature/NewBenchmark`)
5. Open a Pull Request

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Show Your Support

Give a star if this repository helped your understanding of non-autoregressive System 1 models!

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=JayanGupta/Laya-System-1-Model&type=Date&theme=dark)](https://star-history.com/#JayanGupta/Laya-System-1-Model&Date)

</div>

---

<div align="center">

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=120&section=footer" width="100%"/>

<br/>

<b>Engineered for AI Innovation by <a href="https://github.com/JayanGupta">Jayan Gupta</a></b>

<br/><br/>

<img src="https://forthebadge.com/images/badges/built-with-love.svg" />
<img src="https://forthebadge.com/images/badges/made-with-python.svg" />
<img src="https://forthebadge.com/images/badges/open-source.svg" />

</div>
