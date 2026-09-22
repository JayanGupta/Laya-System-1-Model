# Laya: System 1 Decision Model

An open-source (Apache 2.0) project showing how **Laya** (`convaiinnovations/laya`) performs instant text classification, routing, and urgency scoring without requiring paid APIs.

---

## What is Laya and How Does It Work?

Traditional AI models like ChatGPT, Claude, and Llama are **generative (System 2)**. When you ask them to classify a ticket, they generate text word-by-word. This takes 1,000 to 3,000 milliseconds, costs money per token, and sometimes hallucinates or breaks JSON formatting.

**Laya is a System 1 Decision Model**:
1. **Single Forward Pass**: It does not generate text word-by-word. It reads the input text and evaluates your questions in a single mathematical pass (~33 milliseconds).
2. **Zero Hallucinations**: Because it never generates text, it cannot make up fake information or break schemas.
3. **Calibrated Probabilities**: It outputs honest mathematical confidence percentages (e.g., 97% probability) trained using proper scoring rules (RLCD).
4. **100% Free and Local**: Open-source under Apache 2.0. No API keys or credit cards needed.

---

## Project Structure

```
d:\Laya\
├── presentation.html         # Interactive slide deck (open in any web browser)
├── experiment/
│   └── laya_experiment.ipynb # Simple, standalone notebook (Colab ready)
├── data/
│   └── tickets_dataset.json  # Sample tickets dataset
├── laya_engine_hf.py         # Hugging Face Transformers implementation
├── laya_engine.py            # Standard Laya engine wrapper
├── laya_demo.py              # Minimal CLI script (28 lines)
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## How to Run Everything

### 1. Run the Hugging Face Transformers Engine
Execute the dedicated Hugging Face engine script:
```powershell
python laya_engine_hf.py
```
This demonstrates how text is classified into candidate categories with full probability distributions using the Hugging Face ecosystem.

### 2. Run the Minimal CLI Demo
```powershell
# Default test
python laya_demo.py

# Test your own custom text
python laya_demo.py Double charged on my monthly subscription
```

### 3. Run the Jupyter Notebook / Google Colab
Open `experiment/laya_experiment.ipynb` in VS Code, Jupyter, or upload directly to **Google Colab**.
It contains 7 simple steps showing:
- Hugging Face `transformers` tokenization and forward pass.
- Structured decision extraction via Laya.
- Probability distribution plotting with matplotlib.
- Batch evaluation on sample customer tickets.

### 4. View the Slide Presentation
Open `presentation.html` in any web browser:
```powershell
start presentation.html
```
Use the arrow keys (`←` and `→`) to navigate through 7 slides explaining the architecture, benchmarks, and comparison with TypeSafe Jev.

---

## Code Examples

### Direct Decision via Laya (2 Lines)
```python
import laya

# 1. Load the model from Hugging Face
agent = laya.load("convaiinnovations/laya")

# 2. Put text in and get decisions out
state = {"subject": "Double charged on invoice #INV-88912"}
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {
            "billing": "Invoices, charges, refunds",
            "technical_support": "System bugs, crashes"
        }
    }
}

result = agent.predict(state, questions)
print(result["answers"]["department"]["choice"])      # "billing"
print(result["answers"]["department"]["confidence"])  # 0.96
```

### Direct Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModel

model_id = "convaiinnovations/laya"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)

inputs = tokenizer("Double charged on invoice #INV-88912", return_tensors="pt")
outputs = model(**inputs)
```

---

## License
Apache-2.0. Model hosted on Hugging Face: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya).
