"""
laya_engine_hf.py
-----------------
Shows how to load and run Laya using only the standard Hugging Face
`transformers` library — no laya package required for the raw HF usage.

Install: pip install transformers torch

Usage:
    python laya_engine_hf.py
"""

import torch
from transformers import AutoTokenizer, AutoModel


MODEL_ID = "convaiinnovations/laya"


def load_model(model_id: str = MODEL_ID):
    """
    Download (or load from cache) the tokenizer and model weights from
    Hugging Face Hub. First run downloads ~300 MB; subsequent runs are instant.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModel.from_pretrained(model_id, device_map="auto")
    return tokenizer, model


def tokenize(tokenizer, text: str) -> dict:
    """
    Convert raw text into token IDs that the model understands.
    Returns a dict of PyTorch tensors ready to pass into the model.
    """
    return tokenizer(text, return_tensors="pt")


def forward_pass(model, inputs: dict) -> torch.Tensor:
    """
    Run one single forward pass through the transformer.
    Returns last_hidden_state: shape [batch, sequence_length, embedding_dim].
    No token generation loop — this is what makes Laya fast.
    """
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state


def cls_embedding(hidden_state: torch.Tensor) -> torch.Tensor:
    """
    Extract the [CLS] token embedding (position 0).
    This vector is Laya's learned representation of the whole input.
    """
    return hidden_state[0][0]


def main():
    print("=" * 52)
    print("LAYA  —  Hugging Face Transformers Demo")
    print("=" * 52)

    # 1. Load model and tokenizer
    print(f"\nLoading model: {MODEL_ID}")
    tokenizer, model = load_model()
    print("Model ready.\n")

    # 2. Define example text
    text = "CRITICAL: Database connection pool exhausted causing HTTP 500 errors across all services"
    print(f"Input text:\n  {text}\n")

    # 3. Tokenize
    inputs = tokenize(tokenizer, text)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Tokens ({len(tokens)}):")
    print(f"  {tokens}\n")

    # 4. Forward pass
    hidden = forward_pass(model, inputs)
    print(f"Hidden state shape: {hidden.shape}")
    print(f"  (batch=1, tokens={hidden.shape[1]}, embedding_dim={hidden.shape[2]})\n")

    # 5. CLS embedding (model's summary of the text)
    cls = cls_embedding(hidden)
    print(f"[CLS] embedding (first 8 of {cls.shape[0]} dimensions):")
    print(f"  {cls[:8].tolist()}\n")

    print("=" * 52)
    print("For structured decisions (department, urgency, etc.),")
    print("use the laya package:  pip install laya")
    print("Then run:              python laya_demo.py")
    print("=" * 52)


if __name__ == "__main__":
    main()
