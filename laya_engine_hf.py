import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "convaiinnovations/laya"


def load_model(model_id=MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModel.from_pretrained(model_id, device_map="auto")
    return tokenizer, model


def tokenize(tokenizer, text):
    return tokenizer(text, return_tensors="pt")


def forward_pass(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state


def cls_embedding(hidden_state):
    return hidden_state[0][0]


def main():
    print("=" * 52)
    print("LAYA  --  Hugging Face Transformers Demo")
    print("=" * 52)

    print(f"\nLoading model: {MODEL_ID}")
    tokenizer, model = load_model()
    print("Model ready.\n")

    text = "CRITICAL: Database connection pool exhausted causing HTTP 500 errors"
    print(f"Input text:\n  {text}\n")

    inputs = tokenize(tokenizer, text)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"Tokens ({len(tokens)}):  {tokens}\n")

    hidden = forward_pass(model, inputs)
    print(f"Hidden state shape:  {hidden.shape}")
    print(f"  (batch=1, tokens={hidden.shape[1]}, embedding_dim={hidden.shape[2]})\n")

    cls = cls_embedding(hidden)
    print(f"[CLS] embedding (first 8 of {cls.shape[0]} dims):  {cls[:8].tolist()}\n")

    print("=" * 52)
    print("For structured decisions, run:  python laya_demo.py")
    print("=" * 52)


if __name__ == "__main__":
    main()
