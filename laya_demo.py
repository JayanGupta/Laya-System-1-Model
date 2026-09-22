"""
laya_demo.py
------------
Command-line demo for the Laya decision model.
Install: pip install laya transformers torch

Usage:
    python laya_demo.py
    python laya_demo.py "Double charged on invoice #INV-88912"
    python laya_demo.py "Database connection pool exhausted causing HTTP 500"
    python laya_demo.py "Unrecognized login from foreign IP 203.0.113.42"
"""

import sys
from laya_engine import LayaClassifier, get_default_questions


def main():
    # Read text from CLI argument or use a default example
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "CRITICAL: Database connection pool exhausted causing HTTP 500 errors"
    )

    # Load model from Hugging Face (cached after the first download)
    model     = LayaClassifier()
    questions = get_default_questions()

    # Build state dict — laya expects a dict describing the input context
    state  = {"subject": text[:80], "body": text}
    result = model.predict(state, questions)

    answers = result["answers"]
    dept    = answers["department"]
    urg     = answers["urgency"]
    esc     = answers["needs_human_escalation"]

    print("\n--- LAYA DECISION ---")
    print(f"Input:       {text}")
    print(f"Latency:     {result['latency_ms']} ms")
    print(f"Department:  {dept['choice']}  ({dept['confidence'] * 100:.1f}%)")
    print(f"Urgency:     {urg['score'].upper()}")
    print(f"Escalation:  {esc['value']}")
    print("---------------------\n")

    print("Probability distribution:")
    for label, prob in dept["probabilities"].items():
        bar = "#" * int(prob * 40)
        print(f"  {label:<20} {prob * 100:>5.1f}%  {bar}")
    print()


if __name__ == "__main__":
    main()
