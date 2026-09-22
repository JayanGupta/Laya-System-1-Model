"""
laya_demo.py
Simple CLI to test Laya predictions.
"""

import sys
from laya_engine import LayaClassifier, get_default_questions

def main():
    model = LayaClassifier()
    questions = get_default_questions()

    # Get input from command line argument or use default
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = "CRITICAL: Database connection pool exhausted causing HTTP 500 errors"

    state = {"subject": text[:50], "body": text}
    result = model.predict(state, questions)
    answers = result["answers"]

    print("\n--- LAYA DECISION ---")
    print(f"Input:       {text}")
    print(f"Latency:     {result['latency_ms']} ms")
    print(f"Department:  {answers['department']['choice']} ({answers['department']['confidence']*100:.1f}%)")
    print(f"Urgency:     {answers['urgency']['score'].upper()}")
    print(f"Escalation:  {answers['needs_human_escalation']['value']}")
    print("---------------------\n")

if __name__ == "__main__":
    main()
