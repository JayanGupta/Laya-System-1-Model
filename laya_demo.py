import sys
from laya_engine import LayaClassifier, get_default_questions


def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "CRITICAL: Database connection pool exhausted causing HTTP 500 errors"
    )

    model     = LayaClassifier()
    questions = get_default_questions()
    state     = {"subject": text[:80], "body": text}
    result    = model.predict(state, questions)

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
