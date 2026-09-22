"""
laya_engine.py
--------------
Minimal wrapper around the official `laya` Python package.
Install: pip install laya transformers torch

Usage:
    from laya_engine import LayaClassifier, get_default_questions

    model     = LayaClassifier()
    questions = get_default_questions()
    result    = model.predict({"subject": "...", "body": "..."}, questions)
    print(result["answers"]["department"]["choice"])
"""

import laya


class LayaClassifier:
    """
    Loads convaiinnovations/laya from Hugging Face and exposes a single
    .predict() method that returns structured JSON decisions.
    """

    def __init__(self, model_id: str = "convaiinnovations/laya"):
        # laya.load() downloads the model from Hugging Face Hub on the first call,
        # then caches it locally for all subsequent calls.
        self.agent = laya.load(model_id)

    def predict(self, state: dict, questions: dict) -> dict:
        """
        Run one single forward pass through the model.

        Parameters
        ----------
        state     : dict  - The text to classify, e.g. {"subject": "...", "body": "..."}
        questions : dict  - Typed questions defined with laya schema

        Returns
        -------
        dict with keys:
          "answers"    - one entry per question with choice/score/value + confidence
          "latency_ms" - inference time in milliseconds
        """
        return self.agent.predict(state, questions)


def get_default_questions() -> dict:
    """
    Returns a standard set of three typed questions for support ticket triage.

    Question types:
      "choice" - pick one label from a set of named criteria
      "score"  - rank on an ordered scale
      "noul"   - Yes/No boolean decision
    """
    return {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {
                "billing":          "Invoices, payments, double charges, refunds",
                "technical_support":"Server outages, database errors, API failures",
                "security":         "Unauthorized access, data breaches, suspicious IPs",
                "general_inquiry":  "Documentation, general questions, how-to requests",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["low", "medium", "high", "critical"],
        },
        "needs_human_escalation": {
            "type": "noul",
            "instructions": "Does this ticket require a human agent to intervene?",
        },
    }
