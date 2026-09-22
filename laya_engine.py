import laya

class LayaClassifier:
    def __init__(self, model_id="convaiinnovations/laya"):
        self.agent = laya.load(model_id)

    def predict(self, state, questions):
        return self.agent.predict(state, questions)


def get_default_questions():
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
