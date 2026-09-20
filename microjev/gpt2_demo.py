"""English synthetic support tasks for the English GPT-2 checkpoint."""

import itertools


SCHEMA = {
    "intent": {"type": "choice", "instructions": "Classify the customer's request.",
               "criteria": {"refund": "Request money back", "bug": "Report a technical error", "other": "Ask for information"}},
    "urgent": {"type": "noul", "instructions": "Does the customer explicitly request urgent handling? Do not infer urgency from impact."},
    "severity": {"type": "score", "instructions": "Rate the explicitly stated operational impact.",
                 "criteria": ["Minor inconvenience", "Partial disruption with workaround", "Work completely blocked"]},
}


def dataset(split="train"):
    phrases = {
        "train": {
            "intent": [["I want a refund for this purchase.", "Please return the money I paid."],
                       ["The app crashes when I open it.", "I cannot sign in because of an error."],
                       ["Where can I read the product documentation?", "I have a general question about your plans."]],
            "urgent": [["There is no rush.", "You can handle this whenever convenient."],
                       ["This needs immediate attention.", "Please handle this urgently."]],
            "severity": [["Only a small inconvenience; I can keep working.", "The impact is minor and my work continues."],
                         ["Some features are unavailable but there is a workaround.", "Work is partly disrupted, but I can use an alternative."],
                         ["All work is blocked and there is no workaround.", "Nobody can work and everything is stopped."]],
        },
        "validation": {
            "intent": [["I would like my payment refunded."], ["An application error prevents login."], ["Could you explain the available subscription plans?"]],
            "urgent": [["Take your time; this is not urgent."], ["Please prioritize this immediately."]],
            "severity": [["This is a minor inconvenience and I can still work."], ["Part of our work is disrupted; a workaround exists."], ["Our work is completely blocked with no alternative."]],
        },
        "test": {
            "intent": [["Could you send back the amount charged for my order?"], ["The software fails with an error each time it starts."], ["Can you tell me how your pricing works?"]],
            "urgent": [["No need to hurry; a reply later is fine."], ["I need you to deal with this right away."]],
            "severity": [["It is a small annoyance without interrupting our work."], ["Several functions are affected, but an alternative lets us continue."], ["The entire team is unable to do any work and there is no alternative."]],
        },
    }[split]
    rows = []
    templates = ("{intent} {urgent} {severity}", "{severity} {intent} {urgent}") if split == "train" else ("{intent} {urgent} {severity}",)
    for intent, urgent, severity in itertools.product(range(3), range(2), range(3)):
        for words in itertools.product(phrases["intent"][intent], phrases["urgent"][urgent], phrases["severity"][severity]):
            for template in templates:
                rows.append({"text": template.format(intent=words[0], urgent=words[1], severity=words[2]),
                             "labels": {"intent": ("refund", "bug", "other")[intent], "urgent": bool(urgent), "severity": severity}})
    return rows
