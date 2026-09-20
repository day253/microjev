"""Synthetic keyword tasks, deliberately tiny and NOT a language benchmark."""

import itertools


SCHEMA = {
    "intent": {"type": "choice", "instructions": "工单属于哪种类型？",
               "criteria": {"refund": "退款", "bug": "报错", "other": "咨询"}},
    "urgent": {"type": "noul", "instructions": "工单是否需要加急？"},
    "severity": {"type": "score", "instructions": "影响程度",
                 "criteria": ["轻微", "中等", "严重"]},
}


def dataset(split="train"):
    templates = {
        "train": ("类型{intent}，{urgency}，影响{severity}",
                  "{urgency}：{intent}，影响{severity}",
                  "工单：{intent}，影响{severity}，{urgency}"),
        "validation": ("工单{intent}，{urgency}，影响{severity}",),
        "test": ("工单：{urgency}，{intent}，影响{severity}",),
    }
    rows = []
    for (label, intent), urgent, severity in itertools.product(
            (("refund", "退款"), ("bug", "报错"), ("other", "咨询")), (False, True), range(3)):
        for template in templates[split]:
            rows.append({
                "text": template.format(intent=intent, urgency="加急" if urgent else "普通",
                                        severity=("轻微", "中等", "严重")[severity]),
                "labels": {"intent": label, "urgent": urgent, "severity": severity},
            })
    return rows
