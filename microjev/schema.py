"""Validate fixed task schemas and translate labels into distributions."""

import copy
import math


def validate_schema(schema):
    if not isinstance(schema, dict) or not schema:
        raise ValueError("schema must be a nonempty object of question IDs")
    schema = copy.deepcopy(schema)
    for key, question in schema.items():
        if not isinstance(key, str) or not key:
            raise ValueError("question IDs must be nonempty strings")
        if not isinstance(question, dict):
            raise ValueError(f"{key}: question must be an object")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
                raise ValueError(f"{key}: choice needs 2..255 named criteria")
            if any(not isinstance(k, str) or not k for k in criteria):
                raise ValueError(f"{key}: choice names must be nonempty strings")
            if any(v is not None and not isinstance(v, str) for v in criteria.values()):
                raise ValueError(f"{key}: choice descriptions must be strings or null")
        elif kind == "score":
            if (not isinstance(criteria, list) or not 2 <= len(criteria) <= 10
                    or any(not isinstance(v, str) for v in criteria)):
                raise ValueError(f"{key}: score needs 2..10 ordered descriptions")
        elif kind == "noul":
            if criteria is not None:
                raise ValueError(f"{key}: noul criteria are not supported; use instructions")
        else:
            raise ValueError(f"{key}: unknown type {kind!r}")
        if not isinstance(question.get("instructions", ""), str):
            raise ValueError(f"{key}: instructions must be a string")
    return schema


def cardinality(question):
    return 2 if question["type"] == "noul" else len(question["criteria"])


def targets(schema, labels):
    if not isinstance(labels, dict) or labels.keys() != schema.keys():
        raise ValueError("labels must contain exactly the question IDs from the schema")
    result = {}
    for key, question in schema.items():
        label, count = labels[key], cardinality(question)
        kind = question["type"]
        if kind == "noul":
            if not isinstance(label, (bool, int, float)) or not math.isfinite(label) or not 0 <= label <= 1:
                raise ValueError(f"{key}: noul label must be boolean or a probability in [0, 1]")
            result[key] = [1.0 - float(label), float(label)]
            continue
        if kind == "choice":
            options = list(question["criteria"])
            if not isinstance(label, str) or label not in options:
                raise ValueError(f"{key}: unknown choice label {label!r}")
            index = options.index(label)
        else:
            if type(label) is not int or not 0 <= label < count:
                raise ValueError(f"{key}: score label must be an integer from 0 to {count - 1}")
            index = label
        result[key] = [float(i == index) for i in range(count)]
    return result


def confidence(probabilities):
    """Our own normalized-entropy statistic, NOT TypeSafe's proprietary formula."""
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1.0 - entropy / math.log(len(probabilities))))


def answer(question, probabilities):
    kind = question["type"]
    if kind == "noul":
        return {"type": kind, "noul": probabilities[1]}
    result = {"type": kind, "confidence": confidence(probabilities)}
    if kind == "choice":
        options = list(question["criteria"])
        index = max(range(len(options)), key=lambda i: probabilities[i])
        result.update(choice=options[index], probabilities=dict(zip(options, probabilities)))
    else:
        result.update(
            score=sum(i * p for i, p in enumerate(probabilities)),
            probabilities={str(i): p for i, p in enumerate(probabilities)},
            legend={str(i): level for i, level in enumerate(question["criteria"])},
        )
    return result
