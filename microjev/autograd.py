"""Scalar reverse-mode autodiff, inspired by Karpathy's micrograd/microgpt.

The iterative traversal also handles graphs deeper than Python's recursion limit.
Inference uses plain floats and does not construct these graphs.
"""

import math


class Value:
    __slots__ = ("data", "grad", "edges")

    def __init__(self, data, edges=()):
        self.data = float(data)
        self.grad = 0.0
        self.edges = edges

    @staticmethod
    def coerce(value):
        return value if isinstance(value, Value) else Value(value)

    def __add__(self, other):
        other = self.coerce(other)
        return Value(self.data + other.data, ((self, 1.0), (other, 1.0)))

    __radd__ = __add__

    def __mul__(self, other):
        other = self.coerce(other)
        return Value(self.data * other.data, ((self, other.data), (other, self.data)))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + -self.coerce(other)

    def __rsub__(self, other):
        return self.coerce(other) + -self

    def __pow__(self, exponent):
        return Value(self.data ** exponent, ((self, exponent * self.data ** (exponent - 1)),))

    def __truediv__(self, other):
        return self * self.coerce(other) ** -1

    def exp(self):
        result = math.exp(self.data)
        return Value(result, ((self, result),))

    def log(self):
        return Value(math.log(self.data), ((self, 1.0 / self.data),))

    def relu(self):
        return Value(max(0.0, self.data), ((self, float(self.data > 0)),))

    def backward(self):
        order, seen, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent, _ in node.edges)
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            for parent, derivative in node.edges:
                parent.grad += node.grad * derivative


def number(x):
    return x.data if isinstance(x, Value) else x


def exp(x):
    return x.exp() if isinstance(x, Value) else math.exp(x)


def log(x):
    return x.log() if isinstance(x, Value) else math.log(x)


def softmax(logits):
    offset = max(number(x) for x in logits)
    terms = [exp(x - offset) for x in logits]
    total = sum(terms)
    return [x / total for x in terms]


def cross_entropy(logits, target):
    """Stable negative log likelihood; target may be a soft distribution."""
    offset = max(number(x) for x in logits)
    shifted = [x - offset for x in logits]
    normalizer = log(sum(exp(x) for x in shifted))
    return normalizer - sum(weight * x for weight, x in zip(target, shifted))
