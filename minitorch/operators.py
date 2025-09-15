"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x):
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    if lt(x, y):
        return y
    return x


def is_close(x: float, y: float) -> bool:
    return 1e-2 > x - y > -1e-2


def exp(x: float) -> float:
    return math.e ** x


def sigmoid(x: float) -> float:
    if lt(x, 0):
        return exp(x) / (1 + exp(x))
    return 1 / (1 + exp(neg(x)))


def relu(x: float) -> float:
    return max(x, 0)


def log(x: float) -> float:
    return math.log(x)


def log_back(x: float, y: float):
    return


def inv(x: float) -> float:
    if x == 0:
        raise ValueError
    return 1 / x


def inv_back(x: float, y: float):
    return


def relu_back(x: float, y: float):
    if lt(0, x):
        return y
    return 0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
