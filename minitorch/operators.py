"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional


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
    """Multiplies two numbers"""
    return x * y


def id(x):
    """Returns the input unchanged"""
    return x


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y


def neg(x: float) -> float:
    """Negates a number"""
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another"""
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    if lt(x, y):
        return y
    return x


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(x - y) < 1e-2


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if lt(x, 0):
        return exp(x) / (1 + exp(x))
    return 1 / (1 + exp(neg(x)))


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    return max(x, 0)


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    if x == 0:
        raise ValueError
    return 1 / x


def log_back(x: float, y: float):
    """Computes the derivative of log times a second arg"""
    return inv(x) * y


def inv_back(x: float, y: float):
    """Computes the derivative of reciprocal times a second arg"""
    if x == 0:
        raise ValueError
    return -y / (x ** 2)


def relu_back(x: float, y: float):
    """Computes the derivative of ReLU times a second arg"""
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


def map(func: Callable[[float], float], ls: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    return [func(el) for el in ls]


def zipWith(func: Callable[[float, float], float], ls: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    return [func(el, el2) for el, el2 in zip(ls, ls2)]


def reduce(func: Callable[[float, float], float], ls: Iterable[float], initializer: Optional[float] = None) -> float:
    it = iter(ls)
    if initializer:
        temp = initializer
    else:
        try:
            temp = next(it)
        except StopIteration:
            temp = 0
    for el in it:
        temp = func(temp, el)

    return temp


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list"""
    return map(neg, ls)


def addLists(ls: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists"""
    return zipWith(add, ls, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list"""
    return reduce(add, ls, 0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list"""
    return reduce(mul, ls, 1)
