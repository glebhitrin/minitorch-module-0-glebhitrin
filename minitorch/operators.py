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
    """Multiplies two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        Product of the numbers.

    """
    return x * y


def id(x):
    """Returns the input unchanged.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Unchanged number.

    """

    return x


def add(x: float, y: float) -> float:
    """Adds two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        Sum of the numbers.

    """
    return x + y


def neg(x: float) -> float:
    """Negates a number.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Additive inverse.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Checks if one number is less than another.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if the first number is less than the second. False otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True if the numbers are equal. False otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The maximum of the numbers.

    """
    if lt(x, y):
        return y
    return x


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        True is the numbers are close enough. False otherwise.

    """
    return abs(x - y) < 1e-2


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Value of the exponent.

    """
    return math.exp(x)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Value of the sigmoid function.

    """
    if lt(x, 0):
        return exp(x) / (1 + exp(x))
    return 1 / (1 + exp(neg(x)))


def relu(x: float) -> float:
    """Applies the ReLU activation function.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Value of the ReLU function.

    """
    return max(x, 0)


def log(x: float) -> float:
    """Calculates the natural logarithm.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Value of the logarithm function.

    """
    return math.log(x)


def inv(x: float) -> float:
    """Calculates the reciprocal.

    Args:
    ----
        x: Number.

    Returns:
    -------
        Multiplicative inverse.

    """
    if x == 0:
        raise ValueError
    return 1 / x


def log_back(x: float, y: float):
    """Computes the derivative of log times a second arg.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The derivative of log times a second number.

    """
    return inv(x) * y


def inv_back(x: float, y: float):
    """Computes the derivative of reciprocal times a second arg.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The derivative of reciprocal times a second number.

    """
    if x == 0:
        raise ValueError
    return -y / (x ** 2)


def relu_back(x: float, y: float):
    """Computes the derivative of ReLU times a second arg.

    Args:
    ----
        x: First number.
        y: Second number.

    Returns:
    -------
        The derivative of ReLU times a second number.

    """
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
    """Function that applies a given function to each element of an iterable.

    Args:
    ----
        func: A function float -> float.
        ls: An iterable of floats.

    Returns:
    -------
        An iterable of function values.

    """
    return [func(el) for el in ls]


def zipWith(func: Callable[[float, float], float], ls: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Function that combines elements from two iterables using a given function.

    Args:
    ----
        func: A function (float, float) -> float.
        ls: An iterable of floats.
        ls2: An iterable of floats.

    Returns:
    -------
        An iterable of function values.

    """
    return [func(el, el2) for el, el2 in zip(ls, ls2)]


def reduce(func: Callable[[float, float], float], ls: Iterable[float], initializer: Optional[float] = None) -> float:
    """Function that reduces an iterable to a single value using a given function.

    Args:
    ----
        func: A function (float, float) -> float.
        ls: An iterable of floats.
        initializer: The initial value to the function.

    Returns:
    -------
        The result of reducing, float.

    """
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
    """Negate all elements in a list.

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        The list with additive inverses.

    """
    return map(neg, ls)


def addLists(ls: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        ls: An iterable of floats.
        ls2: An iterable of floats.

    Returns:
    -------
        The list with corresponding sums of iterables' elements.

    """
    return zipWith(add, ls, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        The sums of iterable's elements.

    """
    return reduce(add, ls, 0)


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list.

    Args:
    ----
        ls: An iterable of floats.

    Returns:
    -------
        The product of iterable's elements.

    """
    return reduce(mul, ls, 1)
