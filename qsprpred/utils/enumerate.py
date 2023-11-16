"""Enumerating a list of items with leading zeros."""
from typing import Sized


def enumerate_with_zeros(items: Sized, prefix: str = "") -> list[str]:
    """Enumerate a list of items with leading zeros.

    Args:
        items (Iterable): List of items to enumerate.
        prefix (str): Prefix to add to each item.

    Returns:
        enumerated_strings (list[str]): List of enumerated strings.
    """
    num_digits = len(str(len(items)))
    enumerated_strings = []
    for i, item in enumerate(items, 0):
        padded_str = str(i).zfill(num_digits)
        if prefix:
            padded_str = prefix + padded_str
        enumerated_strings.append(padded_str)
    return enumerated_strings
