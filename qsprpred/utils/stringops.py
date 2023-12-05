"""Enumerating a list of items with leading zeros."""
from typing import Sized


def generate_padded_index(items: Sized, prefix: str | None = "") -> list[str]:
    """Generate zero-padded index strings for a list of items.

    Args:
        items (Iterable): List of items to generate index for.
        prefix (str, optional): Prefix to add to each index.

    Returns:
        list[str]: List of zero-padded index strings.
    """
    num_digits = len(str(len(items)))
    enumerated_strings = []
    for i, item in enumerate(items, 0):
        padded_str = str(i).zfill(num_digits)
        if prefix:
            padded_str = f"{prefix + '_' if prefix is not None else ''}{padded_str}"
        enumerated_strings.append(padded_str)
    return enumerated_strings
