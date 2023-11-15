"""Enumerating a list of items with leading zeros."""


def enumerate_with_zeros(items):
    # Find the number of digits in the length of the list
    num_digits = len(str(len(items)))
    enumerated_strings = []
    for i, item in enumerate(items, 0):
        padded_str = str(i).zfill(num_digits)
        enumerated_strings.append(padded_str)
    return enumerated_strings
