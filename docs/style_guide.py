"""Example file to illustrate our style guide.

We try to adhere to PEP 8 (https://peps.python.org/pep-0008/#introduction) and diverge where we think is appropriate. Summary of the implemented and enforced rules:

- Maximum line length (pep8: 79 characters, but we use 88).
- Use double quotes for strings.
- Imports inside the package should be relative rather than absolute.
- Comments/documentation
    - Write docstrings for all (not just public!) modules, functions, classes, and methods.
    - Use the Google docstring format (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    - multi-line docstrings (pep8, trailing quotes on a new line)
    - single-line docstrings (pep8, trailing quotes on the same line)
- Naming conventions
    - module names (pep8: short, all-lowercase names. Underscores can be used in the module name if it improves readability.)
    - class names (pep 8: CapWords)
    - class method and attribute names (own style: mixedCase, deviation from PEP 8)
    - other variable names (pep 8: lowercase, with words separated by underscores as necessary to improve readability.)
    - function names (pep 8: lowercase, with words separated by underscores as necessary to improve readability.)
- White spaces
    - 4 spaces for indentation
    - two spaces between module member definitions
    - one space between class member definitions
    - no spaces between code or comments in function and method bodies
- Type hints
    - use type hint built-in functionality as introduced in Python 3.9 and later (https://docs.python.org/3/library/typing.html)

Example:

    Some of these rules can be enforced automatically with Autoflake by running the following on your files:

        $ black --line-length 88 style_guide.py
        $ isort --profile black style_guide.py

"""

import numpy
from qsprpred import __version__
from .data.data import QSPRDataset # use relative imports for modules inside the package
from rdkit import Chem

module_variable = "String literal"


def function():
    """An example function without arguments. Note the use of comments to divide the function into logical blocks, not spaces."""

    # example for cycle and variable definition
    example_variable = (
        f"{module_variable + 'x'}_y"  # acceptable to use apostrophes here
    )
    for x in example_variable:
        print(x)
    # another code block that just defines some variables and prints them
    a = 1
    b = 2
    c = 3
    print(a, b, c)


class ExampleClass:
    """Simple example class.

    This class shows style patterns for classes. If a class has static/class members they should be at the top followed by built-in overrides like __init__ or __contains__. All instance specific methods follow after.

    Attributes:
        shortName (str): static class attribute
        someAttribute (str): some class attribute set in init
    """

    shortName = "Some short name."  # static members

    class ClassInClass:
        """Example"""

    @staticmethod
    def staticMethod():
        """Example static method."""

    @classmethod
    def classMethod(cls):
        """Example class method"""

    def __init__(self):
        """The init"""

        self.someAttribute = "Value"

    def randomExampleMethod(self, arg_one : dict[str , float], arg_two : list[str | float]) -> str:
        """Just a method on a class.

        Nothing to see here.

        Args:
            arg_one (str): first argument, a dictionary with string keys and float values
            arg_two (str): second argument, a list of strings or floats

        Returns:
            stuff (str): some stuff
        """

        return "stuff"
