"""Example file to illustrate our style guide.

We try to adhere to PEP 8 (https://peps.python.org/pep-0008/#introduction) and diverge
where we think is appropriate. Summary of the implemented and enforced rules:

- Maximum line length (pep8: 79 characters, but we use 88). Docstrings are included.
- Use double quotes for strings.
- Imports inside the package should be relative rather than absolute.
- Comments/documentation
    - Write docstrings for all (not just public!) modules, functions, classes,
      and methods.
    - Use the Google docstring format
      (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    - multi-line docstrings (pep8, trailing quotes on a new line)
    - single-line docstrings (pep8, trailing quotes on the same line)
- Naming conventions
    - module names (pep8: short, all-lowercase names. Underscores can be used in the
      module name if it improves readability.)
    - class names (pep 8: CapWords)
    - class method and attribute names (own style: mixedCase, deviation from PEP 8)
    - other variable names (pep 8: lowercase, with words separated by underscores as
      necessary to improve readability.)
    - function names (pep 8: lowercase, with words separated by underscores as necessary
      to improve readability.)
- White spaces
    - 4 spaces for indentation
    - two spaces between module member definitions
    - one space between class member definitions
    - no spaces between code or comments in function and method bodies
- Type hints
    - use type hint built-in functionality as introduced in Python 3.9 and later
      (https://docs.python.org/3/library/typing.html)
- Code formatting
    - Use yapf for code formatting. See the style configuration under [tool.yapf] in 
      pyproject.toml. You can run yapf on your files with the following command:
        `$ yapf --style pyproject.toml -i <path/to/file>`
    - Use isort for sorting imports. See the style configuration under [tool.isort] in
      pyproject.toml. You can run isort on your files with the following command:
        `$ isort --profile black <path/to/file>`
    - Use Ruff for linting. Some of the rules are suitable for automatic fixing. For 
      doing this, run the following command:
        `$ ruff --fix <path/to/file>`
    - Our current list of rules is under [tool.ruff] in pyproject.toml, and is an 
      adaptation of the configuration file from the Pandas project.
- Dev contributions
    - Use pre-commit hooks to automatically check and fix your code. See the
      .pre-commit-config.yaml file for the list of hooks. This will require an 
      environment with the correct `[dev]` dependencies installed. These can be 
      installed through the command `$ pip install -e .[dev]`.
    - To install the pre-commit hooks, run the following command:
        `$ pre-commit install`
    - To run the pre-commit hooks on all files, run the following command:
        `$ pre-commit run --all-files`
    - The `style.sh` script in the root directory of the repository can also be used to
    autoformat files on demand, i.e.:
        `$ ./style.sh qsprpred/data/*.py`
"""

import numpy
from rdkit import Chem

from qsprpred import __version__

from .data.data import (  # use relative imports for modules inside the package
    QSPRDataset,
)

module_variable = "String literal"


def function():
    """An example function without arguments.

    Note the use of comments to divide the function into logical blocks, not spaces.
    """

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

    This class shows style patterns for classes. If a class has static/class members
    they should be at the top followed by built-in overrides like `__init__` or
    `__contains__`. All instance specific methods follow after.

    Attributes:
        shortName (str): static class attribute
        someAttribute (str): some class attribute set in init
    """

    shortName = "Some short name."  # static members

    class ClassInClass:
        """Example class."""

    @staticmethod
    def staticMethod():
        """Example static method."""

    @classmethod
    def classMethod(cls):
        """Example class method."""

    def __init__(self):
        """The init."""

        self.someAttribute = "Value"

    def randomExampleMethod(
        self, arg_one: dict[str, float], arg_two: list[str | float]
    ) -> str:
        """Just a method on a class.

        Nothing to see here.

        Args:
            arg_one (str): first argument, dictionary with string keys and float values
            arg_two (str): second argument, list of strings or floats

        Returns:
            stuff (str): some stuff
        """

        return "stuff"
