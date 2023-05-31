"""Example file to illustrate our style guide.

We try to adhere to PEP 8 (https://peps.python.org/pep-0008/#introduction) and diverge where we think is appropriate. Summary of the implemented and enforced rules:

- Maximum line length (pep8: 79 characters, but we use 88).
- Use double quotes for strings.
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
"""

# external imports
import numpy
from rdkit import Chem

# imports from the package itself
from qsprpred import __version__
from qsprpred.data import QSPRDataset
#
# three spaces between imports and code definitions
#
module_variable = "String literal"

def function():
    """An example function without arguments."""
    # space after docstring
    example_variable = f"{module_variable + 'x'}_y" # acceptable to use apostrophes here
    
    for x in example_variable:
        print(x)
#
# two spaces between definitions in modules
class ExampleClass:
    """Simple example class.
    
    This class shows style patterns for classes. If a class has static/class members they should be at the top followed by built-in overrides like __init__ or __contains__. All instance specific methods follow after.
    
    Attributes:
        shortName (str): static class attribute
        someAttribute (str): some class attribute set in init
    """
    
    shortName = "Some short name." # static members
    # one space between class definitions
    class ClassInClass:
        """Example"""
        
        pass
    
    @staticmethod
    def staticMethod():
        """Example static method."""
        
        pass
    
    @classmethod
    def classMethod(cls):
        """Example class method"""
        
        pass
    
    def __init__():
        """The init"""
        
        self.someAttribute = "Value"
    
    def randomExampleMethod(self, arg_one, arg_two):
        """Just a method on a class.
        
        Nothing to see here.
        
        Args:
            arg_one (str): first argument
            arg_two (str): second argument
        
        Returns:
            stuff (str): some stuff
        """
        
        return "stuff"
        
    
    
