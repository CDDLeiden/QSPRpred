"""Version of QSPRpred."""
import os

# read verson from file
VERSION = open(os.path.join(os.path.dirname(__file__), 'VERSION.txt')).read().strip()
