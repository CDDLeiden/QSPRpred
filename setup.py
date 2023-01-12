"""
setup.py

Created by: Martin Sicho
On: 24.06.22, 10:33
"""

from distutils.util import convert_path

from setuptools import setup

main_ns = {}
ver_path = convert_path('qsprpred/about.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(version=main_ns['VERSION'], scripts=['scripts/qsprpred'])
