..  _installation:

Installation
============

You do not need anything special to install the package. Just run the following to get the latest version and all dependencies:

..  code-block::

    pip install git+https://github.com/CDDLeiden/QSPRpred.git@main

You can also get tags and development snapshots by varying the :code:`@main` part (i.e. :code:`@1.1.0`). After that you can start building models (see :ref:`usage`).

You can test the installation by running the unit test suite:

..  code-block::

    python -m unittest discover qsprpred