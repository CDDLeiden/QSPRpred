..  _installation-guide:

Installation
============

You do not need anything special to install the package . Just run the following (with python >= 3.10) to get the latest version and all basic dependencies:

..  code-block::

    pip install qsprpred

You can also get tags and development snapshots by varying the :code:`@main` part (i.e. :code:`@1.0.0`). After that you can start building models (see :ref:`cli-usage`).

Note that this will install the basic dependencies, but not the optional dependencies.
If you want to use the optional dependencies, you can install the package with an
option:

..  code-block::
    
    pip install qsprpred[<option>]

The following options are available:

- extra : include extra dependencies for PCM models and extra descriptor sets from
  packages other than RDKit
- deep : include deep learning models (torch and chemprop)
- pyboost : include pyboost model (requires cupy, ``pip install cupy-cudaX``, replace X
  with your `cuda version <https://docs.cupy.dev/en/stable/install.html>`_, you can obtain
  cude toolkit from Anaconda as well: ``conda install cudatoolkit``)
- full : include all optional dependecies (requires cupy, ``pip install cupy-cudaX``,
  replace X with your `cuda version <https://docs.cupy.dev/en/stable/install.html>`_)

You can test the installation by running the unit test suite:

..  code-block:: bash

    python -m unittest discover qsprpred

Note that this can potentially take a long time and some tests may require you to have
additional dependencies installed. However, you can also test each module separately:

..  code-block:: bash

    python -m unittest qsprpred.data.tables.tests