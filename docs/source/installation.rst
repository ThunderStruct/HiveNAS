Installation
============

Source installation
-------------------

Use Python 3.9 for the pinned dependencies. Clone the ``SwarmNAS`` branch and
create a virtual environment:

.. code-block:: bash

   git clone --branch SwarmNAS https://github.com/ThunderStruct/HiveNAS.git
   cd HiveNAS
   python3.9 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r src/requirements.txt
   cd src

Run the usage examples from ``src/``. On Windows, activate the environment
with ``.venv\Scripts\activate``.

NASBench-101 (optional)
-----------------------

To evaluate candidates using NASBench-101, install its API and point it to
the benchmark file:

.. code-block:: bash

   python -m pip install nasbenchapi
   export NASBENCH101_PATH=/path/to/nasbench_only108.pkl

Set ``USE_NASBENCH`` to ``True`` before constructing ``NASInterface()``.

Published package
-----------------

The HiveNAS package is available on
`PyPI <https://pypi.org/project/hivenas/>`_:

.. code-block:: bash

   python -m pip install hivenas

For the branch documented here, use the source installation above.
