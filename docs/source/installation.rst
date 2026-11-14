Installation
============

Source installation
-------------------

Use Python 3.9 for the pinned dependencies. Clone the ``main`` branch and
create a virtual environment:

.. code-block:: bash

   git clone --branch main https://github.com/ThunderStruct/HiveNAS.git
   cd HiveNAS
   python3.9 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r src/requirements.txt
   cd src

Run the usage examples from ``src/``. On Windows, activate the environment
with ``.venv\Scripts\activate``.

Published package
-----------------

The HiveNAS package is available on
`PyPI <https://pypi.org/project/hivenas/>`_:

.. code-block:: bash

   python -m pip install hivenas

For the branch documented here, use the source installation above.
