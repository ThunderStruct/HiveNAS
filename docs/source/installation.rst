Installation
==============

.. role:: bash(code)
   :language: bash

.. highlight:: sh

.. highlight:: bash


PyPi (recommended)
------------------

The Python package is hosted on the `Python Package Index (PyPI) <https://pypi.org/project/hivenas/>`_.

The latest published version of HiveNAS can be installed using

.. code-block:: bash

   $ pip install hivenas

To upgrade to the latest published version, use

.. code-block:: bash

   $ pip install --upgrade hivenas


The framework's dependencies are all in the standard **requirements.txt**, which can be installed using

.. code-block:: bash

   $ pip install -r requirements.txt

in the installed package directory.


Manual Installation
-------------------

Simply clone the entire repo and extract the files in the `src` folder, then import them into your project folder.


Or use one of the shorthand methods below:

GIT
~~~

- :code:`cd` into your project directory
- Use :bash:`sparse-checkout` to pull the source files only into your project directory

.. code-block:: bash
   :linenos:

   $ git init HiveNAS
   $ cd HiveNAs
   $ git remote add -f origin https://github.com/ThunderStruct/HiveNAS.git
   $ git config core.sparseCheckout true
   $ echo "src/*" >> .git/info/sparse-checkout
   $ git pull --depth=1 origin master

- Import the newly pulled files into your project folder


SVN
~~~

- :bash:`cd` into your project directory
- :bash:`checkout` the source files::

   $ svn checkout https://github.com/ThunderStruct/HiveNAS/trunk/src
 
- Import the newly checked out files into your project folder

