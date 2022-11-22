
# ![HiveNAS Logo](https://i.imgur.com/mDTdNim.jpg)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThunderStruct/HiveNAS/blob/main/colab/HiveNas.ipynb) [![Platform](https://img.shields.io/badge/python-v3.7-green)](https://github.com/ThunderStruct/HiveNAS) [![pypi](https://img.shields.io/badge/pypi%20package-0.1.4-lightgrey.svg)](https://pypi.org/project/HiveNAS/0.1.4/) [![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/ThunderStruct/HiveNAS/blob/master/LICENSE) [![Read the Docs](https://readthedocs.org/projects/hivenas/badge/?version=latest)](https://hivenas.readthedocs.io/en/latest/)

A feature-rich, Neural Architecture Search framework based on Artificial Bee Colony optimization

------------------------

## Getting Started

**HiveNAS** ([preprint](https://arxiv.org/abs/2211.10250)) is a modular NAS framework that can find and optimize a neural architecture with state-of-the-art performance.

### Installation

#### PyPi (recommended)

The Python package is hosted on the [Python Package Index (PyPI)](https://pypi.org/project/hivenas/).

The latest published version of HiveNAS can be installed using

```sh
pip install HiveNAS
```

#### Manual Installation
Simply clone the entire repo and extract the files in the `HiveNAS` folder, then import them into your project folder.

Or use one of the shorthand methods below
##### GIT
  - `cd` into your project directory
  - Use `sparse-checkout` to pull the library files only into your project directory
    ```sh
    git init HiveNAS
    cd HiveNAs
    git remote add -f origin https://github.com/ThunderStruct/HiveNAS.git
    git config core.sparseCheckout true
    echo "HiveNAS/*" >> .git/info/sparse-checkout
    git pull --depth=1 origin master
    ```
   - Import the newly pulled files into your project folder
##### SVN
  - `cd` into your project directory
  - `checkout` the library files
    ```sh
    svn checkout https://github.com/ThunderStruct/HiveNAS/trunk/HiveNAS
    ```
  - Import the newly checked out files into your project folder
  

### Documentation

Detailed examples and the full API docs are [hosted on Read the Docs](https://hivenas.readthedocs.io/en/latest/).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/ThunderStruct/HiveNAS/blob/master/LICENSE) file for details


