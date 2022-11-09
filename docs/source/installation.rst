
Installation
==============

**PyPi (recommended)**

:code:`pip install HiveNAS`

**Manual Installation**

Simply clone the entire repo and extract the files in the `HiveNAS` folder, then import them into your project folder.


Or use one of the shorthand methods below:

**GIT**
  - `cd` into your project directory
  - Use `sparse-checkout` to pull the library files only into your project directory::

       git init HiveNAS
       cd HiveNAs
       git remote add -f origin https://github.com/ThunderStruct/HiveNAS.git
       git config core.sparseCheckout true
       echo "HiveNAS/*" >> .git/info/sparse-checkout
       git pull --depth=1 origin master
    
   - Import the newly pulled files into your project folder
**SVN**
  - :code:`cd` into your project directory
  - :code:`checkout` the library files::

      svn checkout https://github.com/ThunderStruct/HiveNAS/trunk/HiveNAS
    
  - Import the newly checked out files into your project folder

