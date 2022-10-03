import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='HiveNAS',
    version='0.1.0',
    author='Mohamed Shahawy',
    author_email='mohamedshahawy@icloud.com',
    description='A Neural Architecture Search framework based on Artificial Bee Colony optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ThunderStruct/HiveNAS',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Utilities',
        'Framework :: Jupyter',
        'Framework :: IPython',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux'
    ],
    keywords='nas neural architecture search optimization automl',
    package_dir = {'': 'src'},
    packages = setuptools.find_packages(where='src'),
    python_requires = '>=3.7'
)