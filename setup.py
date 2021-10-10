#!/usr/bin/env python3
# This source code is largely adapted from DPR (https://github.com/facebookresearch/DPR) repo
#


from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name='DistDR',
    version='0.1.0',
    long_description=readme,
    long_description_content_type='text/markdown',
    setup_requires=[
        'setuptools>=18.0',
    ],
    install_requires=[
        'cython',
        'faiss-cpu>=1.6.1',
        'filelock',
        'numpy',
        'regex',
        'torch=1.7.0',
        'transformers=2.4.1',
        'tqdm>=4.27',
        'wget',
        'spacy>=2.1.8',
    ],
)
