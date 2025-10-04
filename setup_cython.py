#!/usr/bin/env python3

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "cs336_basics.tokenizer_cy",
        ["cs336_basics/tokenizer_cy.pyx"],
        include_dirs=[np.get_include(),"/usr/include/python3.11/"],
        language_level=3,
    )
]

# Setup
setup(
    name="cs336_basics_cython",
    packages=['cs336_basics'],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
            'embedsignature': True,
        }
    ),
    zip_safe=False,
)
