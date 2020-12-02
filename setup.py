#!/usr/bin/env python3
import os
import sys

import setuptools.command.egg_info as egg_info_cmd
from setuptools import setup

SETUP_DIR = os.path.dirname(__file__)
README = os.path.join(SETUP_DIR, "README.md")

try:
    import gittaggers

    tagger = gittaggers.EggInfoFromGit
except ImportError:
    tagger = egg_info_cmd.egg_info

setup_requires = ["torch < 1.8", ]
install_requires = [
    "torch < 1.8", "click < 8", "pandas < 2", 
    "numpy", "scipy", "torch-scatter", "torch-sparse",
    "torch-cluster", "torch-spline-conv", "torch-geometric",
]

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest < 6", "pytest-runner < 5"] if needs_pytest else []

setup(
    name="deepmocca",
    version="1.0.0",
    description="DeepMOCCA",
    long_description=open(README).read(),
    long_description_content_type="text/markdown",
    author="Sara Althubaiti",
    author_email="sara.althubaiti@kaust.edu.sa",
    download_url="https://github.com/bio-ontology-research-group/deepmocca/archive/v1.0.0.tar.gz",
    license="Apache 2.0",
    packages=["deepmocca",],
    package_data={"deepmocca": [],},
    install_requires=install_requires,
    extras_require={},
    setup_requires=setup_requires + pytest_runner,
    tests_require=["pytest<5"],
    entry_points={
        "console_scripts": [
            "deepmocca=deepmocca.main:main",
        ]
    },
    zip_safe=True,
    cmdclass={"egg_info": tagger},
    python_requires="<3.8",
)
