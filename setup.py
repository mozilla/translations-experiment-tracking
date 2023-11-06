#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path

from setuptools import setup


def requirements(path):
    assert os.path.exists(path), "Missing requirements {}".format(path)
    with open(path) as f:
        return f.read().splitlines()


with open("translations_parser/VERSION") as f:
    VERSION = f.read().strip()


with open("README.md") as f:
    LONG_DESC = f.read().strip()


setup(
    name="translations_parser",
    version=VERSION,
    description="A training log parser for the Mozilla translation ML tool",
    long_description=LONG_DESC,
    author="Teklia",
    author_email="team@teklia.com",
    python_requires=">=3.10",
    install_requires=requirements("requirements.txt"),
    packages=["translations_parser"],
    include_package_data=True,
    entry_points={"console_scripts": ["parse_training_logs=translations_parser.parser:main"]},
)
