from setuptools import setup, find_packages

from setuptools import setup, find_packages

setup(
    name="box",
    version="1.0",
    packages=find_packages("pimd_tools"),
    package_dir={"": "pimd_tools"},
    scripts=["bin/box"],
)
