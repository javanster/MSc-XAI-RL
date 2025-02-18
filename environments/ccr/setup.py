from setuptools import find_packages, setup

setup(
    name="ccr",
    version="5.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "importlib",
    ],
)
