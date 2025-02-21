from setuptools import find_packages, setup

setup(
    name="ccr",
    version="5.0.2",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "importlib",
    ],
)
