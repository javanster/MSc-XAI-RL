from setuptools import find_packages, setup

setup(
    name="box_escape",
    version="1.0.9",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
)
