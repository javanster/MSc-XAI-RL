from setuptools import find_packages, setup

setup(
    name="box_escape_lite",
    version="1.0.1",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
)
