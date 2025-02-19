from setuptools import find_packages, setup

setup(
    name="box_escape_minigrid",
    version="1.1.3",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib", "minigrid"],
)
