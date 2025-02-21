from setuptools import find_packages, setup

setup(
    name="gold_run",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
    package_data={
        "gold_run.envs": ["sprites/*.png"],
    },
)
