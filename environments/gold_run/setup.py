from setuptools import find_packages, setup

setup(
    name="gold_run",
    version="2.0.2",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
    package_data={
        "gold_run.envs": ["sprites/*.png"],
    },
)
