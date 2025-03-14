from setuptools import find_packages, setup

setup(
    name="gold_run_mini",
    version="1.1.14",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
    package_data={
        "gold_run_mini.envs": ["sprites/*.png"],
    },
)
