from setuptools import find_packages, setup

setup(
    name="gem_collector",
    version="3.0.12",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "importlib",
    ],
    package_data={
        "gem_collector.envs": ["sprites/*.png"],
    },
)
