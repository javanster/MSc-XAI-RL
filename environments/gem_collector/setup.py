from setuptools import find_packages, setup

setup(
    name="gem_collector",
    version="2.0.9",
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
