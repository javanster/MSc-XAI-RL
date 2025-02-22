from setuptools import find_packages, setup

setup(
    name="minecart_counter",
    version="2.0.5",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
    package_data={
        "minecart_counter.envs": ["sprites/*.png"],
    },
)
