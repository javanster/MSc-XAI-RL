from setuptools import find_packages, setup

setup(
    name="changing_supervisor",
    version="1.1.83",
    packages=find_packages(),
    install_requires=["gymnasium", "numpy", "pygame", "importlib"],
    package_data={
        "changing_supervisor.envs": ["sprites/*.png"],
    },
)
