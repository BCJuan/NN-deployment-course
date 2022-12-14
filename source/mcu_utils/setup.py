# Third-Party Libraries
from setuptools import find_packages, setup

extras = {"dev": ["pre-commit == 2.18.1"]}

setup(
    name="mcu",
    packages=find_packages(
        exclude=["tests", "tests.*"], include=["nn_deployment_course"]
    ),
    install_requires=[
        "torch == 1.7.0",
        "torchvision == 0.8.0",
        "jupyter",
        "tqdm",
        "torch2cmsis @ git+https://github.com/BCJuan/torch2cmsis.git@master",
        "matplotlib",
    ],
    extras_require=extras,
)
