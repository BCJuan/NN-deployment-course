# Third-Party Libraries
from setuptools import find_packages, setup

extras = {"dev": ["pre-commit == 2.18.1"]}

setup(
    name="nn-deployment-course",
    packages=find_packages(
        exclude=["tests", "tests.*"], include=["nn_deployment_course"]
    ),
    install_requires=[
        "torch",
        "torchvision",
        "jupyter",
        "tqdm",
    ],
    extras_require=extras,
)
