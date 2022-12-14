# Third-Party Libraries
from setuptools import find_packages, setup

extras = {"dev": ["pre-commit == 2.18.1"]}

setup(
    name="cpu",
    packages=find_packages(
        exclude=["tests", "tests.*"], include=["nn_deployment_course"]
    ),
    install_requires=[
        "torch",
        "torchvision",
        "jupyter",
        "matplotlib",
        "opencv-python",
        "onnx",
    ],
    extras_require=extras,
)
