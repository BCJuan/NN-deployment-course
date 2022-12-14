# Third-Party Libraries
from setuptools import find_packages, setup

extras = {"dev": ["pre-commit == 2.18.1"]}

setup(
    name="cpu",
    packages=find_packages(exclude=["tests", "tests.*"], include=["cpu"]),
    install_requires=[
        "torch == 1.12",
        "torchvision",
        "jupyter",
        "matplotlib",
        "opencv-python",
        "onnx",
        "onnxruntime",
        "openvino",
        "openvino-dev",
    ],
    extras_require=extras,
)
