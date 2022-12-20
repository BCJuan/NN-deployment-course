# Third-Party Libraries
from setuptools import find_packages, setup

extras = {"dev": ["pre-commit == 2.18.1"]}

setup(
    name="gpu",
    packages=find_packages(
        exclude=["tests", "tests.*"], include=["nn_deployment_course"]
    ),
    install_requires=[
        "torch ~= 1.12, < 1.13",
        "torchvision",
        "jupyter",
        "matplotlib",
        "opencv-python",
        "onnx ~= 1.11, < 1.12",
        "onnxruntime",
        "nvidia-tensorrt",
        "pytorch-lightning",
        "pycuda",
        "protobuf == 3.20.0",
    ],
    extras_require=extras,
)
