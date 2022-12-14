# NN Deployment Course

## Structure

The course is divided in three parts: MCUs, CPU, and embedded GPUs.

## Installation

For each part it is recommended that you create a separate conda or pip environment for each chapter. For example, with conda 

````
cd mcu
conda create -n mcu python=3.8
conda activate mcu
pip install -e .[dev]
````

Then, you can run the notebooks in each chapter.

In the GPU case, to be able to install the nvidia-tensorrt package remember to add `--extra-index-url https://pypi.ngc.nvidia.com` to the pip installation command.