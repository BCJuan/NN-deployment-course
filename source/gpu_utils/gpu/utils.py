import os
from typing import List, Tuple, Union

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from torch import Tensor, nn
from torch.utils import data as utilsdata
from torchvision import transforms
from torchvision.datasets import CIFAR10


def to_GiB(val: int) -> int:
    """Convert Gigabytes to bytes"""
    return val * (1 << 28)


def configure_constraints(config, obey=False, prefer=False):
    """Sets rules for following type constraints."""
    if obey:
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    if prefer:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    return config


def configure_quantization_and_inputs(
    config, network, fp16, inputs_fp16, int8, inputs_int8
):
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        if inputs_fp16:
            # TODO: extend to multiple outputs
            network.get_input(0).dtype = trt.DataType.HALF
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if inputs_int8:
            # TODO: extend to multiple outputs
            network.get_input(0).dtype = trt.DataType.INT8
    return config, network


_MODULE_CONTAINERS = (nn.Sequential, nn.ModuleList, nn.ModuleDict)


def return_pruning_params(model, parameters=None):
    if parameters is None:
        parameters = ["weight"]
    current_modules = [
        (n, m)
        for n, m in model.named_modules()
        if not isinstance(m, _MODULE_CONTAINERS)
    ]
    parameters_to_prune = [
        (m, p)
        for p in parameters
        for n, m in current_modules
        if getattr(m, p, None) is not None and isinstance(m, nn.Conv2d)
    ]
    return parameters_to_prune


def numpy_collate(batch):
    """Stacks arrays in a batch for calibration"""
    return np.stack(batch, axis=0).ravel()


class DummyDataset(utilsdata.Dataset):
    def __init__(self, calibration=False):
        super().__init__()
        self.data = np.random.randint(0, 255, (3, 256, 256))
        self.calibration = calibration

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        if not self.calibration:
            return self.data, None
        else:
            return self.data


# create DataLoader from CIFAR10 dataset
class CifarDataLoader(utilsdata.Dataset):
    """CIFAR10 Dataset for benchmarking drop with Int8 Quantization"""

    def __init__(self, root: str, calibration: bool):
        """
        Initialize config and dataset.
        Args:
            root: Path where data is stored
            calibration: Makes the __getitem__ method only return
                the input array
        """
        super().__init__()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        self.dataset = CIFAR10(
            root=root,
            train=False,
            transform=transform,
            download=True,
        )
        self.indexes, self.pictures, self.labels = self.load_data()
        self._calibration = calibration

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Union[np.ndarray, Tuple[List[int], np.ndarray]]:
        """
        Returns one sample of index, label and picture
        Args:
            index: Index of the picked sample
        """
        if index >= len(self):
            raise IndexError
        if not self._calibration:
            values = self.labels[index], self.pictures[index].numpy()
        else:
            values = self.pictures[index].numpy()
        return values

    def load_data(self) -> Tuple[List[int], List[Tensor], List[int]]:
        """
        Load dataset in needed format.

        Returns
            indexes: List of the indexes for each sample
            pictures: List of image tensors
            labels: LIst of image labels
        """
        pictures, labels, indexes = [], [], []

        for idx, sample in enumerate(self.dataset):
            pictures.append(sample[0])
            labels.append(sample[1])
            indexes.append(idx)

        return indexes, pictures, labels


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self, calib_set: utilsdata.Dataset, cache_file_name: str, batch_size: int
    ):
        super().__init__()
        self.cache_file = cache_file_name
        self.calib_set = calib_set
        sample = next(iter(calib_set))
        self.current_index = 0
        self.batch_size = batch_size

        # Allocate enough memory for a whole batch.
        self.device_input = cuda.mem_alloc(sample.nbytes * self.batch_size)

        self.loader = iter(
            utilsdata.DataLoader(
                self.calib_set,
                batch_size=self.batch_size,
                collate_fn=numpy_collate,
            )
        )

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.calib_set):
            return None
        batch = next(self.loader)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(
    engine: trt.ICudaEngine,
) -> Tuple[
    List[HostDeviceMem],
    List[HostDeviceMem],
    List[int],
    Union[cuda.Stream, None],
]:
    """Reserves memory in Device for inputs, outputs.

    Args:
        engine: TensorRT Cuda Engine with which to allocate memory.

    Returns:
        inputs: List of HostDevice memory maps for inputs.
        outputs: List of HostDevice memory maps for outputs.
        bindings: List of links between memory maps and engine.
        stream: Cuda stream for memory management and async inference.
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding_ in engine:
        max_shape = engine.get_binding_shape(binding_)
        size = trt.volume(max_shape)
        dtype = trt.nptype(engine.get_binding_dtype(binding_))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding_):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(
    context: trt.IExecutionContext,
    bindings: List[int],
    inputs: List[HostDeviceMem],
    outputs: List[HostDeviceMem],
    stream: cuda.Stream,
) -> List[np.ndarray]:

    # Transfer input data to the GPU.
    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for out in outputs:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
