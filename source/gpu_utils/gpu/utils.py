import os

import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
from torch import nn
from torch.utils import data as utilsdata


def to_GiB(val: int) -> int:
    """Convert Gigabytes to bytes"""
    return val * (1 << 28)


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
        self.data = np.random.randint(0, 255, (10, 3, 256, 256))
        self.calibration = calibration

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        if not self.calibration:
            return self.data, None
        else:
            return self.data


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_set: utilsdata.Dataset, batch_size: int):
        super().__init__()
        self.cache_file = "cache.txt"
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
