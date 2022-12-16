import cv2
import numpy as np
import torch
from addict import Dict
from openvino.tools.pot.api import DataLoader, Metric

label_map = [
    (0, 0, 0),  # background
    (128, 0, 0),  # aeroplane
    (0, 128, 0),  # bicycle
    (128, 128, 0),  # bird
    (0, 0, 128),  # boat
    (128, 0, 128),  # bottle
    (0, 128, 128),  # bus
    (128, 128, 128),  # car
    (64, 0, 0),  # cat
    (192, 0, 0),  # chair
    (64, 128, 0),  # cow
    (192, 128, 0),  # dining table
    (64, 0, 128),  # dog
    (192, 0, 128),  # horse
    (64, 128, 128),  # motorbike
    (192, 128, 128),  # person
    (0, 64, 0),  # potted plant
    (128, 64, 0),  # sheep
    (0, 192, 0),  # sofa
    (128, 192, 0),  # train
    (0, 64, 128),  # tv/monitor
]


def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map


def image_overlay(image, segmented_image):
    alpha = 1  # transparency for the original image
    beta = 0.8  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image


# Create a DataLoader from a CIFAR10 dataset.
class CifarDataLoader(DataLoader):
    def __init__(self, config, dataset):
        """
        Initialize config and dataset.
        :param config: created config with DATA_DIR path.
        """
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)
        self.indexes, self.pictures, self.labels = self.load_data(dataset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Return one sample of index, label and picture.
        :param index: index of the taken sample.
        """
        if index >= len(self):
            raise IndexError

        return (self.indexes[index], self.labels[index]), self.pictures[index].numpy()

    def load_data(self, dataset):
        """
        Load dataset in needed format.
        :param dataset:  downloaded dataset.
        """
        pictures, labels, indexes = [], [], []

        for idx, sample in enumerate(dataset):
            pictures.append(sample[0])
            labels.append(sample[1])
            indexes.append(idx)

        return indexes, pictures, labels


# Custom implementation of classification accuracy metric.


class Accuracy(Metric):

    # Required methods
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = "accuracy@top{}".format(self._top_k)
        self._matches = []

    @property
    def value(self):
        """Returns accuracy metric value for the last model output."""
        return {self._name: self._matches[-1]}

    @property
    def avg_value(self):
        """Returns accuracy metric value for all model outputs."""
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        """Updates prediction matches.
        :param output: model output
        :param target: annotations
        """
        if len(output) > 1:
            raise Exception(
                "The accuracy metric cannot be calculated "
                "for a model with multiple outputs"
            )
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k :]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        """Resets collected matches"""
        self._matches = []

    def get_attributes(self):
        """
        Returns a dictionary of metric attributes {metric_name:
            {attribute_name: value}}.
        Required attributes: 'direction': 'higher-better' or 'higher-worse'
                             'type': metric type
        """
        return {self._name: {"direction": "higher-better", "type": "accuracy"}}
