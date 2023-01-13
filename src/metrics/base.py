from abc import abstractmethod

import numpy as np


class IMetric:
    """
    Abstract class that should be used as superclass of all metrics requiring an object,
    therefore a state, to be computed
    """

    @property
    @abstractmethod
    def metric_name(self):
        pass

    def __init__(self, topk=None):
        self.topk = topk
        self.cumulative_value = 0.0
        self.n_users = 0

    def __str__(self):
        return "{:.4f}".format(self.get_metric_value())

    def get_metric_name(self):
        return f"{self.metric_name}@{self.topk:03d}"

    @abstractmethod
    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        raise NotImplementedError()

    @abstractmethod
    def get_metric_value(self):
        raise NotImplementedError()

    @abstractmethod
    def merge_with_other(self, other_metric_object):
        raise NotImplementedError()

    @abstractmethod
    def reset(self):
        raise NotImplementedError()
