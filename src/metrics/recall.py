import numpy as np

from .base import IMetric


def recall_at_k(true_basket: np.ndarray, model_scores: np.ndarray, topk: int):

    scores = model_scores.copy()
    scores[scores.argsort()[:-topk]] = 0
    tp = np.count_nonzero(scores[true_basket])

    recall_score = tp / min(topk, len(true_basket))
    assert 0 <= recall_score <= 1, recall_score
    return recall_score


class Recall(IMetric):

    metric_name: str = "recall"

    def __init__(self, topk=None):
        super().__init__(topk=topk)
        self.cumulative_value = 0.0
        self.n_users = 0

    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        if self.topk is None:
            self.topk = len(model_scores)
        self.cumulative_value += recall_at_k(true_basket, model_scores, self.topk)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_value / self.n_users

    def merge_with_other(self, other_metric_object):
        self.cumulative_value += other_metric_object.cumulative_value
        self.n_users += other_metric_object.n_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0
