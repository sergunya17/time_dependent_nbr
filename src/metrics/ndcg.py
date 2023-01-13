import numpy as np

from .base import IMetric


def dcg_at_k(relevance: np.ndarray, topk: int):
    relevance = relevance[:topk]
    gain = 2**relevance - 1
    discounts = np.log2(np.arange(2, relevance.size + 2))
    dcg = np.sum(gain / discounts)
    return dcg


def ndcg_at_k(true_basket: np.ndarray, model_scores: np.ndarray, topk: int):

    scores = model_scores.copy()
    predicted_items = scores.argsort()[::-1]
    relevance = np.isin(predicted_items, true_basket).astype(int)

    actual_dcg = dcg_at_k(relevance, topk)
    sorted_relevance = np.sort(relevance)[::-1]
    best_dcg = dcg_at_k(sorted_relevance, topk)
    ndcg = actual_dcg / best_dcg

    assert 0 <= ndcg <= 1, ndcg
    return ndcg


class NDCG(IMetric):

    metric_name: str = "ndcg"

    def __init__(self, topk=None):
        super().__init__(topk=topk)
        self.cumulative_value = 0.0
        self.n_users = 0

    def add_recommendations(self, true_basket: np.ndarray, model_scores: np.ndarray):
        if self.topk is None:
            self.topk = len(model_scores)
        self.cumulative_value += ndcg_at_k(true_basket, model_scores, self.topk)
        self.n_users += 1

    def get_metric_value(self):
        return self.cumulative_value / self.n_users

    def merge_with_other(self, other_metric_object):
        self.cumulative_value += other_metric_object.cumulative_value
        self.n_users += other_metric_object.n_users

    def reset(self):
        self.cumulative_value = 0.0
        self.n_users = 0
