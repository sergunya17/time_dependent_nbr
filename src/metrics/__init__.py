from .base import IMetric
from .recall import Recall
from .ndcg import NDCG


METRICS = {
    Recall.metric_name: Recall,
    NDCG.metric_name: NDCG,
}
