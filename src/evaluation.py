from collections import defaultdict
import sys
import time
from typing import List

import numpy as np
import pandas as pd

from src.metrics import METRICS, IMetric
from src.models import IRecommender, IRecommenderNextTs


class Evaluator:
    def __init__(
        self,
        dataset_df: pd.DataFrame,
        cutoff_list: List[int],
        metric_names: List[str] | None = None,
        batch_size: int = 1000,
        save_user_metrics: bool = False,
        verbose=False,
    ):

        super().__init__()
        self.verbose = verbose
        self.save_user_metrics = save_user_metrics

        self.cutoff_list = cutoff_list.copy()
        self._max_cutoff = max(self.cutoff_list)

        self.dataset_df = dataset_df
        self._num_users_to_evaluate = self.dataset_df.shape[0]

        self.batch_size = min(batch_size, self._num_users_to_evaluate)

        self.metrics: List[IMetric] = []
        if metric_names is None:
            metric_names = list(METRICS.keys())
        for metric_name in metric_names:
            metric_cls = METRICS[metric_name]
            for cutoff in self.cutoff_list:
                self.metrics.append(metric_cls(topk=cutoff))

        if self.save_user_metrics:
            self.user_metrics = defaultdict(list)

        self._start_time = np.nan
        self._start_time_print = np.nan
        self._n_users_evaluated = np.nan

    def _print(self, string: str):
        if self.verbose:
            print(f"{type(self).__name__}: {string}")

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()
        if self.save_user_metrics:
            self.user_metrics = defaultdict(list)

    def create_metrics_dict(self):
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric.get_metric_name()] = metric.get_metric_value()
        return metrics_dict

    def _compute_metrics_on_recommendation_list(self, true_baskets_batch: pd.Series, scores_batch):
        assert len(true_baskets_batch) == scores_batch.shape[0]

        for true_basket_i, scores_i in zip(true_baskets_batch, scores_batch):

            true_basket = np.array(true_basket_i)
            model_scores = scores_i.toarray().squeeze()

            for metric in self.metrics:
                if self.save_user_metrics:
                    cumulative_metric_old = metric.cumulative_value
                    metric.add_recommendations(true_basket, model_scores)
                    cumulative_metric_new = metric.cumulative_value
                    self.user_metrics[metric.get_metric_name()].append(
                        cumulative_metric_new - cumulative_metric_old
                    )
                else:
                    metric.add_recommendations(true_basket, model_scores)

            self._n_users_evaluated += 1

        if (
            time.time() - self._start_time_print > 300
            or self._n_users_evaluated == self._num_users_to_evaluate
        ):
            elapsed_time = time.time() - self._start_time

            self._print(
                "Processed {} ({:4.1f}%). Users per second: {:.0f}".format(
                    self._n_users_evaluated,
                    100.0 * float(self._n_users_evaluated) / self._num_users_to_evaluate,
                    float(self._n_users_evaluated) / elapsed_time,
                )
            )

            sys.stdout.flush()
            sys.stderr.flush()

            self._start_time_print = time.time()

    def evaluate_recommender(self, recommender_object: IRecommender):
        self.reset_metrics()

        self._start_time = time.time()
        self._start_time_print = time.time()
        self._n_users_evaluated = 0

        users_batch_start = 0
        while users_batch_start < self._num_users_to_evaluate:
            users_batch_end = users_batch_start + self.batch_size
            users_batch_end = min(users_batch_end, self._num_users_to_evaluate)

            users_batch = self.dataset_df.iloc[users_batch_start:users_batch_end].user_id
            if isinstance(recommender_object, IRecommenderNextTs):
                next_basket_df = self.dataset_df.iloc[users_batch_start:users_batch_end].loc[:, ["user_id", "timestamp"]]
                next_basket_df = next_basket_df.rename(columns={"timestamp": "next_basket_ts"})
                scores_batch = recommender_object.predict(
                    users_batch.to_numpy(),
                    next_basket_df,
                    topk=self._max_cutoff,
                )
            else:
                scores_batch = recommender_object.predict(
                    users_batch.to_numpy(),
                    topk=self._max_cutoff,
                )

            true_baskets_batch = self.dataset_df.iloc[users_batch_start:users_batch_end].basket
            self._compute_metrics_on_recommendation_list(
                true_baskets_batch=true_baskets_batch,
                scores_batch=scores_batch,
            )

            users_batch_start = users_batch_end

        return self.create_metrics_dict()
