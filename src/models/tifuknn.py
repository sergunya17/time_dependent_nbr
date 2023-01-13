import math

import optuna
import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.neighbors import NearestNeighbors

from src.models.core import IRecommender, IRecommenderNextTs
from src.dataset import NBRDatasetBase


class TIFUKNNRecommender(IRecommender):
    def __init__(
        self,
        num_nearest_neighbors: int = 300,
        within_decay_rate: float = 0.9,
        group_decay_rate: float = 0.7,
        alpha: float = 0.7,
        group_count: int = 7,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.group_count = group_count

        self._user_vectors = None
        self._nbrs = None

    def fit(self, dataset: NBRDatasetBase):
        user_basket_df = dataset.train_df.groupby("user_id", as_index=False).apply(self._calculate_basket_weight)
        user_basket_df.reset_index(drop=True, inplace=True)

        df = user_basket_df.explode("basket", ignore_index=True).rename(columns={"basket": "item_id"})
        df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("weight", "sum"))
        self._user_vectors = sps.csr_matrix(
            (df.value, (df.user_id, df.item_id)),
            shape=(dataset.num_users, dataset.num_items),
        )

        self._nbrs = NearestNeighbors(
            n_neighbors=self.num_nearest_neighbors + 1,
            algorithm="brute",
        ).fit(self._user_vectors)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_vectors.shape[1]

        user_vectors = self._user_vectors[user_ids, :]

        user_nn_indices = self._nbrs.kneighbors(user_vectors, return_distance=False)

        user_nn_vectors = []
        for nn_indices in user_nn_indices:
            nn_vectors = self._user_vectors[nn_indices[1:], :].mean(axis=0)
            user_nn_vectors.append(sps.csr_matrix(nn_vectors))
        user_nn_vectors = sps.vstack(user_nn_vectors)

        pred_matrix = self.alpha * user_vectors + (1 - self.alpha) * user_nn_vectors
        return pred_matrix

    def _calculate_basket_weight(self, df: pd.DataFrame):
        df = df.sort_values(by="timestamp", ascending=False, ignore_index=True)

        group_size = math.ceil(len(df) / self.group_count)
        df["group_num"] = df.index // group_size
        real_group_count = df["group_num"].max() + 1

        df["basket_count"] = group_size
        df.loc[df["group_num"] == len(df) // group_size, "basket_count"] = len(df) % group_size
        df["basket_num"] = df.groupby("group_num").cumcount()

        df["weight"] = (self.group_decay_rate ** df["group_num"] / real_group_count) * (
            self.within_decay_rate ** df["basket_num"] / df["basket_count"]
        )

        df.drop(columns=["group_num", "basket_count", "basket_num"], inplace=True)
        return df

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        num_nearest_neighbors = trial.suggest_categorical(
            "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
        )
        within_decay_rate = trial.suggest_categorical(
            "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        group_decay_rate = trial.suggest_categorical(
            "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        group_count = trial.suggest_int("group_count", 2, 23)
        return {
            "num_nearest_neighbors": num_nearest_neighbors,
            "within_decay_rate": within_decay_rate,
            "group_decay_rate": group_decay_rate,
            "alpha": alpha,
            "group_count": group_count,
        }

class TIFUKNNTimeDaysRecommender(TIFUKNNRecommender):
    def __init__(
        self,
        num_nearest_neighbors: int = 300,
        within_decay_rate: float = 0.2,
        group_decay_rate: float = 0.7,
        alpha: float = 0.7,
        group_size_days: int = 30,
        use_log: bool = True,
    ) -> None:
        self.num_nearest_neighbors = num_nearest_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.group_size_days = group_size_days
        self.use_log = use_log

        self._user_vectors = None
        self._nbrs = None

    def _calculate_basket_weight(self, df: pd.DataFrame):
        df = df.sort_values(by="timestamp", ascending=False, ignore_index=True)

        max_timestamp = df["timestamp"].max()
        df['global_days_diff'] = (max_timestamp - df["timestamp"]) / pd.Timedelta(days=1)

        df["group_num"] = (df["global_days_diff"] // self.group_size_days).astype("int")
        df["basket_count"] = df.groupby("group_num")["user_id"].transform("count")
        df["group_max_timestamp"] = max_timestamp - (df["group_num"] * pd.Timedelta(days=self.group_size_days))

        df["local_days_diff"] = (df["group_max_timestamp"] - df["timestamp"]) / pd.Timedelta(days=1)
        group_count = df["group_num"].nunique()

        if self.use_log:
            df["weight"] = (self.group_decay_rate ** df["group_num"] / group_count) * (
                self.within_decay_rate ** np.log(1 + df["local_days_diff"]) / df["basket_count"]
            )
        else:
            df["weight"] = (self.group_decay_rate ** df["group_num"] / group_count) * (
                self.within_decay_rate ** df["local_days_diff"] / df["basket_count"]
            )

        df = df.loc[:, ["user_id", "basket", "timestamp", "weight"]]
        return df

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        num_nearest_neighbors = trial.suggest_categorical(
            "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
        )
        within_decay_rate = trial.suggest_categorical(
            "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        group_decay_rate = trial.suggest_categorical(
            "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        group_size_days = trial.suggest_int("group_size_days", 1, 365)
        use_log = trial.suggest_categorical("use_log", [True, False])
        return {
            "num_nearest_neighbors": num_nearest_neighbors,
            "within_decay_rate": within_decay_rate,
            "group_decay_rate": group_decay_rate,
            "alpha": alpha,
            "group_size_days": group_size_days,
            "use_log": use_log,
        }


class TIFUKNNTimeDaysNextTsRecommender(IRecommenderNextTs):
    def __init__(
        self,
        num_nearest_neighbors: int = 300,
        within_decay_rate: float = 0.2,
        group_decay_rate: float = 0.7,
        alpha: float = 0.7,
        group_size_days: int = 30,
        use_log: bool = True,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.within_decay_rate = within_decay_rate
        self.group_decay_rate = group_decay_rate
        self.alpha = alpha
        self.group_size_days = group_size_days
        self.use_log = use_log

        self._train_df = None
        self._train_num_users = None
        self._train_num_items = None
        self._user_train_vectors = None
        self._nbrs = None

    def fit(self, dataset: NBRDatasetBase):
        self._train_df = dataset.train_df.copy()
        self._train_num_users = dataset.num_users
        self._train_num_items = dataset.num_items

        user_basket_df = self._train_df.groupby("user_id", as_index=False).apply(
            lambda x: self._calculate_basket_weight(x, use_next_basket_ts=False)
        )
        user_basket_df.reset_index(drop=True, inplace=True)

        df = user_basket_df.explode("basket", ignore_index=True).rename(columns={"basket": "item_id"})
        df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("weight", "sum"))
        self._user_train_vectors = sps.csr_matrix(
            (df.value, (df.user_id, df.item_id)),
            shape=(dataset.num_users, dataset.num_items),
        )

        self._nbrs = NearestNeighbors(
            n_neighbors=self.num_nearest_neighbors,
            algorithm="brute",
        ).fit(self._user_train_vectors)

        return self

    def predict(self, user_ids, user_next_basket_ts: pd.DataFrame, topk=None):

        pred_df = self._train_df.merge(user_next_basket_ts, how="inner", on="user_id")
        user_basket_df = pred_df.groupby("user_id", as_index=False).apply(
            lambda x: self._calculate_basket_weight(x, use_next_basket_ts=True)
        )
        user_basket_df.reset_index(drop=True, inplace=True)

        df = user_basket_df.explode("basket", ignore_index=True).rename(columns={"basket": "item_id"})
        df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("weight", "sum"))
        user_vectors = sps.csr_matrix(
            (df.value, (df.user_id, df.item_id)),
            shape=(self._train_num_users, self._train_num_items),
        )
        user_vectors = user_vectors[user_ids, :]

        user_nn_indices = self._nbrs.kneighbors(user_vectors, return_distance=False)

        user_nn_vectors = []
        for nn_indices in user_nn_indices:
            nn_vectors = self._user_train_vectors[nn_indices, :].mean(axis=0)
            user_nn_vectors.append(sps.csr_matrix(nn_vectors))
        user_nn_vectors = sps.vstack(user_nn_vectors)

        pred_matrix = self.alpha * user_vectors + (1 - self.alpha) * user_nn_vectors
        return pred_matrix

    def _calculate_basket_weight(self, df: pd.DataFrame, use_next_basket_ts=False):
        df = df.sort_values(by="timestamp", ascending=False, ignore_index=True)

        if use_next_basket_ts:
            max_timestamp = df.loc[0, "next_basket_ts"]
        else:
            max_timestamp = df["timestamp"].max()

        df['global_days_diff'] = (max_timestamp - df["timestamp"]) / pd.Timedelta(days=1)

        df["group_num"] = (df["global_days_diff"] // self.group_size_days).astype("int")
        df["basket_count"] = df.groupby("group_num")["user_id"].transform("count")
        df["group_max_timestamp"] = max_timestamp - (df["group_num"] * pd.Timedelta(days=self.group_size_days))

        df["local_days_diff"] = (df["group_max_timestamp"] - df["timestamp"]) / pd.Timedelta(days=1)
        group_count = df["group_num"].nunique()

        if self.use_log:
            df["weight"] = (self.group_decay_rate ** df["group_num"] / group_count) * (
                self.within_decay_rate ** np.log(1 + df["local_days_diff"]) / df["basket_count"]
            )
        else:
            df["weight"] = (self.group_decay_rate ** df["group_num"] / group_count) * (
                self.within_decay_rate ** df["local_days_diff"] / df["basket_count"]
            )

        df = df.loc[:, ["user_id", "basket", "timestamp", "weight"]]
        return df

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        num_nearest_neighbors = trial.suggest_categorical(
            "num_nearest_neighbors", [100, 300, 500, 700, 900, 1100, 1300]
        )
        within_decay_rate = trial.suggest_categorical(
            "within_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        group_decay_rate = trial.suggest_categorical(
            "group_decay_rate", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        )
        alpha = trial.suggest_categorical("alpha", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        group_size_days = trial.suggest_int("group_size_days", 1, 365)
        use_log = trial.suggest_categorical("use_log", [True, False])
        return {
            "num_nearest_neighbors": num_nearest_neighbors,
            "within_decay_rate": within_decay_rate,
            "group_decay_rate": group_decay_rate,
            "alpha": alpha,
            "group_size_days": group_size_days,
            "use_log": use_log,
        }
