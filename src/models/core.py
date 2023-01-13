from abc import ABC, abstractmethod

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sps

from src.dataset import NBRDatasetBase


def check_matrix(X, format="csc", dtype=np.float32):
    """
    This function takes a matrix as input and transforms it into the specified format.
    The matrix in input can be either sparse or ndarray.
    If the matrix in input has already the desired format, it is returned as-is
    the dtype parameter is always applied and the default is np.float32
    :param X:
    :param format:
    :param dtype:
    :return:
    """

    if format == "csc" and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == "csr" and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == "coo" and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == "dok" and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == "bsr" and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == "dia" and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == "lil" and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)

    elif format == "npy":
        if sps.issparse(X):
            return X.toarray().astype(dtype)
        else:
            return np.array(X)

    elif isinstance(X, np.ndarray):
        X = sps.csr_matrix(X, dtype=dtype)
        X.eliminate_zeros()
        return check_matrix(X, format=format, dtype=dtype)
    else:
        return X.astype(dtype)


def calculate_user_item_matrix(
    user_basket_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    recency: int = 0,
    time_recency: int = 0,
    use_next_basket_ts: bool = False,
):
    assert recency * time_recency == 0

    if recency > 0:
        df = (
            user_basket_df.sort_values(by=["user_id", "timestamp"])
            .groupby("user_id")
            .tail(recency)
        )
    else:
        df = user_basket_df.copy()

    if time_recency > 0:
        df = user_basket_df.copy()
        if use_next_basket_ts:
            df["min_threshold_ts"] = df["next_basket_ts"] - pd.Timedelta(days=time_recency)
        else:
            df["min_threshold_ts"] = df.groupby("user_id", as_index=False)["timestamp"].transform("max") - pd.Timedelta(
                days=time_recency
            )

        df = df.loc[df["timestamp"] > df["min_threshold_ts"]]
        df.drop(columns=["min_threshold_ts"], inplace=True)

    df = df.explode("basket", ignore_index=True).rename(columns={"basket": "item_id"})
    df = df.groupby(["user_id", "item_id"], as_index=False).agg(value=("timestamp", "count"))
    u_i_matrix = sps.csr_matrix((df.value, (df.user_id, df.item_id)), shape=(num_users, num_items))
    return u_i_matrix


def _log_scaling_confidence(matrix):
    C = check_matrix(matrix, format="csr", dtype=np.float32)
    C.data = np.log(1.0 + C.data)
    return C


def preprocess_matrix(matrix: np.ndarray, preprocessing: str = None):
    assert preprocessing in [None, "none", "binary", "log"]
    matrix = matrix.copy()
    if preprocessing == "binary":
        matrix[matrix > 1] = 1
    elif preprocessing == "log":
        matrix = _log_scaling_confidence(matrix)
    return matrix


class IRecommender(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, dataset: NBRDatasetBase):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, user_ids, topk=None):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        raise NotImplementedError()


class IRecommenderNextTs(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, dataset: NBRDatasetBase):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, user_ids, user_next_basket_ts: pd.DataFrame, topk=None):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        raise NotImplementedError()
