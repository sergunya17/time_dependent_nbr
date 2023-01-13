import numpy as np
import optuna
import scipy.sparse as sps
from sklearn.preprocessing import normalize

from src.models.core import IRecommender, preprocess_matrix, calculate_user_item_matrix
from src.dataset import NBRDatasetBase


class TopPopularRecommender(IRecommender):
    def __init__(self, preprocessing: str = None) -> None:
        super().__init__()
        self.preprocessing = preprocessing

        self._user_item_matrix = None
        self._item_pop = None

    def fit(self, dataset: NBRDatasetBase):
        self._user_item_matrix = calculate_user_item_matrix(
            dataset.train_df, dataset.num_users, dataset.num_items
        )
        self._user_item_matrix = preprocess_matrix(self._user_item_matrix, self.preprocessing)

        self._item_pop = self._user_item_matrix.tocsc().sum(axis=0)
        self._item_pop = normalize(self._item_pop, norm="l1", axis=1)
        self._item_pop = np.squeeze(self._item_pop)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = len(self._item_pop)

        item_pop_topk = self._item_pop.copy()
        item_pop_topk[np.argsort(item_pop_topk)[:-topk]] = 0
        item_scores = sps.csr_matrix(np.ones([len(user_ids), 1])) * sps.csr_matrix(item_pop_topk)
        return item_scores

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        preprocessing = trial.suggest_categorical("preprocessing", [None, "binary", "log"])
        return {
            "preprocessing": preprocessing,
        }


class TopPersonalRecommender(IRecommender):
    def __init__(
        self,
        min_freq: int = 2,
        preprocessing_popular: str = None,
        preprocessing_personal: str = None,
    ) -> None:
        super().__init__()
        self.min_freq = min_freq
        self.preprocessing_popular = preprocessing_popular
        self.preprocessing_personal = preprocessing_personal

        self._user_item_matrix = None
        self._item_popular = None
        self._item_personal = None

    def fit(self, dataset: NBRDatasetBase):
        self._user_item_matrix = calculate_user_item_matrix(
            dataset.train_df, dataset.num_users, dataset.num_items
        )

        u_i_matrix_popular = preprocess_matrix(
            self._user_item_matrix.copy(), self.preprocessing_popular
        )
        self._item_popular = u_i_matrix_popular.tocsc().sum(axis=0)
        self._item_popular = normalize(self._item_popular, norm="l1", axis=1)
        self._item_popular = np.squeeze(self._item_popular)

        u_i_matrix_personal = self._user_item_matrix.copy()
        u_i_matrix_personal = u_i_matrix_personal.multiply(u_i_matrix_personal >= self.min_freq)
        u_i_matrix_personal = preprocess_matrix(u_i_matrix_personal, self.preprocessing_personal)
        self._item_personal = normalize(u_i_matrix_personal, norm="l1", axis=1)

        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = len(self._item_popular)

        item_pop_topk = self._item_popular.copy()
        item_pop_topk[np.argsort(item_pop_topk)[:-topk]] = 0
        item_scores_popular = sps.csr_matrix(np.ones([len(user_ids), 1])) * sps.csr_matrix(
            item_pop_topk
        )

        item_scores_personal = self._item_personal[user_ids].copy()
        item_scores_popular[item_scores_personal > 0] = 0
        item_scores_popular.eliminate_zeros()
        item_scores_personal[item_scores_personal > 0] += 1

        item_scores = item_scores_personal + item_scores_popular
        item_scores[np.argsort(item_scores)[:-topk]] = 0
        item_scores.eliminate_zeros()
        item_scores = normalize(item_scores, norm="l1", axis=1)
        return item_scores

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        min_freq = trial.suggest_int("min_freq", 1, 20)
        preprocessing_popular = trial.suggest_categorical(
            "preprocessing_popular", [None, "binary", "log"]
        )
        preprocessing_personal = trial.suggest_categorical(
            "preprocessing_personal", [None, "binary", "log"]
        )
        return {
            "min_freq": min_freq,
            "preprocessing_popular": preprocessing_popular,
            "preprocessing_personal": preprocessing_personal,
        }


__all__ = [TopPopularRecommender, TopPersonalRecommender]
