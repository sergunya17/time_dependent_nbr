import optuna
import similaripy as sim

from src.models.core import IRecommender, preprocess_matrix, calculate_user_item_matrix
from src.dataset import NBRDatasetBase


class UPCFRecommender(IRecommender):
    def __init__(
        self,
        recency: int = 0,
        q: int = 5,
        alpha: float = 0.25,
        topk_neighbors: int = None,
        preprocessing: str = None,
    ) -> None:
        super().__init__()
        self.recency = recency
        self.q = q
        self.alpha = alpha
        self.topk_neighbors = topk_neighbors
        self.preprocessing = preprocessing

        self._user_item_matrix = None
        self._user_item_matrix_implicit = None
        self._similarity_matrix = None

    def fit(self, dataset: NBRDatasetBase):
        self._user_item_matrix = calculate_user_item_matrix(
            dataset.train_df, dataset.num_users, dataset.num_items, recency=self.recency
        )
        self._user_item_matrix_implicit = preprocess_matrix(self._user_item_matrix, "binary")
        self._user_item_matrix = preprocess_matrix(self._user_item_matrix, self.preprocessing)

        if self.topk_neighbors is None:
            self.topk_neighbors = self._user_item_matrix.shape[0]

        self._similarity_matrix = sim.asymmetric_cosine(
            self._user_item_matrix_implicit, alpha=self.alpha, k=self.topk_neighbors, verbose=False
        )
        return self

    def predict(self, user_ids, topk=None):
        if topk is None:
            topk = self._user_item_matrix.shape[1]

        pred_matrix = sim.dot_product(
            self._similarity_matrix.power(self.q),
            self._user_item_matrix,
            k=topk,
            verbose=False,
        )
        pred_matrix = pred_matrix.tocsr()

        return pred_matrix[user_ids]

    @classmethod
    def sample_params(cls, trial: optuna.Trial) -> dict:
        recency = trial.suggest_categorical("recency", [1, 5, 25, 100])
        q = trial.suggest_categorical("q", [1, 5, 10, 50, 100, 1000])
        alpha = trial.suggest_categorical("alpha", [0, 0.25, 0.5, 0.75, 1])
        topk_neighbors = trial.suggest_categorical("topk_neighbors", [None, 10, 100, 300, 600, 900])
        preprocessing = trial.suggest_categorical("preprocessing", [None, "binary"])
        return {
            "recency": recency,
            "q": q,
            "alpha": alpha,
            "topk_neighbors": topk_neighbors,
            "preprocessing": preprocessing,
        }
