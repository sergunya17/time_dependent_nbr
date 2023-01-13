from abc import abstractmethod, ABC
from ast import literal_eval
import math
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from src.settings import DATA_DIR


class NBRDatasetBase(ABC):
    def __init__(
        self,
        dataset_folder_name: str,
        min_baskets_per_user: int = 3,
        min_items_per_user: int = 0,
        min_users_per_item: int = 0,
        max_users_num: int = 0,
        max_items_num: int = 0,
        verbose=False,
    ):
        self.min_baskets_per_user = min_baskets_per_user
        self.min_items_per_user = min_items_per_user
        self.min_users_per_item = min_users_per_item
        self.max_users_num = max_users_num
        self.max_items_num = max_items_num
        self.verbose = verbose

        self.dataset_dir = os.path.join(DATA_DIR, dataset_folder_name)

        self.raw_path = os.path.join(self.dataset_dir, "raw")
        if not os.path.exists(self.raw_path) or len(os.listdir(self.raw_path)) == 0:
            raise FileNotFoundError(
                f"Dataset not found in folder: {self.raw_path}. Please download it."
            )

        self.processed_path = os.path.join(self.dataset_dir, "processed")
        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)

        self.split_path = os.path.join(self.dataset_dir, "split")
        if not os.path.exists(self.split_path):
            os.mkdir(self.split_path)

        self.train_df = None
        self.val_df = None
        self.test_df = None

        self._index2user = None
        self._user2index = None
        self._index2item = None
        self._item2index = None

    @property
    def num_users(self):
        if self._user2index is not None:
            return len(self._user2index)
        else:
            return 0

    @property
    def num_items(self):
        if self._item2index is not None:
            return len(self._item2index)
        else:
            return 0

    @abstractmethod
    def _preprocess(self) -> pd.DataFrame:
        raise NotImplementedError()

    def preprocess(self):
        self._print("Start preprocessing...")

        interactions = self._preprocess()
        interactions.to_csv(os.path.join(self.processed_path, "interactions.csv"), index=False)

        self._print("Interactions preprocessed and saved")

        return self

    def make_leave_one_basket_split(
        self, test_users_rate=0.5, random_basket=False, random_state=42
    ):
        self._print("Start leave-one-basket splitting...")

        if self.min_baskets_per_user < 2:
            raise ValueError("min_baskets_per_user must be at least 2 for this split!")

        data = self._load_and_filter_interactions()
        self._print("Splitting...")
        data = self._leave_one_basket_split(
            data,
            test_users_rate=test_users_rate,
            random_basket=random_basket,
            random_state=random_state,
        )
        self._print("Splitting is done")
        self._save_split(data)

        return self

    def make_leave_two_baskets_split(self, random_baskets=False, random_state=42):
        self._print("Start leave-two-baskets splitting...")

        if self.min_baskets_per_user < 3:
            raise ValueError("min_baskets_per_user must be at least 3 for this split!")

        data = self._load_and_filter_interactions()
        self._print("Splitting...")
        data = self._leave_two_baskets_split(
            data, random_baskets=random_baskets, random_state=random_state
        )
        self._print("Splitting is done")
        self._save_split(data)

        return self

    def load_split(self) -> tuple:
        self._print("Loading split data...")

        user_mapping_path = os.path.join(self.split_path, "user_mapping.csv")
        item_mapping_path = os.path.join(self.split_path, "item_mapping.csv")
        train_path = os.path.join(self.split_path, "train.csv")
        val_path = os.path.join(self.split_path, "validate.csv")
        test_path = os.path.join(self.split_path, "test.csv")

        if not (
            os.path.exists(user_mapping_path)
            and os.path.exists(item_mapping_path)
            and os.path.exists(train_path)
            and os.path.exists(val_path)
            and os.path.exists(test_path)
        ):
            raise RuntimeError("Make a split before loading it!")

        user_mapping = pd.read_csv(os.path.join(self.split_path, "user_mapping.csv"))
        self._index2user = dict(zip(user_mapping.index, user_mapping.user_id))
        self._user2index = dict(zip(user_mapping.user_id, user_mapping.index))

        self._print("User mapping loaded")

        item_mapping = pd.read_csv(os.path.join(self.split_path, "item_mapping.csv"))
        self._index2item = dict(zip(item_mapping.index, item_mapping.item_id))
        self._item2index = dict(zip(item_mapping.item_id, item_mapping.index))

        self._print("Item mapping loaded")

        self.train_df = pd.read_csv(
            train_path,
            converters={"basket": literal_eval},
            parse_dates=["timestamp"],
            infer_datetime_format=True,
        )
        self.val_df = pd.read_csv(
            val_path,
            converters={"basket": literal_eval},
            parse_dates=["timestamp"],
            infer_datetime_format=True,
        )
        self.test_df = pd.read_csv(
            test_path,
            converters={"basket": literal_eval},
            parse_dates=["timestamp"],
            infer_datetime_format=True,
        )

        self._print("Interactions loaded")

        return self.train_df, self.val_df, self.test_df

    def _print(self, string: str):
        if self.verbose:
            print(f"{type(self).__name__}: {string}")

    @staticmethod
    def _leave_one_basket_split(
        data: pd.DataFrame, test_users_rate=0.5, random_basket=False, random_state=None
    ):
        data["split_flag"] = "train"
        if random_basket:
            data = shuffle(data, random_state=random_state)
        else:
            data.sort_values(by=["timestamp"], inplace=True)

        last_baskets = data.groupby("user_id").last()["basket_id"]
        data.loc[data["basket_id"].isin(last_baskets), "split_flag"] = "validate"

        users = data["user_id"].unique()
        test_size = math.ceil(len(users) * test_users_rate)
        test_users = shuffle(users, random_state=random_state)[:test_size]
        data.loc[
            (data["user_id"].isin(test_users)) & (data["split_flag"] == "validate"), "split_flag"
        ] = "test"

        return data

    @staticmethod
    def _leave_two_baskets_split(data: pd.DataFrame, random_baskets=False, random_state=None):
        data["split_flag"] = "train"
        if random_baskets:
            data = shuffle(data, random_state=random_state)
        else:
            data.sort_values(by=["timestamp"], inplace=True)

        last_baskets = data.groupby("user_id").last()["basket_id"]
        data.loc[data["basket_id"].isin(last_baskets), "split_flag"] = "test"

        penult_baskets = (
            data.loc[~data["basket_id"].isin(last_baskets)].groupby("user_id").last()["basket_id"]
        )
        data.loc[data["basket_id"].isin(penult_baskets), "split_flag"] = "validate"

        return data

    def _save_split(self, data: pd.DataFrame):
        # save mappings
        users = data["user_id"].unique()
        user_mapping = pd.DataFrame({"index": np.arange(len(users)), "user_id": users})
        user2index = dict(zip(user_mapping.user_id, user_mapping.index))

        items = data["item_id"].unique()
        item_mapping = pd.DataFrame({"index": np.arange(len(items)), "item_id": items})
        item2index = dict(zip(item_mapping.item_id, item_mapping.index))

        data["user_id"] = data["user_id"].map(user2index)
        data["item_id"] = data["item_id"].map(item2index)
        user_mapping.to_csv(os.path.join(self.split_path, "user_mapping.csv"), index=False)
        item_mapping.to_csv(os.path.join(self.split_path, "item_mapping.csv"), index=False)

        self._print("Mappings saved")

        # save interactions
        data = data.groupby(["user_id", "split_flag", "timestamp", "basket_id"]).item_id.apply(
            pd.Series.tolist
        )
        data = data.rename("basket").reset_index()

        train = data.loc[data["split_flag"] == "train", ["user_id", "basket", "timestamp"]]
        val = data.loc[data["split_flag"] == "validate", ["user_id", "basket", "timestamp"]]
        test = data.loc[data["split_flag"] == "test", ["user_id", "basket", "timestamp"]]

        train.to_csv(os.path.join(self.split_path, "train.csv"), index=False)
        val.to_csv(os.path.join(self.split_path, "validate.csv"), index=False)
        test.to_csv(os.path.join(self.split_path, "test.csv"), index=False)

        self._print("Split interactions saved")

    def _load_and_filter_interactions(self):
        processed_file_path = os.path.join(self.processed_path, "interactions.csv")

        if not os.path.exists(processed_file_path):
            self._print("Needs to be preprocessed first")
            self.preprocess()

        data = pd.read_csv(processed_file_path)
        data = self._filter_data(data)

        return data

    def _filter_data(self, data: pd.DataFrame):
        self._print("Filtering data...")

        if self.max_users_num > 0:
            data = self._filter_by_unique_count(data, "user_id", self.max_users_num)

        if self.max_items_num > 0:
            data = self._filter_by_unique_count(data, "item_id", self.max_items_num)

        num_interactions = len(data.index)
        iteration = 0
        while True:
            # Filter out users that have less than min_baskets_per_user baskets
            if self.min_baskets_per_user > 0:
                data = self._filter_by_count(
                    data, "user_id", "basket_id", self.min_baskets_per_user
                )

            # Filter out users that have less than min_items_per_user items
            if self.min_items_per_user > 0:
                data = self._filter_by_count(data, "user_id", "item_id", self.min_items_per_user)

            # Filter out items that have less than min_users_per_item users
            if self.min_users_per_item > 0:
                data = self._filter_by_count(data, "item_id", "user_id", self.min_users_per_item)

            iteration += 1
            self._print(f"Iteration #{iteration}")

            new_num_interactions = len(data.index)
            if num_interactions != new_num_interactions:
                num_interactions = new_num_interactions
            else:
                break  # no change

        if len(data.index) < 1:
            raise RuntimeError(
                "This dataset contains no interaction after filtering. "
                "Please change filter setup of this split!"
            )

        return data

    @staticmethod
    def _filter_by_count(data, group_column, filter_column, min_count):
        object_count = (
            data.groupby([group_column])[filter_column].nunique().rename("count").reset_index()
        )
        filtered_data = data.loc[
            data[group_column].isin(object_count[object_count["count"] >= min_count][group_column])
        ]
        return filtered_data

    @staticmethod
    def _filter_by_unique_count(data, filter_column, max_count):
        unique = data[filter_column].unique()
        if len(unique) > max_count:
            unique = unique[:max_count]
        filtered_data = data.loc[data[filter_column].isin(unique)]
        return filtered_data
