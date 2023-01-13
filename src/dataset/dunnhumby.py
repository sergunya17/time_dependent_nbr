import os

import pandas as pd

from .base import NBRDatasetBase


class DunnhumbyDataset(NBRDatasetBase):
    def __init__(
        self,
        dataset_folder_name: str = "dunnhumby",
        min_baskets_per_user: int = 3,
        min_items_per_user: int = 0,
        min_users_per_item: int = 40,
        verbose=False,
    ):
        super().__init__(
            dataset_folder_name,
            min_baskets_per_user=min_baskets_per_user,
            min_items_per_user=min_items_per_user,
            min_users_per_item=min_users_per_item,
            verbose=verbose,
        )

    def _preprocess(self) -> pd.DataFrame:
        transaction_data_path = os.path.join(self.raw_path, "transaction_data.csv")
        df = pd.read_csv(transaction_data_path)

        df = df.rename(
            columns={"household_key": "user_id", "BASKET_ID": "basket_id", "PRODUCT_ID": "item_id"}
        )
        df["timestamp"] = pd.to_datetime(
            df.DAY * 1440 + df.TRANS_TIME // 100 * 60 + df.TRANS_TIME % 100,
            unit="m",
        )

        df = df[["user_id", "basket_id", "item_id", "timestamp"]].drop_duplicates()
        return df
