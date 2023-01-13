import os

import pandas as pd

from .base import NBRDatasetBase


class TafengDataset(NBRDatasetBase):
    def __init__(
        self,
        dataset_folder_name: str = "tafeng",
        min_baskets_per_user: int = 3,
        min_items_per_user: int = 0,
        min_users_per_item: int = 5,
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
        transaction_data_path = os.path.join(self.raw_path, "ta_feng_all_months_merged.csv")
        df = pd.read_csv(transaction_data_path)

        df["timestamp"] = pd.to_datetime(df["TRANSACTION_DT"])
        df.rename(columns={"CUSTOMER_ID": "user_id", "PRODUCT_ID": "item_id"}, inplace=True)
        df["basket_id"] = df.groupby(["user_id", "timestamp"]).ngroup()

        df = df[["user_id", "basket_id", "item_id", "timestamp"]].drop_duplicates()
        return df
