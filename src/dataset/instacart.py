import os

import pandas as pd

from .base import NBRDatasetBase


class InstacartDataset(NBRDatasetBase):
    def __init__(
        self,
        dataset_folder_name: str = "instacart",
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
            max_users_num=20000,
            verbose=verbose,
        )

    def _preprocess(self) -> pd.DataFrame:

        orders = pd.read_csv(os.path.join(self.raw_path, "orders.csv"))
        orders = orders[orders["eval_set"] != "test"].fillna(0)
        orders["timestamp"] = orders.groupby("user_id")["days_since_prior_order"].cumsum()
        orders["timestamp"] = pd.to_datetime(
            orders.timestamp * 86400 + orders.order_hour_of_day * 3600 + orders.order_number,
            unit="s",
        )
        orders = orders[["order_id", "user_id", "timestamp"]]

        order_products_prior = pd.read_csv(os.path.join(self.raw_path, "order_products__prior.csv"))
        order_products_train = pd.read_csv(os.path.join(self.raw_path, "order_products__train.csv"))
        order_products = pd.concat([order_products_prior, order_products_train])
        order_products = order_products[["order_id", "product_id"]]

        interactions = orders.set_index("order_id").join(
            order_products.set_index("order_id"), how="inner"
        )
        interactions = interactions.reset_index().rename(
            columns={"order_id": "basket_id", "product_id": "item_id"}
        )

        df = interactions[["user_id", "basket_id", "item_id", "timestamp"]].drop_duplicates()
        return df
