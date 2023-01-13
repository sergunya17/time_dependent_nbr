import os

import path

SRC_DIR = path.Path(os.path.abspath(os.path.dirname(__file__))).abspath()
DATA_DIR = path.Path(os.path.dirname(__file__)).joinpath("../data").abspath()
RESULTS_DIR = path.Path(os.path.dirname(__file__)).joinpath("../results").abspath()
REPORTS_DIR = RESULTS_DIR.joinpath("./reports").abspath()

SEED = 42

# ITEM_COL = "item_id"
# USER_COL = "user_id"
# TIME_COL = "timestamp"
