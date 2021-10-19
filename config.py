import numpy as np

# all column names should be lowercase
CAT_COLS = {
    "id_universe": np.int64,
    "id_category": np.int64,
    "id_brand": np.int64,
    "id_subcat": np.int64,
    "id_sub_subcat": np.int64,
    "id_model": np.int64,
    "id_color": np.int64,
    "id_material": np.int64,
    "id_condition": np.int64,
    "vintage": np.int64,
    "id_bracelet": np.int64,
    "id_box": np.int64,
    "id_mechanism": np.int64,
    "id_size_type": np.int64,
    # "order_currency": str,
    "year": np.int64,
    "month": np.int64,
    "day_of_month": np.int64,
    "day_of_week": np.int64,
}
NUM_COLS = {
    "hours_online": np.float32,
}
TARGET_COL = {
    "price": np.float32,
}
# variable that needs to be setup
EMBEDDING_SIZES = [(10, 100) for k, v in CAT_COLS.items()]
