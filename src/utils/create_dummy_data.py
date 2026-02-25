"""
Generate dummy labeled data based on TRAIN_PATH schema. For testing. 
"""

from pathlib import Path
import numpy as np
import pandas as pd

from utils.config import TRAIN_PATH

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
OUTPUT_PATH = Path("dataset/dummy_labeled_data.csv")
N_ROWS = 1000
RANDOM_STATE = 42

DRIFT_FACTOR = 1

TARGET_COL = "SalePrice"


# -------------------------------------------------
# DUMMY DATA GENERATOR
# -------------------------------------------------
def generate_dummy_data(
    train_path: Path,
    n_rows: int,
    drift_factor: float = 0.0,
) -> pd.DataFrame:

    np.random.seed(RANDOM_STATE)

    df_train = pd.read_csv(train_path)

    if TARGET_COL not in df_train.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found")

    dummy = {}

    for col in df_train.columns:
        col_data = df_train[col]

        # Numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            mean = col_data.mean()
            std = col_data.std()

            # Inject drift by shifting mean
            shifted_mean = mean * (1 + drift_factor)

            dummy[col] = np.random.normal(
                loc=shifted_mean,
                scale=std,
                size=n_rows
            )

        # Categorical columns
        else:
            values = col_data.dropna().unique()
            probs = col_data.value_counts(normalize=True)

            # Optional categorical drift (reweight probabilities)
            if drift_factor > 0:
                probs = probs * (1 + drift_factor)
                probs = probs / probs.sum()

            dummy[col] = np.random.choice(
                probs.index,
                size=n_rows,
                p=probs.values
            )

    dummy_df = pd.DataFrame(dummy)

    # Ensure no negative prices
    dummy_df[TARGET_COL] = dummy_df[TARGET_COL].clip(lower=0)

    return dummy_df


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":
    print("Generating dummy labeled data")

    dummy_df = generate_dummy_data(
        train_path=TRAIN_PATH,
        n_rows=N_ROWS,
        drift_factor=DRIFT_FACTOR
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dummy_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dummy data saved to {OUTPUT_PATH}")
    print(f"Rows: {len(dummy_df)}")
    print(f"Columns: {list(dummy_df.columns)}")
