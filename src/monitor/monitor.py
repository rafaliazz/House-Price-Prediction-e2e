import time
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp
from data.data_preprocessor import preprocessing_pipeline

from model.trainer_tuner import train_and_tune_model
from utils.config import (
    PROCESSED_TRAIN_PATH,
    PROCESSED_TRAIN_PATH_TARGET,
    TRAIN_PATH, SAMPLE_NEW_DATA_PATH
)

TARGET_COLUMN = "SalePrice"
CHECK_INTERVAL_SECONDS = 5
DRIFT_P_THRESHOLD = 0.05


# =====================
# DRIFT DETECTION
# =====================
def detect_drift(ref: pd.DataFrame, new: pd.DataFrame):
    drifted_cols = []

    for col in ref.columns:
        stat, p = ks_2samp(ref[col], new[col])
        if p < DRIFT_P_THRESHOLD:
            drifted_cols.append((col, p))

    return drifted_cols


# =====================
# MAIN LOOP
# =====================
def main():
    print("Monitor running")

    reference = pd.read_csv(TRAIN_PATH)

    while True:
        if not SAMPLE_NEW_DATA_PATH.exists():
            time.sleep(CHECK_INTERVAL_SECONDS)
            continue

        new_data = pd.read_csv(SAMPLE_NEW_DATA_PATH)

        print(new_data.columns)

        ref_X = reference.drop(columns=[TARGET_COLUMN])
        ref_y = reference[TARGET_COLUMN]

        new_X = new_data.drop(columns=[TARGET_COLUMN])
        new_y = new_data[TARGET_COLUMN]

        feature_drift = detect_drift(ref_X, new_X)
        target_drift = detect_drift(ref_y.to_frame(), new_y.to_frame())

        if feature_drift or target_drift:
            print("Drift detected — retraining model")

            # Retrain
            X, y, _ = preprocessing_pipeline(new_data)


            train_and_tune_model(
                X=X,
                y=y,
                log_mlflow=False,
                tune_hyperparams=False, 
                model_output=Path("artifacts/model_retrained.pkl"), 
                holdout_path=Path("artifacts/holdout_retrained.pkl")
            )

            # Prevent retraining loop
            SAMPLE_NEW_DATA_PATH.rename(
                SAMPLE_NEW_DATA_PATH.with_suffix(".used")
            )

        else:
            print("No drift detected")

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
