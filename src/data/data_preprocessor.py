"""
Contains preprocessing pipeline which is used to create preprocessed csv datasets 
with optional MLflow logging
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow

from utils.config import PROCESSED_TRAIN_PATH, MODEL_PATH, PROCESSED_TRAIN_PATH_TARGET, TRAIN_PATH, TEST_PATH, PROCESSED_TEST_PATH, PREPROCESSOR_PATH

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from data.eda_summary import (
    remove_outliers,
    drop_sparse_columns,
    fill_categorical_na,
    impute_mas_vnr_area,
    impute_garage_year,
    add_time_features,
    add_amenities_features,
    ORDINAL_CATEGORIES,
    ORDINAL_FEATURES,
)

# ---------------------------
# FINAL MODEL FEATURE SET
# ---------------------------
COLS_TO_KEEP = [
    "OverallQual",
    "YearBuilt",
    "YearRemodAdd",
    "ExterQual",
    "BsmtQual",
    "TotalBsmtSF",
    "1stFlrSF",
    "GrLivArea",
    "FullBath",
    "KitchenQual",
    "TotRmsAbvGrd",
    "FireplaceQu",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "HouseAge",
]

# ---------------------------
# ALIGN ORDINAL FEATURES SAFELY
# ---------------------------
ORDINAL_FEATURES_ALIGNED = []
ORDINAL_CATEGORIES_ALIGNED = []

for feat, cats in zip(ORDINAL_FEATURES, ORDINAL_CATEGORIES):
    if feat in COLS_TO_KEEP:
        ORDINAL_FEATURES_ALIGNED.append(feat)
        ORDINAL_CATEGORIES_ALIGNED.append(cats)

assert len(ORDINAL_FEATURES_ALIGNED) == len(
    ORDINAL_CATEGORIES_ALIGNED
), "Ordinal features and categories mismatch!"


# ---------------------------
# SCALER / ENCODER PIPELINE
# ---------------------------
def build_scaler_pipeline(numerical_features):

    minmax_columns = [
        "HouseAge",
        "YearBuilt",
        "YearRemodAdd",
    ]

    standard_columns = [c for c in numerical_features if c not in minmax_columns]

    return ColumnTransformer(
        transformers=[
            (
                "std",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                standard_columns,
            ),
            (
                "mm",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", MinMaxScaler()),
                    ]
                ),
                minmax_columns,
            ),
            (
                "ord",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=ORDINAL_CATEGORIES_ALIGNED,
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ORDINAL_FEATURES_ALIGNED,
            ),
        ],
        remainder="drop",
    )


# ---------------------------
# TRAIN PREPROCESSING
# ---------------------------
def preprocessing_pipeline(df: pd.DataFrame):
    df = df.copy()

    numerical_features = df.select_dtypes(include="number").columns.tolist()
    df = remove_outliers(df)
    df = drop_sparse_columns(df)
    df = fill_categorical_na(df, numerical_features)
    df = impute_mas_vnr_area(df)
    df = impute_garage_year(df)
    df = add_time_features(df)
    df = add_amenities_features(df)

    y = df["SalePrice"]
    X = df[COLS_TO_KEEP]

    numerical_features = X.select_dtypes(include="number").columns.tolist()
    preprocessor = build_scaler_pipeline(numerical_features)
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


# ---------------------------
# INFERENCE PREPROCESSING
# ---------------------------
def inference_transform(df: pd.DataFrame, preprocessor):
    df = df.copy()
    numerical_features = df.select_dtypes(include="number").columns.tolist()
    df = drop_sparse_columns(df)
    df = fill_categorical_na(df, numerical_features)
    df = impute_mas_vnr_area(df)
    df = impute_garage_year(df)
    df = add_time_features(df)
    df = add_amenities_features(df)

    X = df[COLS_TO_KEEP]
    return preprocessor.transform(X)


# ---------------------------
# TEST AND LOG METHOD (RUN IN MAIN)
# ---------------------------
def test(log_mlflow=False):
    print("\nRunning data_preprocessor self-test...\n")
    ROOT = Path(__file__).resolve().parents[2]
    TRAIN_PATH = ROOT / "dataset" / "train.csv"
    TEST_PATH = ROOT / "dataset" / "test.csv"

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train, preprocessor = preprocessing_pipeline(train_df)
    X_test = inference_transform(test_df, preprocessor)

    print("Train:")
    print("  X shape:", X_train.shape)
    print("  y shape:", y_train.shape)

    print("\nInference:")
    print("  X shape:", X_test.shape)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch!"
    assert not np.isnan(X_train).any(), "NaNs in training data!"
    assert not np.isnan(X_test).any(), "NaNs in inference data!"

    print("\n Self-test PASSED\n")

    if log_mlflow:  # Log here (might as well)
        with mlflow.start_run():
            # Log dataset shapes
            mlflow.log_param("X_train_rows", X_train.shape[0])
            mlflow.log_param("X_train_cols", X_train.shape[1])
            mlflow.log_param("X_test_rows", X_test.shape[0])
            mlflow.log_param("X_test_cols", X_test.shape[1])

            # Log preprocessor
            mlflow.log_artifact(str(PREPROCESSOR_PATH)) 

            # Log preprocessed CSVs
            mlflow.log_artifact(str(PROCESSED_TRAIN_PATH))
            mlflow.log_artifact(str(PROCESSED_TEST_PATH))
            mlflow.log_artifact(str(PROCESSED_TRAIN_PATH_TARGET))


# ---------------------------
# run from terminal using python -m src.data.data_preprocessor
# ---------------------------
if __name__ == "__main__":
    test(log_mlflow=False)

    print("CREATING PREPROCESSED FILES")

    # Create artifacts folder if it doesn't exist
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Preprocess training data
    X_train, y_train, preprocessor = preprocessing_pipeline(train_df)
    X_test = inference_transform(test_df, preprocessor)

    # Save preprocessed datasets
    pd.DataFrame(X_train, columns=COLS_TO_KEEP).to_csv(PROCESSED_TRAIN_PATH, index=False)
    pd.DataFrame(X_test, columns=COLS_TO_KEEP).to_csv(PROCESSED_TEST_PATH, index=False)
    y_train.to_csv(PROCESSED_TRAIN_PATH_TARGET, index=False)

    # Save fitted preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    print("Preprocessing complete!")
    print(f"Saved preprocessed train data to: {PROCESSED_TRAIN_PATH}")
    print(f"Saved preprocessed test data to: {PROCESSED_TEST_PATH}")
    print(f"Saved preprocessor pipeline to: {PREPROCESSOR_PATH}")
