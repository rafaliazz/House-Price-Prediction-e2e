"""
Train a baseline finetuned XGBoost model with optional hyperparameter search and MLflow logging.
"""

from pathlib import Path
from typing import Dict, Optional

from utils.config import PROCESSED_TRAIN_PATH, MODEL_PATH, PROCESSED_TRAIN_PATH_TARGET, HOLDOUT_PATH

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import mlflow
import mlflow.sklearn


def train_and_tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.5, 
    model_output: Path | str = MODEL_PATH,
    holdout_path: Path | str = HOLDOUT_PATH, 
    model_params: Optional[Dict] = None,
    search_params: Optional[Dict] = None,      # For hyperparameter space
    search_settings: Optional[Dict] = None,    # For RandomizedSearchCV settings like n_iter, cv
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    tune_hyperparams: bool = False,
    log_mlflow: bool = False
):

    # Optional downsampling
    if sample_frac is not None:
        X, _, y, _ = train_test_split(X, y, train_size=sample_frac, random_state=random_state)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    X_val, X_holdout, y_val, y_holdout = train_test_split(X_val, y_val, test_size=test_size, random_state=random_state)


    # Default XGBoost parameters
    default_model_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "tree_method": "hist",
    }
    if model_params:
        default_model_params.update(model_params)

    # Create base model
    model = XGBRegressor(**default_model_params)

    # MLflow logging
    if log_mlflow:
        mlflow.start_run()

    # Hyperparameter tuning
    if tune_hyperparams:
        # Default hyperparameter search space
        default_search_params = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 300, 500],
            "subsample": [0.7, 0.8, 1.0]
        }
        if search_params:
            for key in default_search_params:
                if key in search_params:
                    default_search_params[key] = search_params[key]

        # Default RandomizedSearchCV settings
        default_search_settings = {
            "n_iter": 20,
            "cv": 3,
            "scoring": "neg_mean_squared_error",
            "n_jobs": -1
        }
        if search_settings:
            default_search_settings.update(search_settings)

        # Run RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=default_search_params,
            n_iter=default_search_settings["n_iter"],
            cv=default_search_settings["cv"],
            scoring=default_search_settings["scoring"],
            n_jobs=default_search_settings["n_jobs"],
            random_state=random_state
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best hyperparameters:", search.best_params_)
        if log_mlflow:
            mlflow.log_params(search.best_params_)

    else:
        # Train model without tuning
        model.fit(X_train, y_train)
        if log_mlflow:
            mlflow.log_params(default_model_params)

    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    print(f"Validation MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    if log_mlflow:
        mlflow.log_metrics({"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})
        mlflow.sklearn.log_model(model, "model")
        mlflow.end_run()

    # Save model locally
    dump(model, model_output)
    print(f"Model saved to {model_output}")

    # Save holdout locally for evaluations 
    dump([X_holdout, y_holdout], holdout_path)
    print(f"Holdout datasets saved to {holdout_path}")

    return model

# run from terminal using python -m src.model.trainer_tuner
if __name__ == "__main__":
    print("TRAINING AND TUNING MODEL")
    X = pd.read_csv(PROCESSED_TRAIN_PATH)
    y = pd.read_csv(PROCESSED_TRAIN_PATH_TARGET)
    train_and_tune_model(X= X, y=y, log_mlflow=True, tune_hyperparams=True)

