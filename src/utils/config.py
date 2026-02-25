from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "dataset" / "train.csv"
TEST_PATH = ROOT / "dataset" / "test.csv"
PREPROCESSOR_PATH = ROOT / "artifacts" / "preprocessor.pkl"
PROCESSED_TRAIN_PATH = ROOT / "dataset" / "X_train_processed.csv"
PROCESSED_TEST_PATH = ROOT / "dataset" / "X_test_processed.csv"
PROCESSED_TRAIN_PATH_TARGET = ROOT / "dataset" / "y_train_processed.csv"
MODEL_PATH = ROOT / "artifacts" / "xgb_model.pkl"
HOLDOUT_PATH = ROOT / "artifacts" / "holdout.pkl"
SAMPLE_NEW_DATA_PATH = ROOT / "dataset" / "new_data.csv"
