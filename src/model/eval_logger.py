"""
This file is just to log the plots made in eval.ipynb 
to the same MLFlow db (doesn't work if you do it in the notebook)
"""
import mlflow
from pathlib import Path

FIGURES_DIR = Path(r"D:\personalProjects\house_price_prediction_e2e\figures")

# Dictionary of plots (filenames only)
plots = {
    "residuals": "residuals.png",
    "true_vs_pred": "truevpred.png",
    "residuals_hist": "residuals_hist.png",
    "predvrank": "pred_vs_rank.png"
}

for name, filename in plots.items():
    plt_path = FIGURES_DIR / filename  # absolute path
        
    # Log the existing file as an artifact
    mlflow.log_artifact(str(plt_path))
