# File: evaluation/__init__.py

from .eval_meta import evaluate_model
from .eval_retrieval import evaluate_retrieval_augmented
from .eval_baselines import evaluate_sklearn_baselines
# 🔻 --- NEW --- 🔻
from .eval_pca_knn import evaluate_pca_knn
# 🔺 --- END NEW --- 🔺