"""
Machine Learning Models Module for Job Seeker Analysis

This module implements various classification models with Optuna hyperparameter
optimization for predicting young job seekers' preferences for Strong SMEs.

Models implemented:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- LightGBM Classifier
- XGBoost Classifier

Author: Yoonju Cho et al.
Reference: Korean Information Systems Society Conference (2023)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import warnings

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

# Gradient Boosting Libraries
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')


@dataclass
class ModelMetrics:
    """Data class for storing model evaluation metrics."""
    precision: float
    specificity: float
    sensitivity: float  # Recall
    f1_score: float
    accuracy: float
    error_rate: float


class ModelEvaluator:
    """Class for evaluating classification model performance."""

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """
        Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            ModelMetrics object with computed metrics
        """
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        error_rate = (1 - accuracy) * 100

        return ModelMetrics(
            precision=precision,
            specificity=specificity,
            sensitivity=recall,
            f1_score=f1,
            accuracy=accuracy,
            error_rate=error_rate
        )

    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Get detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred)


class OptunaOptimizer:
    """Class for Optuna-based hyperparameter optimization."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        n_trials: int = 10,
        random_state: int = 42
    ):
        """
        Initialize OptunaOptimizer.

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            n_trials: Number of optimization trials
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_trials = n_trials
        self.random_state = random_state

    def _objective_logistic_regression(self, trial: optuna.Trial) -> float:
        """Objective function for Logistic Regression."""
        C = trial.suggest_float("C", 1e-5, 1e5, log=True)

        model = LogisticRegression(C=C, max_iter=10000, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        return f1_score(self.y_test, y_pred)

    def _objective_random_forest(self, trial: optuna.Trial) -> float:
        """Objective function for Random Forest."""
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        return f1_score(self.y_test, y_pred)

    def _objective_gradient_boosting(self, trial: optuna.Trial) -> float:
        """Objective function for Gradient Boosting."""
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        return f1_score(self.y_test, y_pred)

    def _objective_lgbm(self, trial: optuna.Trial) -> float:
        """Objective function for LightGBM."""
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
        num_leaves = trial.suggest_int("num_leaves", 2, 512)

        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=self.random_state,
            verbose=-1
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        return f1_score(self.y_test, y_pred)

    def _objective_xgboost(self, trial: optuna.Trial) -> float:
        """Objective function for XGBoost."""
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 32, log=True)

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        return f1_score(self.y_test, y_pred)

    def optimize(self, model_name: str) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters for a given model.

        Args:
            model_name: Name of the model to optimize

        Returns:
            Tuple of (best parameters, best score)
        """
        objective_map = {
            "LogisticRegression": self._objective_logistic_regression,
            "RandomForestClassifier": self._objective_random_forest,
            "GradientBoostingClassifier": self._objective_gradient_boosting,
            "LGBMClassifier": self._objective_lgbm,
            "XGBClassifier": self._objective_xgboost
        }

        if model_name not in objective_map:
            raise ValueError(f"Unknown model: {model_name}")

        # Create Optuna study with TPE sampler
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            objective_map[model_name],
            n_trials=self.n_trials,
            show_progress_bar=False
        )

        return study.best_params, study.best_value


class ModelFactory:
    """Factory class for creating models with optimized parameters."""

    @staticmethod
    def create_model(model_name: str, params: Dict[str, Any], random_state: int = 42):
        """
        Create a model with given parameters.

        Args:
            model_name: Name of the model
            params: Model parameters
            random_state: Random seed

        Returns:
            Initialized model
        """
        if model_name == "LogisticRegression":
            return LogisticRegression(
                **params,
                max_iter=10000,
                random_state=random_state
            )
        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(
                **params,
                random_state=random_state
            )
        elif model_name == "GradientBoostingClassifier":
            return GradientBoostingClassifier(
                **params,
                random_state=random_state
            )
        elif model_name == "LGBMClassifier":
            return LGBMClassifier(
                **params,
                random_state=random_state,
                verbose=-1
            )
        elif model_name == "XGBClassifier":
            return XGBClassifier(
                **params,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")


class FeatureImportanceAnalyzer:
    """Class for analyzing feature importance."""

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize FeatureImportanceAnalyzer.

        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model does not have feature importance attribute")

        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        })

        return df.sort_values('Importance', ascending=False).reset_index(drop=True)

    def get_logistic_coefficients(
        self,
        logistic_model: LogisticRegression
    ) -> pd.DataFrame:
        """
        Get coefficients from Logistic Regression for direction analysis.

        Args:
            logistic_model: Trained Logistic Regression model

        Returns:
            DataFrame with feature names and coefficients
        """
        coefficients = logistic_model.coef_[0]

        df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients
        })

        return df.sort_values('Coefficient', key=abs, ascending=False).reset_index(drop=True)


class ModelTrainer:
    """Main class for training and comparing all models."""

    MODEL_NAMES = [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
        "XGBClassifier"
    ]

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 10
    ):
        """
        Initialize ModelTrainer.

        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            random_state: Random seed
            n_trials: Number of Optuna optimization trials
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Storage for results
        self.best_params: Dict[str, Dict] = {}
        self.trained_models: Dict[str, Any] = {}
        self.metrics: Dict[str, ModelMetrics] = {}

    def train_all_models(self, verbose: bool = True) -> pd.DataFrame:
        """
        Train all models with Optuna optimization.

        Args:
            verbose: Whether to print progress

        Returns:
            DataFrame with model comparison results
        """
        optimizer = OptunaOptimizer(
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            n_trials=self.n_trials,
            random_state=self.random_state
        )

        results = []

        for model_name in self.MODEL_NAMES:
            if verbose:
                print(f"Training {model_name}...")

            # Optimize hyperparameters
            best_params, _ = optimizer.optimize(model_name)
            self.best_params[model_name] = best_params

            # Create and train model with best parameters
            model = ModelFactory.create_model(
                model_name, best_params, self.random_state
            )
            model.fit(self.X_train, self.y_train)
            self.trained_models[model_name] = model

            # Evaluate model
            y_pred = model.predict(self.X_test)
            metrics = ModelEvaluator.compute_metrics(self.y_test, y_pred)
            self.metrics[model_name] = metrics

            results.append({
                'Model': model_name,
                'Precision': metrics.precision,
                'Specificity': metrics.specificity,
                'Sensitivity (Recall)': metrics.sensitivity,
                'F1-Score': metrics.f1_score,
                'Overall Error Rate (%)': metrics.error_rate
            })

        return pd.DataFrame(results)

    def get_best_model(self, metric: str = 'f1_score') -> Tuple[str, Any]:
        """
        Get the best performing model based on a metric.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model name, trained model)
        """
        if not self.metrics:
            raise ValueError("No models trained yet. Call train_all_models() first.")

        best_model_name = max(
            self.metrics.keys(),
            key=lambda x: getattr(self.metrics[x], metric)
        )

        return best_model_name, self.trained_models[best_model_name]

    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance for a model.

        Args:
            model_name: Name of the model (default: best model)

        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name, _ = self.get_best_model()

        model = self.trained_models[model_name]
        analyzer = FeatureImportanceAnalyzer(model, list(self.X.columns))

        return analyzer.get_feature_importance()

    def get_coefficient_analysis(self) -> pd.DataFrame:
        """
        Get Logistic Regression coefficients for direction analysis.

        Returns:
            DataFrame with coefficients
        """
        if "LogisticRegression" not in self.trained_models:
            raise ValueError("Logistic Regression not trained yet.")

        model = self.trained_models["LogisticRegression"]
        analyzer = FeatureImportanceAnalyzer(model, list(self.X.columns))

        return analyzer.get_logistic_coefficients(model)

    def get_combined_analysis(self, model_name: str = "LGBMClassifier") -> pd.DataFrame:
        """
        Get combined analysis with feature importance and coefficients.

        Args:
            model_name: Name of the model for feature importance

        Returns:
            DataFrame with combined analysis
        """
        importance_df = self.get_feature_importance(model_name)
        coef_df = self.get_coefficient_analysis()

        combined = importance_df.merge(
            coef_df,
            on='Feature',
            how='left'
        )

        return combined


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 10,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Main function to train all models and get results.

    Args:
        X: Features DataFrame
        y: Target Series
        n_trials: Number of Optuna trials
        verbose: Whether to print progress

    Returns:
        Tuple of (results DataFrame, feature importance DataFrame, best model name)
    """
    trainer = ModelTrainer(X, y, n_trials=n_trials)
    results_df = trainer.train_all_models(verbose=verbose)

    best_model_name, _ = trainer.get_best_model()

    if verbose:
        print(f"\nBest Model: {best_model_name}")

    importance_df = trainer.get_combined_analysis(best_model_name)

    return results_df, importance_df, best_model_name


if __name__ == "__main__":
    print("Machine Learning Models Module for Job Seeker Analysis")
    print("=" * 50)
    print("Available Models:")
    for name in ModelTrainer.MODEL_NAMES:
        print(f"  - {name}")
    print("\nUsage:")
    print("  from src.models import train_and_evaluate")
    print("  results, importance, best_model = train_and_evaluate(X, y)")
