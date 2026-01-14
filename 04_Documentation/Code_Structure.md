# Code Structure Documentation

This document describes the architecture and organization of the Strong SME Preference Prediction codebase.

## Module Overview

```
01_Code/
├── main.py                 # Entry point and orchestration
├── config.py               # Configuration management
└── src/
    ├── __init__.py         # Package initialization
    ├── data_preprocessing.py   # Data handling classes
    ├── models.py               # ML model classes
    └── utils.py                # Utility functions
```

---

## 1. main.py - Main Execution Script

The entry point that orchestrates the entire analysis pipeline.

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_arguments()` | Parse command line arguments |
| `load_and_preprocess_data()` | Load and preprocess data |
| `engineer_features()` | Perform feature engineering |
| `train_models()` | Train all ML models |
| `analyze_features()` | Analyze feature importance |
| `export_results()` | Export results to files |

### Usage

```bash
python main.py [OPTIONS]

Options:
  --data-dir DIR        Data directory (default: 02_Data/raw)
  --results-dir DIR     Results directory (default: 03_Results)
  --trials N            Optuna trials (default: 10)
  --test-size FLOAT     Test set proportion (default: 0.2)
  --no-plots            Skip visualization
```

---

## 2. config.py - Configuration Management

Centralized configuration using Python dataclasses.

### Configuration Classes

```python
@dataclass
class DataConfig:
    """Data file paths and preprocessing parameters"""
    data_dir: str = "02_Data/raw"
    results_dir: str = "03_Results"
    government_data_file: str = "2023_strong_sme_government.xlsx"
    worknet_data_file: str = "youth_worknet_crawling.csv"
    iqr_multiplier: float = 1.5

@dataclass
class ModelConfig:
    """Model training parameters"""
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 10
    primary_metric: str = "f1_score"

@dataclass
class ProjectConfig:
    """Main configuration combining all sub-configs"""
    data: DataConfig
    model: ModelConfig
    visualization: VisualizationConfig
    export: ExportConfig
```

---

## 3. src/data_preprocessing.py

### Classes

#### DataLoader
Handles loading data from various file formats.

```python
class DataLoader:
    def __init__(self, data_dir: str)
    def load_file(self, filename: str) -> pd.DataFrame
    def load_government_data(self, filename: str) -> pd.DataFrame
    def load_worknet_data(self, filename: str) -> pd.DataFrame
    def merge_datasets(self, gov_df, worknet_df) -> pd.DataFrame
```

#### DataPreprocessor
Handles data cleaning and preprocessing with method chaining.

```python
class DataPreprocessor:
    def __init__(self, df: pd.DataFrame)
    def remove_unnecessary_columns(self) -> 'DataPreprocessor'
    def handle_missing_values(self) -> 'DataPreprocessor'
    def clean_interest_column(self) -> 'DataPreprocessor'
    def standardize_company_size(self) -> 'DataPreprocessor'
    def standardize_industry(self) -> 'DataPreprocessor'
    def remove_outliers(self, columns, iqr_multiplier) -> 'DataPreprocessor'
    def add_derived_features(self) -> 'DataPreprocessor'
    def get_dataframe(self) -> pd.DataFrame
```

#### FeatureEngineer
Creates derived features and prepares data for modeling.

```python
class FeatureEngineer:
    def __init__(self, df: pd.DataFrame)
    def create_industry_flags(self) -> 'FeatureEngineer'
    def bin_employee_count(self) -> 'FeatureEngineer'
    def create_brand_flags(self) -> 'FeatureEngineer'
    def encode_categorical_features(self) -> 'FeatureEngineer'
    def create_target_variable(self, threshold_method) -> 'FeatureEngineer'
    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]
```

---

## 4. src/models.py

### Classes

#### ModelEvaluator
Computes classification metrics.

```python
class ModelEvaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred) -> ModelMetrics
    @staticmethod
    def get_confusion_matrix(y_true, y_pred) -> np.ndarray
    @staticmethod
    def get_classification_report(y_true, y_pred) -> str
```

#### OptunaOptimizer
Handles hyperparameter optimization using Optuna.

```python
class OptunaOptimizer:
    def __init__(self, X_train, X_test, y_train, y_test, n_trials, random_state)
    def optimize(self, model_name: str) -> Tuple[Dict, float]
```

#### ModelTrainer
Main class for training and comparing all models.

```python
class ModelTrainer:
    MODEL_NAMES = [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
        "XGBClassifier"
    ]

    def __init__(self, X, y, test_size, random_state, n_trials)
    def train_all_models(self, verbose) -> pd.DataFrame
    def get_best_model(self, metric) -> Tuple[str, Any]
    def get_feature_importance(self, model_name) -> pd.DataFrame
    def get_coefficient_analysis(self) -> pd.DataFrame
    def get_combined_analysis(self, model_name) -> pd.DataFrame
```

---

## 5. src/utils.py

### Classes

#### Visualizer
Creates various visualizations.

```python
class Visualizer:
    def __init__(self, style, figsize)
    def plot_missing_values(self, df, columns, save_path) -> plt.Figure
    def plot_brand_distribution(self, df, column, save_path) -> plt.Figure
    def plot_industry_distribution(self, df, column, save_path) -> plt.Figure
    def plot_model_comparison(self, results_df, metrics, save_path) -> plt.Figure
    def plot_feature_importance(self, importance_df, top_n, save_path) -> plt.Figure
    def plot_coefficient_analysis(self, coef_df, top_n, save_path) -> plt.Figure
```

#### ResultExporter
Exports results to various formats.

```python
class ResultExporter:
    def __init__(self, output_dir)
    def export_to_csv(self, df, filename, include_timestamp) -> str
    def export_to_excel(self, dataframes, filename, include_timestamp) -> str
    def export_model_results(self, results_df, importance_df, best_model, best_params) -> str
    def export_json(self, data, filename, include_timestamp) -> str
```

---

## Design Patterns Used

1. **Method Chaining**: DataPreprocessor and FeatureEngineer use method chaining for fluent API
2. **Factory Pattern**: ModelFactory creates models based on name and parameters
3. **Strategy Pattern**: Different objective functions for each model in OptunaOptimizer
4. **Dataclass**: Configuration management using Python dataclasses

---

## Dependencies

```
Core: pandas, numpy, scipy
ML: scikit-learn, lightgbm, xgboost
Optimization: optuna
Visualization: matplotlib, seaborn
IO: openpyxl
```
