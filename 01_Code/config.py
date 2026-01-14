"""
Configuration Settings for Job Seeker Analysis

This module contains all configuration parameters for the project.

Author: Yoonju Cho et al.
Reference: Korean Information Systems Society Conference (2023)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # File paths
    data_dir: str = "../02_Data/raw"
    results_dir: str = "../03_Results"

    # Data file names
    government_data_file: str = "2023_strong_sme_government.xlsx"
    worknet_data_file: str = "youth_worknet_crawling.csv"

    # Columns to drop during preprocessing
    columns_to_drop: List[str] = field(default_factory=lambda: [
        '연번', '소재지', '근로자수', '업종/규모', '주소'
    ])

    # Columns for outlier removal
    outlier_columns: List[str] = field(default_factory=lambda: [
        '직원수', '기업개수', '브랜드신청수', '대표자수'
    ])

    # IQR multiplier for outlier detection
    iqr_multiplier: float = 1.5


@dataclass
class ModelConfig:
    """Configuration for machine learning models."""

    # Train-test split
    test_size: float = 0.2
    random_state: int = 42

    # Optuna optimization
    n_trials: int = 10

    # Models to train
    model_names: List[str] = field(default_factory=lambda: [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "LGBMClassifier",
        "XGBClassifier"
    ])

    # Primary metric for model selection
    primary_metric: str = "f1_score"


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""

    # Figure settings
    figsize: tuple = (10, 6)
    style: str = "whitegrid"
    dpi: int = 300

    # Feature importance plot
    top_n_features: int = 15


@dataclass
class ExportConfig:
    """Configuration for result export."""

    # Export formats
    export_csv: bool = True
    export_excel: bool = True
    export_json: bool = True
    export_plots: bool = True

    # Include timestamp in filenames
    include_timestamp: bool = True


@dataclass
class ProjectConfig:
    """Main project configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Project metadata
    project_name: str = "Job Seeker Analysis"
    version: str = "1.0.0"
    authors: List[str] = field(default_factory=lambda: [
        "Yoonju Cho",
        "Jinsu Kim",
        "Hwanseok Bae",
        "Sanghyeok Yoon",
        "Seongbyung Yang"
    ])

    def __post_init__(self):
        """Create necessary directories after initialization."""
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.data.results_dir, exist_ok=True)


# Default configuration instance
config = ProjectConfig()


# Employee count bins configuration
EMPLOYEE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
EMPLOYEE_LABELS = [
    '0-10', '11-20', '21-30', '31-40', '41-50',
    '51-60', '61-70', '71-80', '81-90', '91-100', '100+'
]


# Industry categories
INDUSTRY_CATEGORIES = {
    'manufacturing': '제조업',
    'service': '서비스업'
}


# Brand categories
BRAND_CATEGORIES = {
    'innobiz': '이노비즈',
    'mainbiz': '메인비즈',
    'other': '기타'
}


# Korean to English column mapping for output
COLUMN_NAME_MAPPING = {
    '기업명': 'company_name',
    '관심기업': 'interest_count',
    '직원수': 'employee_count',
    '기업규모': 'company_size',
    '소재지(대)': 'region_major',
    '소재지(중)': 'region_minor',
    '브랜드명': 'brand_name',
    '대표자명': 'representative',
    '업종(중분류)': 'industry_category'
}


if __name__ == "__main__":
    # Print configuration summary
    print("Job Seeker Analysis - Configuration")
    print("=" * 50)
    print(f"Project: {config.project_name}")
    print(f"Version: {config.version}")
    print(f"\nData Settings:")
    print(f"  - Data directory: {config.data.data_dir}")
    print(f"  - Results directory: {config.data.results_dir}")
    print(f"\nModel Settings:")
    print(f"  - Test size: {config.model.test_size}")
    print(f"  - Random state: {config.model.random_state}")
    print(f"  - Optuna trials: {config.model.n_trials}")
    print(f"  - Primary metric: {config.model.primary_metric}")
