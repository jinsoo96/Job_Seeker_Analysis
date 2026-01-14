# Strong SME Preference Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://pep8.org/)
[![Research](https://img.shields.io/badge/Research-KAIS%202023-orange.svg)](Papers/)

**Machine Learning-based Prediction Model for Young Job Seekers' Preference for Strong Small and Medium Enterprises**

머신러닝을 활용한 청년 구직자의 강소기업 선호 예측모형 개발 및 요인별 상대적 중요도 분석

---

## Overview

This project develops a machine learning-based classification model to predict and analyze young job seekers' preferences for "Strong SMEs" (강소기업) in South Korea. By integrating government public data with web-crawled Youth Worknet data, we identify key factors influencing job seekers' company preferences using multiple ML classifiers with Optuna hyperparameter optimization.

### Key Features

- **Data Integration**: Merges Ministry of Employment and Labor data with Youth Worknet crawled data
- **Advanced Preprocessing**: IQR-based outlier removal, feature engineering, and categorical encoding
- **Multiple ML Models**: Compares 5 classification models with automated hyperparameter tuning
- **Feature Analysis**: LGBM feature importance combined with Logistic Regression coefficients for directional interpretation
- **Comprehensive Outputs**: Publication-ready visualizations and exportable results

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Total Companies | 27,737 |
| Analysis Sample (after preprocessing) | 23,778 |
| Features | 36 |
| Train/Test Split | 80/20 |
| Best Model | LightGBM Classifier |
| Best F1-Score | 0.670 |

---

## Sample Results

### Model Performance Comparison

| Model | Precision | Specificity | Sensitivity | F1-Score |
|-------|-----------|-------------|-------------|----------|
| Logistic Regression | 0.653 | 0.695 | 0.684 | 0.668 |
| Random Forest | 0.664 | 0.715 | 0.671 | 0.667 |
| Gradient Boosting | 0.671 | 0.727 | 0.662 | 0.666 |
| **LightGBM** | **0.674** | **0.730** | 0.665 | **0.670** |
| XGBoost | 0.667 | 0.715 | 0.678 | 0.672 |

### Top Feature Importance (LGBM)

| Rank | Feature | Importance | Direction | Interpretation |
|------|---------|------------|-----------|----------------|
| 1 | Manufacturing Industry | 849 | Positive (+) | Increases preference |
| 2 | Service Industry | 586 | Positive (+) | Increases preference |
| 3 | Gyeonggi Region | 551 | Negative (-) | Decreases preference |
| 4 | Seoul Region | 515 | Negative (-) | Decreases preference |
| 5 | Employee 0-10 | 438 | Negative (-) | Decreases preference |

### Key Findings

1. **Industry Type**: Manufacturing and service industries positively influence job seeker preferences
2. **Regional Paradox**: Seoul/Gyeonggi show lower preferences (likely due to large corporation competition)
3. **Size Matters**: Companies with ≤10 employees have significantly lower appeal to young job seekers

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/jinsoo96/Job_Seeker_Analysis.git
cd Job_Seeker_Analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis

```bash
# Basic execution
python 01_Code/main.py

# With custom parameters
python 01_Code/main.py --trials 20 --test-size 0.3

# View all options
python 01_Code/main.py --help
```

---

## Repository Structure

```
Job_Seeker_Analysis/
│
├── 01_Code/                          # Source code and analysis
│   ├── main.py                       # Main execution script
│   ├── config.py                     # Configuration settings
│   ├── 01_data_analysis.ipynb        # Jupyter notebook (original analysis)
│   └── src/                          # Python modules
│       ├── __init__.py
│       ├── data_preprocessing.py     # Data loading and preprocessing
│       ├── models.py                 # ML models and training
│       └── utils.py                  # Visualization and utilities
│
├── 02_Data/                          # Data files
│   ├── raw/                          # Original data
│   │   └── DATA_DESCRIPTION.txt      # Data source description
│   └── processed/                    # Processed data (generated)
│
├── 03_Results/                       # Analysis outputs
│   ├── figures/                      # Visualization plots
│   └── tables/                       # Result tables (CSV, Excel)
│
├── 04_Documentation/                 # Additional documentation
│   ├── Code_Structure.md             # Code architecture details
│   └── Analysis_Workflow.md          # Step-by-step workflow
│
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
└── .gitignore                        # Git ignore rules
```

---

## Methodology

### 1. Data Collection

| Source | Description | Records |
|--------|-------------|---------|
| Ministry of Employment and Labor | 2023 Strong SME Selection Status | 27,737 |
| Youth Worknet (Selenium Crawling) | Company preferences and details | 27,780 |

### 2. Data Preprocessing

- **Missing Value Handling**: Removed 304 rows with null values
- **Outlier Removal**: IQR × 1.5 method for numerical columns
- **Data Standardization**: Unified industry classifications and company sizes
- **Feature Engineering**: Created industry flags, employee bins, brand indicators

### 3. Feature Engineering

```
Original Features → Derived Features
─────────────────────────────────────
직원수 (Employee Count) → 11 binned categories (0-10, 11-20, ..., 100+)
업종(중분류) (Industry) → 제조업_if (Manufacturing flag), 서비스업_if (Service flag)
브랜드명 (Brand) → 이노비즈, 메인비즈, 기타 flags
소재지(대) (Region) → One-hot encoded (16 regions)
```

### 4. Model Training

- **Optimization**: Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Validation**: Train/Test split (80/20)
- **Metric**: F1-Score for model selection

### 5. Result Interpretation

- **Feature Importance**: LGBM's built-in importance scores
- **Direction Analysis**: Logistic Regression coefficients for effect interpretation

---

## Code Example

```python
from src.data_preprocessing import DataLoader, DataPreprocessor, FeatureEngineer
from src.models import ModelTrainer

# Load data
loader = DataLoader("02_Data/raw")
gov_df = loader.load_government_data("2023_strong_sme_government.xlsx")
worknet_df = loader.load_worknet_data("youth_worknet_crawling.csv")
merged_df = loader.merge_datasets(gov_df, worknet_df)

# Preprocess
preprocessor = DataPreprocessor(merged_df)
df = (preprocessor
      .remove_unnecessary_columns()
      .handle_missing_values()
      .standardize_industry()
      .remove_outliers()
      .get_dataframe())

# Feature engineering
engineer = FeatureEngineer(df)
X, y = (engineer
        .create_industry_flags()
        .bin_employee_count()
        .create_brand_flags()
        .encode_categorical_features()
        .create_target_variable()
        .get_features_and_target())

# Train models with Optuna optimization
trainer = ModelTrainer(X, y, n_trials=10)
results = trainer.train_all_models()

# Get best model and feature importance
best_name, best_model = trainer.get_best_model()
importance_df = trainer.get_combined_analysis(best_name)
```

---

## Requirements

### Python Packages

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.1.0
lightgbm>=3.3.0
xgboost>=1.7.0
optuna>=3.0.0
matplotlib>=3.6.0
seaborn>=0.12.0
openpyxl>=3.0.0
```

### System Requirements

- RAM: 8GB+ recommended
- Storage: 500MB for data and results
- OS: Windows, macOS, or Linux

---

## Documentation

Detailed documentation available in `04_Documentation/`:

- [Code_Structure.md](04_Documentation/Code_Structure.md) - Module architecture and class descriptions
- [Analysis_Workflow.md](04_Documentation/Analysis_Workflow.md) - Step-by-step analysis guide

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Korean Association for Information Systems (KAIS)
- Ministry of Employment and Labor, South Korea
- Youth Worknet for data accessibility

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open a Pull Request


---

## Contact

For questions or collaborations:

- **Author**: Cho YoonJu, Kim Jin Soo, Bae Hwan Seok
- **GitHub**: [@justlikeazoo](https://github.com/justlikeazoo), [@jinsoo96](https://github.com/jinsoo96), [@baedol2](https://github.com/baedol2)

---
