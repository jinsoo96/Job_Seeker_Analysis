# Analysis Workflow

This document describes the step-by-step workflow for analyzing young job seekers' preferences for Strong SMEs.

---

## Overview Flowchart

```
┌─────────────────┐
│  Data Collection │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Merging   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │
│   Engineering   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│ (with Optuna)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Evaluation    │
│   & Analysis    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Result Export   │
└─────────────────┘
```

---

## Step 1: Data Collection

### 1.1 Government Data (고용노동부)
- **Source**: Ministry of Employment and Labor
- **File**: `2023_strong_sme_government.xlsx`
- **Contents**: Company name, industry, address, representative, brand certification

### 1.2 Youth Worknet Data (청년워크넷)
- **Source**: Web crawling using Selenium
- **File**: `youth_worknet_crawling.csv`
- **Contents**: Company name, interest count, employee count, company size, location

### Data Collection Code
```python
# Selenium-based web crawling (reference)
from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("https://www.work.go.kr/jobyoung/")
# ... crawling logic
```

---

## Step 2: Data Merging

Merge datasets using company name as the key.

```python
from src.data_preprocessing import DataLoader

loader = DataLoader("02_Data/raw")
gov_df = loader.load_government_data("2023_strong_sme_government.xlsx")
worknet_df = loader.load_worknet_data("youth_worknet_crawling.csv")

# Left merge on company name
merged_df = loader.merge_datasets(gov_df, worknet_df)
print(f"Merged records: {len(merged_df)}")
```

**Result**: 27,790 merged records

---

## Step 3: Data Preprocessing

### 3.1 Remove Unnecessary Columns
```python
columns_to_drop = ['연번', '소재지', '근로자수', '업종/규모', '주소']
```

### 3.2 Handle Missing Values
```python
# Missing value distribution
missing_cols = ['소재지(중)', '브랜드명', '사업장명', '대표자명', '업종(중분류)']
# Remove 304 rows with missing values
df = df.dropna()
```

### 3.3 Clean Interest Column
```python
# Remove '건' suffix and convert to integer
df['관심기업'] = df['관심기업'].str.replace('건', '').astype(int)
```

### 3.4 Standardize Categories
```python
# Company size: Replace '알수없음' with '중소기업'
df['기업규모'] = df['기업규모'].replace('알수없음', '중소기업')

# Industry: Unify all manufacturing subcategories
df['업종(중분류)'] = df['업종(중분류)'].apply(
    lambda x: '제조업' if '제조업' in str(x) else x
)
```

### 3.5 Remove Outliers (IQR Method)
```python
def remove_outliers(df, columns, iqr_multiplier=1.5):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers(df, ['직원수', '기업개수', '브랜드신청수', '대표자수'])
```

**Result**: 23,778 records after preprocessing

---

## Step 4: Feature Engineering

### 4.1 Create Industry Flags
```python
df['제조업_if'] = df['업종(중분류)'].apply(lambda x: 1 if '제조업' in x else 0)
df['서비스업_if'] = df['업종(중분류)'].apply(lambda x: 1 if '서비스업' in x else 0)
```

### 4.2 Bin Employee Count
```python
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
labels = ['0-10', '11-20', '21-30', '31-40', '41-50',
          '51-60', '61-70', '71-80', '81-90', '91-100', '100+']
df['직원수_binned'] = pd.cut(df['직원수'], bins=bins, labels=labels)

# Create one-hot encoded columns
for label in labels:
    df[f'직원수_{label}'] = (df['직원수_binned'] == label).astype(int)
```

### 4.3 Create Brand Flags
```python
df['브랜드_이노비즈'] = df['브랜드명'].apply(lambda x: 1 if '이노비즈' in x else 0)
df['브랜드_메인비즈'] = df['브랜드명'].apply(lambda x: 1 if '메인비즈' in x else 0)
df['브랜드_기타'] = ((df['브랜드_이노비즈'] == 0) & (df['브랜드_메인비즈'] == 0)).astype(int)
```

### 4.4 One-Hot Encode Categorical Variables
```python
df = pd.get_dummies(df, columns=['기업규모', '소재지(대)', '제조업_if', '서비스업_if'], drop_first=True)
```

### 4.5 Create Target Variable
```python
# Binary classification based on median interest
median_interest = df['관심기업'].median()  # = 4
df['target'] = (df['관심기업'] > median_interest).astype(int)
```

---

## Step 5: Model Training

### 5.1 Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['관심기업', 'target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 5.2 Optuna Hyperparameter Optimization
```python
import optuna
from lightgbm import LGBMClassifier

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512)
    }
    model = LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
best_params = study.best_params
```

### 5.3 Train All Models
```python
from src.models import ModelTrainer

trainer = ModelTrainer(X, y, n_trials=10)
results_df = trainer.train_all_models()
```

---

## Step 6: Evaluation & Analysis

### 6.1 Model Comparison
```python
# Metrics: Precision, Specificity, Sensitivity, F1-Score
print(results_df)
```

### 6.2 Feature Importance (LGBM)
```python
importance_df = trainer.get_feature_importance("LGBMClassifier")
```

### 6.3 Coefficient Analysis (Logistic Regression)
```python
coef_df = trainer.get_coefficient_analysis()
```

### 6.4 Combined Analysis
```python
combined_df = trainer.get_combined_analysis("LGBMClassifier")
# Shows both importance scores and directional effects
```

---

## Step 7: Result Export

### 7.1 Export Tables
```python
from src.utils import ResultExporter

exporter = ResultExporter("03_Results/tables")
exporter.export_model_results(results_df, importance_df, best_model, best_params)
```

### 7.2 Export Visualizations
```python
from src.utils import Visualizer

viz = Visualizer()
viz.plot_model_comparison(results_df, save_path="03_Results/figures/model_comparison.png")
viz.plot_feature_importance(importance_df, save_path="03_Results/figures/feature_importance.png")
viz.plot_coefficient_analysis(combined_df, save_path="03_Results/figures/coefficients.png")
```

---

## Key Results

### Best Model: LightGBM Classifier
- Precision: 0.674
- Specificity: 0.730
- Sensitivity: 0.665
- F1-Score: 0.670

### Top 5 Important Features
1. Manufacturing Industry (제조업) - Positive effect
2. Service Industry (서비스업) - Positive effect
3. Gyeonggi Region (경기) - Negative effect
4. Seoul Region (서울) - Negative effect
5. Employee 0-10 (직원수_0-10) - Negative effect

---

## Reproducibility Notes

1. Set `random_state=42` for all random operations
2. Use the same train-test split ratio (80/20)
3. Run Optuna with at least 10 trials for stable results
4. Data should be processed in the same order as described
