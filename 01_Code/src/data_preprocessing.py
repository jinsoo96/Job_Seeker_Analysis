"""
Data Preprocessing Module for Job Seeker Analysis

This module handles data loading, cleaning, and preprocessing for the
machine learning-based prediction model analyzing young job seekers'
preferences for Strong SMEs (강소기업).

Author: Yoonju Cho et al.
Reference: Korean Information Systems Society Conference (2023)
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Dict, Optional


class DataLoader:
    """Class for loading and merging datasets from different sources."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize DataLoader.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir

    def load_file(self, filename: str) -> pd.DataFrame:
        """
        Load a single file (CSV or Excel).

        Args:
            filename: Name of the file to load

        Returns:
            Loaded DataFrame
        """
        filepath = os.path.join(self.data_dir, filename)
        _, ext = os.path.splitext(filename)

        if ext == '.csv':
            try:
                return pd.read_csv(filepath, encoding='utf-8-sig')
            except UnicodeDecodeError:
                return pd.read_csv(filepath, encoding='CP949')
        elif ext == '.xlsx':
            return pd.read_excel(filepath, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def load_government_data(self, filename: str) -> pd.DataFrame:
        """
        Load government (고용노동부) Strong SME data.

        Args:
            filename: Name of the government data file

        Returns:
            Processed government data DataFrame
        """
        df = self.load_file(filename)

        # Standardize column names
        if '연번' in df.columns:
            df['연번'] = df['연번'].str.slice(0, 4)

        # Rename columns
        rename_map = {
            '업종명\n(대분류)': '업종(대분류)',
            '업종명\n(중분류)': '업종(중분류)'
        }
        df = df.rename(columns=rename_map)

        # Drop unnecessary columns
        if '업종(대분류)' in df.columns:
            df = df.drop(columns=['업종(대분류)'])

        return df

    def load_worknet_data(self, filename: str) -> pd.DataFrame:
        """
        Load Youth Worknet (청년워크넷) crawled data.

        Args:
            filename: Name of the worknet data file

        Returns:
            Loaded worknet DataFrame
        """
        return self.load_file(filename)

    def merge_datasets(
        self,
        gov_df: pd.DataFrame,
        worknet_df: pd.DataFrame,
        left_on: str = '기업명',
        right_on: str = '사업장명'
    ) -> pd.DataFrame:
        """
        Merge government and worknet datasets.

        Args:
            gov_df: Government data DataFrame
            worknet_df: Worknet data DataFrame
            left_on: Column name for left join key
            right_on: Column name for right join key

        Returns:
            Merged DataFrame
        """
        merged_df = worknet_df.merge(
            gov_df,
            left_on=left_on,
            right_on=right_on,
            how='left'
        )
        return merged_df


class DataPreprocessor:
    """Class for preprocessing merged dataset."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataPreprocessor.

        Args:
            df: DataFrame to preprocess
        """
        self.df = df.copy()

    def remove_unnecessary_columns(
        self,
        columns_to_drop: List[str] = None
    ) -> 'DataPreprocessor':
        """
        Remove unnecessary columns from the dataset.

        Args:
            columns_to_drop: List of column names to drop

        Returns:
            Self for method chaining
        """
        if columns_to_drop is None:
            columns_to_drop = ['연번', '소재지', '근로자수', '업종/규모', '주소']

        existing_columns = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_columns)
        return self

    def handle_missing_values(self) -> 'DataPreprocessor':
        """
        Handle missing values by removing rows with any null values.

        Returns:
            Self for method chaining
        """
        self.df = self.df.dropna()
        return self

    def clean_interest_column(self) -> 'DataPreprocessor':
        """
        Clean the '관심기업' column by removing '건' suffix and converting to int.

        Returns:
            Self for method chaining
        """
        if '관심기업' in self.df.columns:
            if self.df['관심기업'].dtype == object:
                self.df['관심기업'] = self.df['관심기업'].str.replace('건', '').astype(int)
        return self

    def remove_duplicate_columns(self) -> 'DataPreprocessor':
        """
        Remove duplicate or redundant columns.

        Returns:
            Self for method chaining
        """
        if '관심기업건수' in self.df.columns:
            self.df = self.df.drop(columns=['관심기업건수'])
        if '선정분야' in self.df.columns:
            self.df = self.df.drop(columns=['선정분야'])
        if '사업장명' in self.df.columns:
            self.df = self.df.drop(columns=['사업장명'])
        return self

    def standardize_company_size(self) -> 'DataPreprocessor':
        """
        Standardize company size by replacing '알수없음' with '중소기업'.

        Returns:
            Self for method chaining
        """
        if '기업규모' in self.df.columns:
            self.df['기업규모'] = self.df['기업규모'].replace('알수없음', '중소기업')
        return self

    def standardize_industry(self) -> 'DataPreprocessor':
        """
        Standardize industry classification by grouping all manufacturing industries.

        Returns:
            Self for method chaining
        """
        if '업종(중분류)' in self.df.columns:
            self.df['업종(중분류)'] = self.df['업종(중분류)'].apply(
                lambda x: '제조업' if '제조업' in str(x) else x
            )
        return self

    def remove_outliers(
        self,
        columns: List[str] = None,
        iqr_multiplier: float = 1.5
    ) -> 'DataPreprocessor':
        """
        Remove outliers using IQR method.

        Args:
            columns: List of columns to check for outliers
            iqr_multiplier: Multiplier for IQR range (default: 1.5)

        Returns:
            Self for method chaining
        """
        if columns is None:
            columns = ['직원수', '기업개수', '브랜드신청수', '대표자수']

        existing_columns = [col for col in columns if col in self.df.columns]

        for col in existing_columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]

        return self

    def add_derived_features(self) -> 'DataPreprocessor':
        """
        Add derived features like company count, representative count, and brand count.

        Returns:
            Self for method chaining
        """
        # Company count per company name
        if '기업명' in self.df.columns:
            self.df['기업개수'] = self.df.groupby('기업명')['기업명'].transform('count')

        # Representative count
        if '기업명' in self.df.columns and '대표자명' in self.df.columns:
            grouped = self.df.groupby(['기업명', '대표자명']).size().reset_index(name='counts')
            rep_count = grouped.groupby('기업명')['대표자명'].count().reset_index(name='대표자수')
            self.df = pd.merge(self.df, rep_count, on='기업명', how='left')

        # Brand application count
        if '기업명' in self.df.columns and '브랜드명' in self.df.columns:
            brand_counts = self.df.groupby('기업명')['브랜드명'].nunique()
            self.df = self.df.assign(브랜드신청수=self.df['기업명'].map(brand_counts))

        return self

    def remove_location_duplicates(self) -> 'DataPreprocessor':
        """
        Remove duplicates based on company name and location.

        Returns:
            Self for method chaining
        """
        if all(col in self.df.columns for col in ['기업명', '소재지(대)', '소재지(중)']):
            self.df = self.df.drop_duplicates(subset=['기업명', '소재지(대)', '소재지(중)'])
        return self

    def reset_index(self) -> 'DataPreprocessor':
        """
        Reset DataFrame index.

        Returns:
            Self for method chaining
        """
        self.df = self.df.reset_index(drop=True)
        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the preprocessed DataFrame.

        Returns:
            Preprocessed DataFrame
        """
        return self.df


class FeatureEngineer:
    """Class for feature engineering operations."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer.

        Args:
            df: DataFrame for feature engineering
        """
        self.df = df.copy()

    def create_industry_flags(self) -> 'FeatureEngineer':
        """
        Create binary flags for manufacturing and service industries.

        Returns:
            Self for method chaining
        """
        if '업종(중분류)' in self.df.columns:
            self.df['제조업_if'] = self.df['업종(중분류)'].apply(
                lambda x: 1 if '제조업' in str(x) else 0
            )
            self.df['서비스업_if'] = self.df['업종(중분류)'].apply(
                lambda x: 1 if '서비스업' in str(x) else 0
            )
        return self

    def bin_employee_count(self) -> 'FeatureEngineer':
        """
        Bin employee count into categories.

        Returns:
            Self for method chaining
        """
        if '직원수' not in self.df.columns:
            return self

        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
        labels = ['0-10', '11-20', '21-30', '31-40', '41-50',
                  '51-60', '61-70', '71-80', '81-90', '91-100', '100+']

        self.df['직원수_binned'] = pd.cut(
            self.df['직원수'],
            bins=bins,
            labels=labels,
            right=False
        )

        # Create one-hot encoded columns for binned values
        for label in labels:
            self.df[f'직원수_{label}'] = np.where(
                self.df['직원수_binned'] == label, 1, 0
            )

        return self

    def create_brand_flags(self) -> 'FeatureEngineer':
        """
        Create binary flags for brand types (이노비즈, 메인비즈, 기타).

        Returns:
            Self for method chaining
        """
        if '브랜드명' in self.df.columns:
            self.df['브랜드_이노비즈'] = self.df['브랜드명'].apply(
                lambda x: 1 if '이노비즈' in str(x) else 0
            )
            self.df['브랜드_메인비즈'] = self.df['브랜드명'].apply(
                lambda x: 1 if '메인비즈' in str(x) else 0
            )
            self.df['브랜드_기타'] = self.df.apply(
                lambda row: 1 if row['브랜드_이노비즈'] == 0 and row['브랜드_메인비즈'] == 0 else 0,
                axis=1
            )
        return self

    def encode_categorical_features(
        self,
        columns_to_encode: List[str] = None,
        columns_to_drop: List[str] = None
    ) -> 'FeatureEngineer':
        """
        One-hot encode categorical features.

        Args:
            columns_to_encode: List of columns to encode
            columns_to_drop: List of columns to drop before encoding

        Returns:
            Self for method chaining
        """
        if columns_to_drop is None:
            columns_to_drop = ['소재지(중)', '브랜드명', '브랜드신청수', '업종(중분류)']

        existing_columns = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_columns)

        if columns_to_encode is None:
            columns_to_encode = ['기업규모', '소재지(대)', '제조업_if', '서비스업_if']

        existing_columns = [col for col in columns_to_encode if col in self.df.columns]
        self.df = pd.get_dummies(self.df, columns=existing_columns, drop_first=True)

        return self

    def create_target_variable(
        self,
        threshold_method: str = 'median'
    ) -> 'FeatureEngineer':
        """
        Create binary target variable for classification.

        Args:
            threshold_method: Method to determine threshold ('median' or 'mean')

        Returns:
            Self for method chaining
        """
        if '관심기업' not in self.df.columns:
            return self

        if threshold_method == 'median':
            threshold = self.df['관심기업'].median()
        else:
            threshold = self.df['관심기업'].mean()

        self.df['target'] = self.df['관심기업'].apply(
            lambda x: 1 if x > threshold else 0
        )

        return self

    def drop_non_feature_columns(
        self,
        columns_to_drop: List[str] = None
    ) -> 'FeatureEngineer':
        """
        Drop non-feature columns for modeling.

        Args:
            columns_to_drop: List of column names to drop

        Returns:
            Self for method chaining
        """
        if columns_to_drop is None:
            columns_to_drop = ['기업명', '대표자명', '직원수', '직원수_binned']

        existing_columns = [col for col in columns_to_drop if col in self.df.columns]
        self.df = self.df.drop(columns=existing_columns)

        return self

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get the feature-engineered DataFrame.

        Returns:
            Feature-engineered DataFrame
        """
        return self.df

    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if 'target' not in self.df.columns:
            raise ValueError("Target variable not created. Call create_target_variable() first.")

        X = self.df.drop(columns=['관심기업', 'target'])
        y = self.df['target']

        return X, y


def preprocess_pipeline(
    gov_data_path: str,
    worknet_data_path: str,
    data_dir: str = "data"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete preprocessing pipeline from raw data to model-ready features.

    Args:
        gov_data_path: Path to government data file
        worknet_data_path: Path to worknet data file
        data_dir: Directory containing data files

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Load data
    loader = DataLoader(data_dir)
    gov_df = loader.load_government_data(gov_data_path)
    worknet_df = loader.load_worknet_data(worknet_data_path)
    merged_df = loader.merge_datasets(gov_df, worknet_df)

    # Preprocess data
    preprocessor = DataPreprocessor(merged_df)
    preprocessed_df = (
        preprocessor
        .remove_unnecessary_columns()
        .handle_missing_values()
        .clean_interest_column()
        .remove_duplicate_columns()
        .standardize_company_size()
        .standardize_industry()
        .add_derived_features()
        .remove_location_duplicates()
        .remove_outliers()
        .reset_index()
        .get_dataframe()
    )

    # Feature engineering
    engineer = FeatureEngineer(preprocessed_df)
    X, y = (
        engineer
        .create_industry_flags()
        .bin_employee_count()
        .create_brand_flags()
        .drop_non_feature_columns()
        .encode_categorical_features()
        .create_target_variable()
        .get_features_and_target()
    )

    return X, y


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module for Job Seeker Analysis")
    print("=" * 50)
    print("Usage:")
    print("  from src.data_preprocessing import preprocess_pipeline")
    print("  X, y = preprocess_pipeline('2023_strong_sme_government.xlsx', 'youth_worknet_crawling.csv')")
