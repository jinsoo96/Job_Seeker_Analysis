"""
Utility Functions Module for Job Seeker Analysis

This module provides utility functions for visualization, result export,
and general helper functions.

Author: Yoonju Cho et al.
Reference: Korean Information Systems Society Conference (2023)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
import os
import json
from datetime import datetime


# Set up Korean font (cross-platform)
def setup_korean_font():
    """
    Configure matplotlib to display Korean characters properly.
    """
    import platform

    system = platform.system()

    if system == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        try:
            import matplotlib.font_manager as fm
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rc('font', family=font_name)
        except:
            plt.rc('font', family='AppleGothic')
    elif system == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    else:  # Linux
        plt.rc('font', family='NanumGothic')

    plt.rc('axes', unicode_minus=False)


class Visualizer:
    """Class for creating visualizations."""

    def __init__(self, style: str = 'whitegrid', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize Visualizer.

        Args:
            style: Seaborn style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        sns.set_style(style)
        setup_korean_font()

    def plot_missing_values(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot missing values distribution.

        Args:
            df: DataFrame to analyze
            columns: Columns to check (default: all with missing values)
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if columns is None:
            columns = df.columns[df.isnull().any()].tolist()

        if not columns:
            print("No missing values found.")
            return None

        missing_values = [df[col].isnull().sum() for col in columns]

        fig, ax = plt.subplots(figsize=self.figsize)
        colors = plt.cm.Set3(np.linspace(0, 1, len(columns)))
        ax.bar(columns, missing_values, color=colors)
        ax.set_xlabel('Columns')
        ax.set_ylabel('Missing Values')
        ax.set_title('Missing Values in Each Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_brand_distribution(
        self,
        df: pd.DataFrame,
        column: str = '브랜드명',
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot brand distribution.

        Args:
            df: DataFrame
            column: Column name for brands
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        brand_counts = df[column].value_counts()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=brand_counts.index, y=brand_counts.values, ax=ax)
        ax.set_xlabel('Brand Name (브랜드명)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Strong SME Brands')
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_industry_distribution(
        self,
        df: pd.DataFrame,
        column: str = '업종(중분류)',
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot manufacturing vs non-manufacturing distribution.

        Args:
            df: DataFrame
            column: Column name for industry
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        manufacturing = df[df[column].str.contains('제조업', na=False)]
        non_manufacturing = df[~df[column].str.contains('제조업', na=False)]

        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Manufacturing (제조업)', 'Non-Manufacturing (비제조업)']
        counts = [len(manufacturing), len(non_manufacturing)]

        ax.bar(categories, counts, color=['#3498db', '#e74c3c'])
        ax.set_xlabel('Industry Type')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Industry Types')

        # Add percentage labels
        total = sum(counts)
        for i, (cat, count) in enumerate(zip(categories, counts)):
            ax.annotate(
                f'{count:,} ({count/total*100:.1f}%)',
                xy=(i, count),
                ha='center',
                va='bottom',
                fontsize=12
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_location_distribution(
        self,
        df: pd.DataFrame,
        column: str = '소재지(대)',
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot geographical distribution.

        Args:
            df: DataFrame
            column: Column name for location
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        location_counts = df[column].value_counts()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=location_counts.index, y=location_counts.values, ax=ax)
        ax.set_xlabel('Region (소재지)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Strong SMEs by Region')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = None,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot model comparison results.

        Args:
            results_df: DataFrame with model results
            metrics: List of metrics to plot
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = ['Precision', 'Specificity', 'Sensitivity (Recall)', 'F1-Score']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 0.8, len(results_df)))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.set_title(f'Model Comparison: {metric}')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar, val in zip(bars, results_df[metric]):
                ax.annotate(
                    f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        top_features = importance_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_features)))[::-1]

        bars = ax.barh(
            top_features['Feature'][::-1],
            top_features['Importance'][::-1],
            color=colors
        )

        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top {top_n} Feature Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_coefficient_analysis(
        self,
        coef_df: pd.DataFrame,
        top_n: int = 15,
        save_path: str = None
    ) -> plt.Figure:
        """
        Plot Logistic Regression coefficients with direction.

        Args:
            coef_df: DataFrame with coefficients
            top_n: Number of top features to show
            save_path: Path to save the figure

        Returns:
            Matplotlib figure
        """
        top_features = coef_df.head(top_n)

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = ['#e74c3c' if x < 0 else '#27ae60' for x in top_features['Coefficient'][::-1]]

        ax.barh(
            top_features['Feature'][::-1],
            top_features['Coefficient'][::-1],
            color=colors
        )

        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Coefficient (Logistic Regression)')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Coefficients: Direction of Effect')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#27ae60', label='Positive Effect'),
            Patch(facecolor='#e74c3c', label='Negative Effect')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class ResultExporter:
    """Class for exporting results to various formats."""

    def __init__(self, output_dir: str = 'results'):
        """
        Initialize ResultExporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_to_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Export DataFrame to CSV.

        Args:
            df: DataFrame to export
            filename: Base filename
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Path to saved file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}.csv"
        else:
            filename = f"{filename}.csv"

        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        return filepath

    def export_to_excel(
        self,
        dataframes: Dict[str, pd.DataFrame],
        filename: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Export multiple DataFrames to Excel with multiple sheets.

        Args:
            dataframes: Dictionary of {sheet_name: DataFrame}
            filename: Base filename
            include_timestamp: Whether to include timestamp

        Returns:
            Path to saved file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}.xlsx"
        else:
            filename = f"{filename}.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        return filepath

    def export_model_results(
        self,
        results_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        best_model: str,
        best_params: Dict[str, Any]
    ) -> str:
        """
        Export complete model results.

        Args:
            results_df: Model comparison results
            importance_df: Feature importance
            best_model: Name of best model
            best_params: Best model parameters

        Returns:
            Path to saved Excel file
        """
        # Create summary DataFrame
        summary = pd.DataFrame({
            'Item': ['Best Model', 'Best F1-Score', 'Date'],
            'Value': [
                best_model,
                f"{results_df.loc[results_df['Model'] == best_model, 'F1-Score'].values[0]:.4f}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        })

        # Create parameters DataFrame
        params_df = pd.DataFrame([
            {'Parameter': k, 'Value': str(v)}
            for k, v in best_params.items()
        ])

        dataframes = {
            'Summary': summary,
            'Model Comparison': results_df,
            'Feature Importance': importance_df,
            'Best Model Parameters': params_df
        }

        return self.export_to_excel(dataframes, 'model_results')

    def export_json(
        self,
        data: Dict[str, Any],
        filename: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Export data to JSON.

        Args:
            data: Dictionary to export
            filename: Base filename
            include_timestamp: Whether to include timestamp

        Returns:
            Path to saved file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{filename}_{timestamp}.json"
        else:
            filename = f"{filename}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return filepath


class DataSummary:
    """Class for generating data summaries."""

    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get basic statistics for numerical columns.

        Args:
            df: DataFrame to summarize

        Returns:
            Summary statistics DataFrame
        """
        return df.describe()

    @staticmethod
    def get_categorical_summary(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get value counts for categorical columns.

        Args:
            df: DataFrame to summarize

        Returns:
            Dictionary of {column: value_counts}
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        return {col: df[col].value_counts() for col in categorical_cols}

    @staticmethod
    def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Get missing value summary.

        Args:
            df: DataFrame to analyze

        Returns:
            Missing value summary DataFrame
        """
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100

        return pd.DataFrame({
            'Missing Count': missing,
            'Missing Percent': missing_percent
        }).sort_values('Missing Count', ascending=False)


if __name__ == "__main__":
    print("Utility Functions Module for Job Seeker Analysis")
    print("=" * 50)
    print("Classes:")
    print("  - Visualizer: Create various visualizations")
    print("  - ResultExporter: Export results to CSV/Excel/JSON")
    print("  - DataSummary: Generate data summaries")
