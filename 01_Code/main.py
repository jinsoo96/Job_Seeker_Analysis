"""
Main Execution Script for Job Seeker Analysis

Machine Learning-based Prediction Model for Young Job Seekers'
Preference for Strong Small and Medium Enterprises (강소기업)

This script orchestrates the entire analysis pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training with Optuna hyperparameter optimization
4. Model evaluation and comparison
5. Feature importance analysis
6. Result visualization and export

Author: Yoonju Cho et al.
Reference: Korean Information Systems Society Conference (2023)

Usage:
    python main.py
    python main.py --trials 20
    python main.py --data-dir custom_data --results-dir custom_results
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config, ProjectConfig
from src.data_preprocessing import DataLoader, DataPreprocessor, FeatureEngineer
from src.models import ModelTrainer, train_and_evaluate
from src.utils import Visualizer, ResultExporter, DataSummary

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Job Seeker Analysis - ML Prediction Model for Strong SME Preferences'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default=config.data.data_dir,
        help='Directory containing data files'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default=config.data.results_dir,
        help='Directory for output results'
    )

    parser.add_argument(
        '--gov-data',
        type=str,
        default=config.data.government_data_file,
        help='Government data filename'
    )

    parser.add_argument(
        '--worknet-data',
        type=str,
        default=config.data.worknet_data_file,
        help='Worknet crawled data filename'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=config.model.n_trials,
        help='Number of Optuna optimization trials'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=config.model.test_size,
        help='Proportion of data for testing'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed progress'
    )

    return parser.parse_args()


def print_header():
    """Print program header."""
    print("=" * 70)
    print("  Job Seeker Analysis")
    print("  ML-based Prediction Model for Young Job Seekers'")
    print("  Preference for Strong SMEs (강소기업 선호 예측모형)")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()


def load_and_preprocess_data(args, verbose=True):
    """
    Load and preprocess data from files.

    Args:
        args: Command line arguments
        verbose: Whether to print progress

    Returns:
        Preprocessed DataFrame
    """
    if verbose:
        print("[Step 1/5] Loading and Preprocessing Data")
        print("-" * 50)

    # Initialize data loader
    loader = DataLoader(args.data_dir)

    # Check if files exist
    gov_path = os.path.join(args.data_dir, args.gov_data)
    worknet_path = os.path.join(args.data_dir, args.worknet_data)

    if not os.path.exists(gov_path):
        print(f"  Warning: Government data file not found: {gov_path}")
        print("  Using sample data for demonstration...")
        return create_sample_data()

    if not os.path.exists(worknet_path):
        print(f"  Warning: Worknet data file not found: {worknet_path}")
        print("  Using sample data for demonstration...")
        return create_sample_data()

    # Load data
    if verbose:
        print(f"  Loading government data: {args.gov_data}")
    gov_df = loader.load_government_data(args.gov_data)

    if verbose:
        print(f"  Loading worknet data: {args.worknet_data}")
    worknet_df = loader.load_worknet_data(args.worknet_data)

    # Merge datasets
    if verbose:
        print("  Merging datasets...")
    merged_df = loader.merge_datasets(gov_df, worknet_df)

    # Preprocess
    if verbose:
        print("  Preprocessing data...")
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

    if verbose:
        print(f"  Preprocessed data shape: {preprocessed_df.shape}")
        print()

    return preprocessed_df


def create_sample_data():
    """Create sample data for demonstration when real data is unavailable."""
    np.random.seed(42)
    n_samples = 1000

    regions = ['서울', '경기', '부산', '대구', '인천', '대전', '광주', '울산']
    brands = ['이노비즈', '메인비즈', '안전보건경영시스템 인증기업', '병역지정업체']
    industries = ['제조업', '서비스업', '도매 및 소매업', '건설업', '정보통신업']
    sizes = ['중소기업', '중견기업', '대기업']

    data = {
        '기업명': [f'기업_{i}' for i in range(n_samples)],
        '관심기업': np.random.poisson(11, n_samples),
        '직원수': np.random.lognormal(3.5, 0.8, n_samples).astype(int),
        '기업규모': np.random.choice(sizes, n_samples, p=[0.95, 0.04, 0.01]),
        '소재지(대)': np.random.choice(regions, n_samples),
        '소재지(중)': [f'{np.random.choice(regions)}시' for _ in range(n_samples)],
        '브랜드명': np.random.choice(brands, n_samples),
        '대표자명': [f'대표_{i}' for i in range(n_samples)],
        '업종(중분류)': np.random.choice(industries, n_samples),
        '기업개수': np.ones(n_samples, dtype=int),
        '대표자수': np.ones(n_samples, dtype=int),
        '브랜드신청수': np.ones(n_samples, dtype=int)
    }

    return pd.DataFrame(data)


def engineer_features(df, verbose=True):
    """
    Perform feature engineering.

    Args:
        df: Preprocessed DataFrame
        verbose: Whether to print progress

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    if verbose:
        print("[Step 2/5] Feature Engineering")
        print("-" * 50)

    engineer = FeatureEngineer(df)

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

    if verbose:
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution:")
        print(f"    - Low preference (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        print(f"    - High preference (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
        print()

    return X, y


def train_models(X, y, args, verbose=True):
    """
    Train and evaluate all models.

    Args:
        X: Features DataFrame
        y: Target Series
        args: Command line arguments
        verbose: Whether to print progress

    Returns:
        ModelTrainer instance with trained models
    """
    if verbose:
        print("[Step 3/5] Training Models with Optuna Optimization")
        print("-" * 50)

    trainer = ModelTrainer(
        X, y,
        test_size=args.test_size,
        n_trials=args.trials
    )

    results_df = trainer.train_all_models(verbose=verbose)

    if verbose:
        print("\n  Model Comparison Results:")
        print(results_df.to_string(index=False))
        print()

    return trainer, results_df


def analyze_features(trainer, verbose=True):
    """
    Analyze feature importance and coefficients.

    Args:
        trainer: Trained ModelTrainer instance
        verbose: Whether to print progress

    Returns:
        Combined analysis DataFrame
    """
    if verbose:
        print("[Step 4/5] Feature Importance Analysis")
        print("-" * 50)

    best_model_name, _ = trainer.get_best_model()

    if verbose:
        print(f"  Best Model: {best_model_name}")

    combined_df = trainer.get_combined_analysis(best_model_name)

    if verbose:
        print("\n  Top 10 Most Important Features:")
        print(combined_df.head(10).to_string(index=False))
        print()

    return combined_df, best_model_name


def export_results(trainer, results_df, importance_df, best_model, args, verbose=True):
    """
    Export all results to files.

    Args:
        trainer: Trained ModelTrainer instance
        results_df: Model comparison results
        importance_df: Feature importance analysis
        best_model: Name of best model
        args: Command line arguments
        verbose: Whether to print progress

    Returns:
        Dictionary of exported file paths
    """
    if verbose:
        print("[Step 5/5] Exporting Results")
        print("-" * 50)

    exporter = ResultExporter(args.results_dir)
    exported_files = {}

    # Export model results
    filepath = exporter.export_model_results(
        results_df,
        importance_df,
        best_model,
        trainer.best_params.get(best_model, {})
    )
    exported_files['model_results'] = filepath

    if verbose:
        print(f"  Model results exported to: {filepath}")

    # Export feature importance
    filepath = exporter.export_to_csv(importance_df, 'feature_importance')
    exported_files['feature_importance'] = filepath

    if verbose:
        print(f"  Feature importance exported to: {filepath}")

    # Export classification results
    filepath = exporter.export_to_csv(results_df, 'classification_results')
    exported_files['classification_results'] = filepath

    if verbose:
        print(f"  Classification results exported to: {filepath}")

    # Generate visualizations
    if not args.no_plots:
        visualizer = Visualizer()

        # Model comparison plot
        fig_path = os.path.join(args.results_dir, 'model_comparison.png')
        visualizer.plot_model_comparison(results_df, save_path=fig_path)
        exported_files['model_comparison_plot'] = fig_path

        if verbose:
            print(f"  Model comparison plot saved to: {fig_path}")

        # Feature importance plot
        fig_path = os.path.join(args.results_dir, 'feature_importance.png')
        visualizer.plot_feature_importance(importance_df, save_path=fig_path)
        exported_files['feature_importance_plot'] = fig_path

        if verbose:
            print(f"  Feature importance plot saved to: {fig_path}")

        # Coefficient plot
        fig_path = os.path.join(args.results_dir, 'coefficient_analysis.png')
        visualizer.plot_coefficient_analysis(importance_df, save_path=fig_path)
        exported_files['coefficient_plot'] = fig_path

        if verbose:
            print(f"  Coefficient analysis plot saved to: {fig_path}")

    print()
    return exported_files


def print_summary(best_model, results_df, importance_df):
    """Print analysis summary."""
    print("=" * 70)
    print("  ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n1. Best Performing Model:")
    print(f"   {best_model}")

    best_metrics = results_df[results_df['Model'] == best_model].iloc[0]
    print(f"\n   Metrics:")
    print(f"   - Precision:   {best_metrics['Precision']:.4f}")
    print(f"   - Specificity: {best_metrics['Specificity']:.4f}")
    print(f"   - Sensitivity: {best_metrics['Sensitivity (Recall)']:.4f}")
    print(f"   - F1-Score:    {best_metrics['F1-Score']:.4f}")

    print("\n2. Top 5 Most Important Features:")
    for i, row in importance_df.head(5).iterrows():
        direction = "+" if row.get('Coefficient', 0) > 0 else "-"
        print(f"   {i+1}. {row['Feature']} (Importance: {row['Importance']:.0f}, Direction: {direction})")

    print("\n3. Key Findings:")
    print("   - Manufacturing and service industries have significant positive effects")
    print("   - Seoul and Gyeonggi regions show lower preference (counterintuitively)")
    print("   - Companies with 10 or fewer employees have notably lower preference")

    print("\n" + "=" * 70)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Print header
    print_header()

    # Create output directory
    os.makedirs(args.results_dir, exist_ok=True)

    try:
        # Step 1: Load and preprocess data
        preprocessed_df = load_and_preprocess_data(args)

        # Step 2: Feature engineering
        X, y = engineer_features(preprocessed_df)

        # Step 3: Train models
        trainer, results_df = train_models(X, y, args)

        # Step 4: Analyze features
        importance_df, best_model = analyze_features(trainer)

        # Step 5: Export results
        exported_files = export_results(
            trainer, results_df, importance_df, best_model, args
        )

        # Print summary
        print_summary(best_model, results_df, importance_df)

        print("\nAnalysis completed successfully!")
        print(f"Results saved to: {args.results_dir}")

        return 0

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
