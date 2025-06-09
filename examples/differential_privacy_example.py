"""
Example demonstrating the usage of the Differential Privacy module.

This example shows how to apply differential privacy techniques to protect
sensitive information in a dataset while maintaining utility for analysis.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.privacy_protection.differential_privacy import DifferentialPrivacy

def create_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create a synthetic dataset with sensitive information.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame containing synthetic data
    """
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'loan_amount': np.random.normal(200000, 100000, n_samples),
        'years_employed': np.random.randint(0, 40, n_samples)
    })
    
    return data

def main():
    # Create synthetic dataset
    print("Creating synthetic dataset with sensitive information...")
    data = create_synthetic_dataset()
    
    print("\nOriginal Data Summary:")
    print(data.describe())
    
    # Initialize differential privacy
    print("\nInitializing differential privacy with conservative privacy budget...")
    dp = DifferentialPrivacy(
        epsilon=0.1,  # Strong privacy guarantee
        delta=1e-6,
        sensitivity=1.0
    )
    
    # Define sensitive columns
    sensitive_columns = ['income', 'loan_amount', 'credit_score']
    
    # Apply differential privacy to the dataset
    print("\nApplying differential privacy to sensitive columns...")
    private_data, privacy_metrics = dp.privatize_dataset(
        data,
        sensitive_columns=sensitive_columns
    )
    
    print("\nPrivatized Data Summary:")
    print(private_data.describe())
    
    # Show privacy metrics for each column
    print("\nPrivacy Metrics per Column:")
    for column, metrics in privacy_metrics.items():
        print(f"\n{column}:")
        print(f"  Epsilon Used: {metrics.epsilon_used:.6f}")
        print(f"  Noise Scale: {metrics.noise_scale:.6f}")
        print(f"  Privacy Guarantee: {metrics.privacy_guarantee}")
    
    # Demonstrate private aggregation
    print("\nDemonstrating private aggregation...")
    
    # Calculate private mean income
    private_mean, mean_metrics = dp.privatize_aggregation(
        data['income'].values,
        operation='mean'
    )
    
    print("\nIncome Statistics:")
    print(f"True Mean: {data['income'].mean():.2f}")
    print(f"Private Mean: {private_mean:.2f}")
    print(f"Privacy Guarantee: {mean_metrics.privacy_guarantee}")
    
    # Calculate private sum of loan amounts
    private_sum, sum_metrics = dp.privatize_aggregation(
        data['loan_amount'].values,
        operation='sum'
    )
    
    print("\nLoan Amount Statistics:")
    print(f"True Sum: {data['loan_amount'].sum():.2f}")
    print(f"Private Sum: {private_sum:.2f}")
    print(f"Privacy Guarantee: {sum_metrics.privacy_guarantee}")
    
    # Show overall privacy report
    print("\nFinal Privacy Report:")
    print(dp.get_privacy_report())

if __name__ == "__main__":
    main() 