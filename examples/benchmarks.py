"""
Benchmarks for the responsible AI implementation.
"""

import time
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from src.bias_detection.bias_detector import BiasDetector
from src.fairness_metrics.fairness_calculator import FairnessCalculator
from src.explainability.shap_explainer import ShapExplainer
from src.privacy_protection.differential_privacy import DifferentialPrivacy
from src.safety_monitoring.model_monitoring import ModelMonitor

def generate_benchmark_data(n_samples: int = 10000):
    """Generate synthetic data for benchmarking."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Add sensitive attributes
    sensitive_features = pd.DataFrame({
        'gender': np.random.binomial(1, 0.5, n_samples),
        'age_group': np.random.choice(['18-25', '26-35', '36-50', '50+'], n_samples),
        'ethnicity': np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    })
    
    X = np.hstack([X, sensitive_features])
    feature_names = [f'feature_{i}' for i in range(20)] + list(sensitive_features.columns)
    
    return pd.DataFrame(X, columns=feature_names), y

def benchmark_bias_detection(data: pd.DataFrame, predictions: np.ndarray, n_runs: int = 5):
    """Benchmark bias detection performance."""
    detector = BiasDetector(sensitive_features=['gender', 'age_group', 'ethnicity'])
    
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        detector.evaluate_bias(data, predictions, 'target')
        times.append(time.time() - start_time)
    
    return np.mean(times), np.std(times)

def benchmark_fairness_metrics(data: pd.DataFrame, predictions: np.ndarray, n_runs: int = 5):
    """Benchmark fairness metrics calculation."""
    calculator = FairnessCalculator(sensitive_features=['gender', 'age_group', 'ethnicity'])
    
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        calculator.evaluate_fairness(data, predictions, 'target')
        times.append(time.time() - start_time)
    
    return np.mean(times), np.std(times)

def benchmark_explainability(model, data: pd.DataFrame, n_runs: int = 5):
    """Benchmark SHAP explainability."""
    explainer = ShapExplainer(model, model_type="tree")
    
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        explainer.explain_dataset(data.iloc[:100])  # Use subset for efficiency
        times.append(time.time() - start_time)
    
    return np.mean(times), np.std(times)

def benchmark_privacy(data: pd.DataFrame, n_runs: int = 5):
    """Benchmark differential privacy transformation."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
    
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        dp.fit_transform(data)
        times.append(time.time() - start_time)
    
    return np.mean(times), np.std(times)

def plot_benchmarks(results: dict):
    """Plot benchmark results."""
    components = list(results.keys())
    means = [result[0] for result in results.values()]
    stds = [result[1] for result in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(components, means, yerr=stds, capsize=5)
    plt.title('Component Performance Benchmarks')
    plt.xlabel('Component')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

def main():
    """Run benchmarks and generate report."""
    print("Running benchmarks...")
    
    # Generate data
    print("\nGenerating benchmark data...")
    X, y = generate_benchmark_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Run benchmarks
    results = {}
    
    print("\nBenchmarking bias detection...")
    results['Bias Detection'] = benchmark_bias_detection(X_test, predictions)
    
    print("Benchmarking fairness metrics...")
    results['Fairness Metrics'] = benchmark_fairness_metrics(X_test, predictions)
    
    print("Benchmarking explainability...")
    results['Explainability'] = benchmark_explainability(model, X_test)
    
    print("Benchmarking privacy protection...")
    results['Privacy Protection'] = benchmark_privacy(X_test)
    
    # Generate report
    print("\nBenchmark Results:")
    print("================")
    for component, (mean_time, std_time) in results.items():
        print(f"{component}:")
        print(f"  Mean time: {mean_time:.4f} seconds")
        print(f"  Std dev:   {std_time:.4f} seconds")
    
    # Plot results
    plot_benchmarks(results)
    print("\nBenchmark plot saved as 'benchmark_results.png'")

if __name__ == "__main__":
    main() 