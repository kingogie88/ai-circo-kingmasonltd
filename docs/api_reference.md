# API Reference

## Bias Detection

### BiasDetector

The `BiasDetector` class provides tools for identifying and measuring various types of bias in machine learning models and datasets.

```python
from src.bias_detection.bias_detector import BiasDetector

detector = BiasDetector(sensitive_features=['gender', 'race'])
```

#### Parameters

- `sensitive_features` (List[str]): List of column names representing protected attributes

#### Methods

##### evaluate_bias
```python
def evaluate_bias(
    data: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str,
    reference_group: Optional[str] = None
) -> Dict[str, Dict[str, float]]
```

Performs comprehensive bias evaluation combining multiple metrics.

**Parameters:**
- `data`: DataFrame containing the features
- `predictions`: Model predictions
- `target_col`: Name of the target column
- `reference_group`: Optional reference group for comparison

**Returns:**
Dictionary containing all bias metrics

## Fairness Metrics

### FairnessCalculator

The `FairnessCalculator` class provides comprehensive fairness metrics for machine learning models.

```python
from src.fairness_metrics.fairness_calculator import FairnessCalculator

calculator = FairnessCalculator(sensitive_features=['gender', 'race'])
```

#### Parameters

- `sensitive_features` (List[str]): List of column names representing protected attributes

#### Methods

##### evaluate_fairness
```python
def evaluate_fairness(
    data: pd.DataFrame,
    predictions: np.ndarray,
    target_col: str
) -> Dict[str, Union[Dict[str, float], float]]
```

Performs comprehensive fairness evaluation combining multiple metrics.

**Parameters:**
- `data`: DataFrame containing the features
- `predictions`: Model predictions
- `target_col`: Name of the target column

**Returns:**
Dictionary containing all fairness metrics

## Explainability

### ShapExplainer

The `ShapExplainer` class provides model interpretability using SHAP values.

```python
from src.explainability.shap_explainer import ShapExplainer

explainer = ShapExplainer(model, model_type="tree")
```

#### Parameters

- `model` (BaseEstimator): The trained model to explain
- `background_data` (Optional[Union[np.ndarray, pd.DataFrame]]): Background data for SHAP explainer initialization
- `model_type` (str): Type of model ('tree', 'linear', 'kernel', or 'deep')

#### Methods

##### explain_instance
```python
def explain_instance(
    instance: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]
```

Generates SHAP explanations for a single instance.

**Parameters:**
- `instance`: The instance to explain
- `feature_names`: List of feature names

**Returns:**
Dictionary mapping features to their SHAP values

## Privacy Protection

### DifferentialPrivacy

The `DifferentialPrivacy` class implements differential privacy techniques for privacy-preserving machine learning.

```python
from src.privacy_protection.differential_privacy import DifferentialPrivacy

dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
```

#### Parameters

- `epsilon` (float): Privacy budget (smaller values = more privacy)
- `delta` (float): Privacy failure probability
- `mechanism` (str): Privacy mechanism ('laplace' or 'gaussian')
- `clip_threshold` (Optional[float]): Maximum absolute value for clipping
- `random_state` (Optional[int]): Random seed for reproducibility

#### Methods

##### fit_transform
```python
def fit_transform(
    X: Union[np.ndarray, pd.DataFrame],
    y: Optional[np.ndarray] = None
) -> np.ndarray
```

Fits and transforms the data using differential privacy.

**Parameters:**
- `X`: Input data
- `y`: Target values (not used)

**Returns:**
Transformed data with differential privacy

## Safety Monitoring

### ModelMonitor

The `ModelMonitor` class provides tools for monitoring model safety and performance.

```python
from src.safety_monitoring.model_monitoring import ModelMonitor

monitor = ModelMonitor(
    model=model,
    feature_names=feature_names,
    safety_constraints={'age': (18, 100)}
)
```

#### Parameters

- `model` (BaseEstimator): The model to monitor
- `feature_names` (List[str]): List of feature names
- `safety_constraints` (Optional[Dict[str, Tuple[float, float]]]): Dictionary mapping features to (min, max) constraints
- `performance_threshold` (float): Minimum acceptable performance score
- `drift_threshold` (float): Maximum acceptable drift score

#### Methods

##### check_safety_constraints
```python
def check_safety_constraints(
    data: Union[np.ndarray, pd.DataFrame]
) -> Tuple[bool, List[str]]
```

Checks if data meets safety constraints.

**Parameters:**
- `data`: Input data to check

**Returns:**
Tuple of (is_safe, violation_messages)

## Example Usage

Here's a complete example demonstrating the usage of all components:

```python
from src.bias_detection.bias_detector import BiasDetector
from src.fairness_metrics.fairness_calculator import FairnessCalculator
from src.explainability.shap_explainer import ShapExplainer
from src.privacy_protection.differential_privacy import DifferentialPrivacy
from src.safety_monitoring.model_monitoring import ModelMonitor

# Initialize components
bias_detector = BiasDetector(sensitive_features=['gender', 'race'])
fairness_calc = FairnessCalculator(sensitive_features=['gender', 'race'])
explainer = ShapExplainer(model, model_type="tree")
dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
monitor = ModelMonitor(
    model=model,
    feature_names=feature_names,
    safety_constraints={'age': (18, 100)}
)

# Evaluate bias and fairness
bias_metrics = bias_detector.evaluate_bias(data, predictions, target_col)
fairness_metrics = fairness_calc.evaluate_fairness(data, predictions, target_col)

# Generate explanations
explanations = explainer.explain_instance(instance)

# Apply privacy protection
private_data = dp.fit_transform(data)

# Monitor safety
is_safe, violations = monitor.check_safety_constraints(data)
```

For more detailed examples and use cases, please refer to the [examples](../examples/) directory. 