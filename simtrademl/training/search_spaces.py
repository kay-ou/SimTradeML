"""Hyperparameter search spaces for model optimization.

This module defines model-specific hyperparameter search spaces for use with
Optuna optimization. Each search space is defined as a dictionary mapping
parameter names to tuples specifying the search range.

Tuple formats:
    - (min, max): Uniform distribution between min and max
    - (min, max, "log"): Log-scale distribution between min and max

The search spaces are carefully designed based on empirical best practices
for each model type, balancing exploration and computational efficiency.
"""

from typing import Dict, Tuple, Union

# Type alias for search space value
SearchSpaceValue = Union[Tuple[float, float], Tuple[float, float, str], Tuple[int, int]]

# Model-specific hyperparameter search spaces
SEARCH_SPACES: Dict[str, Dict[str, SearchSpaceValue]] = {
    "xgboost": {
        # Maximum tree depth
        # Range: 3-10 to prevent overfitting while allowing sufficient complexity
        "max_depth": (3, 10),

        # Learning rate (step size shrinkage)
        # Range: 0.01-0.3 on log scale for efficient exploration of small values
        "learning_rate": (0.01, 0.3, "log"),

        # Number of boosting rounds
        # Range: 100-1000 to balance training time and model performance
        "n_estimators": (100, 1000),

        # Subsample ratio of training instances
        # Range: 0.6-1.0 to prevent overfitting through row sampling
        "subsample": (0.6, 1.0),

        # Subsample ratio of columns when constructing each tree
        # Range: 0.6-1.0 to prevent overfitting through column sampling
        "colsample_bytree": (0.6, 1.0),

        # Minimum sum of instance weight needed in a child
        # Range: 1-10 to control tree complexity
        "min_child_weight": (1, 10),
    },
    "lightgbm": {
        # Maximum number of leaves in one tree
        # Range: 20-100 to control model complexity
        "num_leaves": (20, 100),

        # Learning rate (shrinkage rate)
        # Range: 0.01-0.3 on log scale for efficient exploration
        "learning_rate": (0.01, 0.3, "log"),

        # Number of boosting iterations
        # Range: 100-1000 to balance performance and training time
        "n_estimators": (100, 1000),

        # Subsample ratio of training data
        # Range: 0.6-1.0 for regularization through bagging
        "subsample": (0.6, 1.0),

        # Subsample ratio of columns when constructing each tree
        # Range: 0.6-1.0 for feature sampling regularization
        "colsample_bytree": (0.6, 1.0),

        # Minimum number of data points in one leaf
        # Range: 10-100 to prevent overfitting on small groups
        "min_child_samples": (10, 100),
    },
}


def get_search_space(model_type: str) -> Dict[str, SearchSpaceValue]:
    """Get search space for a specific model type.

    Args:
        model_type: Model type identifier (e.g., 'xgboost', 'lightgbm')

    Returns:
        Dictionary mapping parameter names to search ranges

    Raises:
        KeyError: If model_type is not in SEARCH_SPACES

    Example:
        >>> space = get_search_space("xgboost")
        >>> space["max_depth"]
        (3, 10)
        >>> space["learning_rate"]
        (0.01, 0.3, 'log')
    """
    if model_type not in SEARCH_SPACES:
        raise KeyError(
            f"No search space defined for model type '{model_type}'. "
            f"Available types: {list(SEARCH_SPACES.keys())}"
        )
    return SEARCH_SPACES[model_type]


def get_available_model_types() -> list[str]:
    """Get list of model types with defined search spaces.

    Returns:
        List of model type identifiers

    Example:
        >>> get_available_model_types()
        ['xgboost', 'lightgbm']
    """
    return list(SEARCH_SPACES.keys())
