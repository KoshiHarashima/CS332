# value_generate.py
# Functions to generate values from different distributions for auction simulations

import numpy as np
from typing import Union, Literal


def generate_value(distribution: Literal['uniform', 'exponential'], 
                   size: int = 1,
                   **kwargs) -> Union[float, np.ndarray]:
    """
    Generate values from specified distribution.
    
    Args:
        distribution: 'uniform' or 'exponential'
        size: Number of values to generate (default: 1)
        **kwargs: Distribution-specific parameters:
            - For 'uniform': low (default: 0.0), high (default: 10.0)
            - For 'exponential': scale (default: 1.0)
    
    Returns:
        float if size=1, numpy array if size>1
    
    Examples:
        >>> # Generate single value from uniform distribution
        >>> v = generate_value('uniform', low=5.0, high=15.0)
        
        >>> # Generate multiple values from uniform distribution
        >>> values = generate_value('uniform', size=10, low=0.0, high=20.0)
        
        >>> # Generate single value from exponential distribution
        >>> v = generate_value('exponential', scale=10.0)
        
        >>> # Generate multiple values from exponential distribution
        >>> values = generate_value('exponential', size=5, scale=5.0)
    """
    if distribution == 'uniform':
        low = kwargs.get('low', 0.0)
        high = kwargs.get('high', 10.0)
        
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        
        values = np.random.uniform(low=low, high=high, size=size)
        
    elif distribution == 'exponential':
        scale = kwargs.get('scale', 1.0)
        
        if scale <= 0:
            raise ValueError(f"scale ({scale}) must be positive")
        
        values = np.random.exponential(scale=scale, size=size)
        
    else:
        raise ValueError(f"Unknown distribution: {distribution}. Must be 'uniform' or 'exponential'")
    
    # Return single value if size=1, otherwise return array
    if size == 1:
        return float(values[0])
    else:
        return values


def generate_uniform_value(low: float = 0.0, high: float = 10.0, size: int = 1) -> Union[float, np.ndarray]:
    """
    Convenience function to generate values from uniform distribution.
    
    Args:
        low: Lower bound of uniform distribution (default: 0.0)
        high: Upper bound of uniform distribution (default: 10.0)
        size: Number of values to generate (default: 1)
    
    Returns:
        float if size=1, numpy array if size>1
    """
    return generate_value('uniform', size=size, low=low, high=high)


def generate_exponential_value(scale: float = 1.0, size: int = 1) -> Union[float, np.ndarray]:
    """
    Convenience function to generate values from exponential distribution.
    
    Args:
        scale: Scale parameter (mean) of exponential distribution (default: 1.0)
        size: Number of values to generate (default: 1)
    
    Returns:
        float if size=1, numpy array if size>1
    """
    return generate_value('exponential', size=size, scale=scale)

