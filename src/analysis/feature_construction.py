"""
Feature construction module for orderbook data analysis.
"""
from typing import List, Dict, Any, Callable
import pandas as pd
import numpy as np
from loguru import logger

def build_features(df: pd.DataFrame, feature_functions: List[Callable]) -> pd.DataFrame:
    """
    Build features from orderbook data using provided feature functions.
    
    Args:
        df: DataFrame with orderbook data
        feature_functions: List of functions that construct features
        
    Returns:
        DataFrame with constructed features
    """
    result_df = df.copy()
    
    for feature_func in feature_functions:
        try:
            result_df = feature_func(result_df)
        except Exception as e:
            logger.error(f"Error applying feature function {feature_func.__name__}: {e}")
            
    return result_df

def register_feature(name: str, feature_func: Callable) -> Callable:
    """
    Decorator to register feature functions.
    
    Args:
        name: Name of the feature
        feature_func: Function that constructs the feature
        
    Returns:
        The feature function with metadata
    """
    feature_func.feature_name = name
    return feature_func