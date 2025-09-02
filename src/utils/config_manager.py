"""
Configuration management utilities.
"""
import yaml
from pathlib import Path
from loguru import logger
from typing import Dict, Any


def load_config(config_path: str) -> Dict[Any, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise