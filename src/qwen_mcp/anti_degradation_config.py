# src/qwen_mcp/anti_degradation_config.py
"""
Configuration loader for the Anti-Degradation System.

Provides typed configuration via dataclasses with YAML loading,
environment variable overrides, and singleton pattern.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(".anti_degradation/config.yaml")


@dataclass
class ShadowModeConfig:
    """Configuration for shadow mode operations."""
    enabled: bool = True
    log_level: str = "INFO"
    audit_output: str = ".anti_degradation/audit_logs"


@dataclass
class ProductionModeConfig:
    """Configuration for production mode operations."""
    enabled: bool = False
    block_on_regression: bool = True
    block_threshold: float = 0.8


@dataclass
class ThresholdsConfig:
    """Configuration for system thresholds."""
    max_latency_seconds: float = 5.0
    regression_risk_threshold: float = 0.7
    min_files_for_baseline: int = 10


@dataclass
class FilePatternsConfig:
    """Configuration for file inclusion/exclusion patterns."""
    include: List[str] = field(default_factory=lambda: ["*.py", "*.js", "*.ts"])
    exclude: List[str] = field(default_factory=lambda: ["*.test.py", "*.spec.ts", "node_modules/**"])


@dataclass
class SnapshotsConfig:
    """Configuration for snapshot storage."""
    storage_dir: str = ".anti_degradation/snapshots"
    baseline_name: str = "baseline"
    max_snapshots: int = 10


@dataclass
class AntiDegradationConfig:
    """Main configuration container for Anti-Degradation System."""
    shadow_mode: ShadowModeConfig = field(default_factory=ShadowModeConfig)
    production_mode: ProductionModeConfig = field(default_factory=ProductionModeConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    file_patterns: FilePatternsConfig = field(default_factory=FilePatternsConfig)
    snapshots: SnapshotsConfig = field(default_factory=SnapshotsConfig)
    config_path: Path = field(default=DEFAULT_CONFIG_PATH)


class ConfigLoader:
    """
    Singleton configuration loader for Anti-Degradation System.
    
    Loads configuration from YAML file with support for:
    - Default values when config file is missing
    - Environment variable overrides
    - Singleton pattern for efficiency
    """
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[AntiDegradationConfig] = None
    
    def __new__(cls, config_path: Optional[Path] = None) -> 'ConfigLoader':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized:
            return
        
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self._initialized = True
    
    def _load_yaml_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(
                f"Config file not found at {self.config_path}, using defaults"
            )
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                logger.info(f"Loaded config from {self.config_path}")
                return config_data or {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            return {}
        except IOError as e:
            logger.error(f"Failed to read config file: {e}")
            return {}
    
    def _apply_env_overrides(self, config: AntiDegradationConfig) -> AntiDegradationConfig:
        """Apply environment variable overrides to configuration."""
        # Shadow mode override
        shadow_env = os.getenv("ANTI_DEGRADATION_SHADOW_MODE")
        if shadow_env is not None:
            config.shadow_mode.enabled = shadow_env.lower() in ("true", "1", "yes")
            logger.debug(f"Shadow mode overridden by env: {config.shadow_mode.enabled}")
        
        # Production mode override
        prod_env = os.getenv("ANTI_DEGRADATION_PRODUCTION")
        if prod_env is not None:
            config.production_mode.enabled = prod_env.lower() in ("true", "1", "yes")
            logger.debug(f"Production mode overridden by env: {config.production_mode.enabled}")
        
        return config
    
    def _build_config_from_dict(self, data: dict) -> AntiDegradationConfig:
        """Build AntiDegradationConfig from dictionary data."""
        shadow_data = data.get("shadow_mode", {})
        production_data = data.get("production_mode", {})
        thresholds_data = data.get("thresholds", {})
        file_patterns_data = data.get("file_patterns", {})
        snapshots_data = data.get("snapshots", {})
        
        config = AntiDegradationConfig(
            config_path=self.config_path,
            shadow_mode=ShadowModeConfig(
                enabled=shadow_data.get("enabled", True),
                log_level=shadow_data.get("log_level", "INFO"),
                audit_output=shadow_data.get("audit_output", ".anti_degradation/audit_logs")
            ),
            production_mode=ProductionModeConfig(
                enabled=production_data.get("enabled", False),
                block_on_regression=production_data.get("block_on_regression", True),
                block_threshold=production_data.get("block_threshold", 0.8)
            ),
            thresholds=ThresholdsConfig(
                max_latency_seconds=thresholds_data.get("max_latency_seconds", 5.0),
                regression_risk_threshold=thresholds_data.get("regression_risk_threshold", 0.7),
                min_files_for_baseline=thresholds_data.get("min_files_for_baseline", 10)
            ),
            file_patterns=FilePatternsConfig(
                include=file_patterns_data.get("include", ["*.py", "*.js", "*.ts"]),
                exclude=file_patterns_data.get("exclude", ["*.test.py", "*.spec.ts", "node_modules/**"])
            ),
            snapshots=SnapshotsConfig(
                storage_dir=snapshots_data.get("storage_dir", ".anti_degradation/snapshots"),
                baseline_name=snapshots_data.get("baseline_name", "baseline"),
                max_snapshots=snapshots_data.get("max_snapshots", 10)
            )
        )
        
        return self._apply_env_overrides(config)
    
    def load(self, config_path: Optional[Path] = None) -> AntiDegradationConfig:
        """
        Load configuration from file with defaults and env overrides.
        
        Args:
            config_path: Optional path to config file (overrides instance path)
        
        Returns:
            AntiDegradationConfig instance
        """
        if config_path is not None:
            self.config_path = config_path
        
        yaml_data = self._load_yaml_config()
        config = self._build_config_from_dict(yaml_data)
        
        logger.info(
            f"Config loaded: shadow={config.shadow_mode.enabled}, "
            f"production={config.production_mode.enabled}"
        )
        
        return config
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._config = None


@lru_cache(maxsize=1)
def get_config(config_path: Optional[str] = None) -> AntiDegradationConfig:
    """
    Get singleton configuration instance.
    
    Args:
        config_path: Optional path to config file (string or Path)
    
    Returns:
        AntiDegradationConfig instance with loaded configuration
    """
    loader = ConfigLoader()
    
    if config_path is not None:
        path = Path(config_path)
    else:
        path = DEFAULT_CONFIG_PATH
    
    return loader.load(path)


def reload_config(config_path: Optional[str] = None) -> AntiDegradationConfig:
    """
    Force reload configuration (bypasses cache).
    
    Args:
        config_path: Optional path to config file
    
    Returns:
        Fresh AntiDegradationConfig instance
    """
    ConfigLoader.reset()
    get_config.cache_clear()
    
    return get_config(config_path)


__all__ = [
    "AntiDegradationConfig",
    "ShadowModeConfig",
    "ProductionModeConfig",
    "ThresholdsConfig",
    "FilePatternsConfig",
    "SnapshotsConfig",
    "ConfigLoader",
    "get_config",
    "reload_config",
]