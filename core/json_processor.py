"""
JSON processing utilities for multi-timeframe strategy configuration.

This module provides utilities for loading, validating, and processing
JSON-based strategy configurations, including schema validation and
error handling.
"""

import json
import jsonschema
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from .json_schemas import (
    STRATEGY_SCHEMA,
    INDICATOR_SCHEMA,
    RISK_SCHEMA,
    TIMING_SCHEMA,
    IndicatorType,
    TimeframeType,
    SignalDirection,
    ExecutionMode,
    get_example_strategy,
    get_single_timeframe_example
)


logger = logging.getLogger(__name__)


class JSONProcessingError(Exception):
    """Base exception for JSON processing errors."""
    pass


class ValidationError(JSONProcessingError):
    """Raised when JSON validation fails."""
    pass


class ConversionError(JSONProcessingError):
    """Raised when JSON to object conversion fails."""
    pass


@dataclass
class ValidationResult:
    """Result of JSON validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    schema_version: Optional[str] = None


class JSONProcessor:
    """
    Main processor for handling JSON strategy configurations.

    Provides methods for loading, validating, and converting JSON
    configurations to internal objects.
    """

    def __init__(self):
        """Initialize the JSON processor."""
        self.schemas = {
            'strategy': STRATEGY_SCHEMA,
            'indicator': INDICATOR_SCHEMA,
            'risk': RISK_SCHEMA,
            'timing': TIMING_SCHEMA
        }
        self.validator = jsonschema.Draft7Validator(STRATEGY_SCHEMA)

    def load_strategy_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load strategy configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file

        Returns:
            Dictionary containing strategy configuration

        Raises:
            JSONProcessingError: If file cannot be loaded or parsed
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise JSONProcessingError(f"Strategy file not found: {file_path}")

            if not path.suffix.lower() == '.json':
                raise JSONProcessingError(f"File must have .json extension: {file_path}")

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            logger.info(f"Successfully loaded strategy from {file_path}")
            return data

        except json.JSONDecodeError as e:
            raise JSONProcessingError(f"Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            raise JSONProcessingError(f"Error loading strategy file {file_path}: {e}")

    def save_strategy_to_file(self, strategy: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save strategy configuration to JSON file.

        Args:
            strategy: Strategy configuration dictionary
            file_path: Path where to save the JSON file

        Raises:
            JSONProcessingError: If file cannot be saved
        """
        try:
            # Validate before saving
            validation = self.validate_strategy(strategy)
            if not validation.is_valid:
                raise ValidationError(f"Cannot save invalid strategy: {validation.errors}")

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Add metadata
            if 'metadata' not in strategy:
                strategy['metadata'] = {}

            strategy['metadata']['last_modified'] = datetime.utcnow().isoformat() + 'Z'

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(strategy, f, indent=2, ensure_ascii=False)

            logger.info(f"Successfully saved strategy to {file_path}")

        except Exception as e:
            raise JSONProcessingError(f"Error saving strategy to {file_path}: {e}")

    def validate_strategy(self, strategy: Dict[str, Any]) -> ValidationResult:
        """
        Validate strategy configuration against schema.

        Args:
            strategy: Strategy configuration to validate

        Returns:
            ValidationResult with validation status and any errors
        """
        errors = []
        warnings = []

        try:
            # Schema validation
            schema_errors = list(self.validator.iter_errors(strategy))
            for error in schema_errors:
                error_path = " -> ".join(str(p) for p in error.path)
                errors.append(f"{error_path}: {error.message}")

            # Business logic validation
            business_errors, business_warnings = self._validate_business_logic(strategy)
            errors.extend(business_errors)
            warnings.extend(business_warnings)

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                schema_version="1.0.0"
            )

        except Exception as e:
            errors.append(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings
            )

    def _validate_business_logic(self, strategy: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """
        Perform business logic validation beyond schema checking.

        Args:
            strategy: Strategy configuration to validate

        Returns:
            Tuple of (errors, warnings)
        """
        errors = []
        warnings = []

        try:
            # Validate indicators
            indicators = strategy.get('indicators', [])
            if not indicators:
                errors.append("Strategy must have at least one indicator")

            # Check for duplicate indicators on same timeframe
            seen_combinations = set()
            for indicator in indicators:
                combo = (indicator.get('type'), indicator.get('timeframe'))
                if combo in seen_combinations:
                    errors.append(f"Duplicate indicator {combo[0]} on timeframe {combo[1]}")
                seen_combinations.add(combo)

            # Validate timeframe consistency
            timeframes = [ind.get('timeframe') for ind in indicators]
            primary_tf = strategy.get('execution', {}).get('primary_timeframe')
            if primary_tf and primary_tf not in timeframes:
                errors.append(f"Primary timeframe {primary_tf} not used by any indicator")

            # Validate risk parameters
            risk = strategy.get('risk', {})
            if risk:
                sl_pips = risk.get('stop_loss', {}).get('pips', 0)
                tp_pips = risk.get('take_profit', {}).get('pips', 0)

                if sl_pips <= 0:
                    errors.append("Stop loss pips must be greater than 0")
                if tp_pips <= 0:
                    errors.append("Take profit pips must be greater than 0")

                # Risk-reward ratio warning
                if sl_pips > 0 and tp_pips > 0:
                    rr_ratio = tp_pips / sl_pips
                    if rr_ratio < 1.0:
                        warnings.append(f"Risk-reward ratio is {rr_ratio:.2f} (< 1.0)")

            # Validate execution mode vs indicators
            execution_mode = strategy.get('execution', {}).get('mode')
            if execution_mode == 'sequential' and len(indicators) == 1:
                warnings.append("Sequential mode with single indicator - consider parallel mode")

        except Exception as e:
            errors.append(f"Business logic validation error: {e}")

        return errors, warnings

    def convert_to_legacy_format(self, json_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert JSON strategy to legacy format for backward compatibility.

        Args:
            json_strategy: JSON strategy configuration

        Returns:
            Dictionary in legacy format

        Raises:
            ConversionError: If conversion fails
        """
        try:
            # Extract basic info
            name = json_strategy.get('name', 'Unnamed Strategy')
            indicators = json_strategy.get('indicators', [])
            risk = json_strategy.get('risk', {})
            timing = json_strategy.get('timing', {})
            execution = json_strategy.get('execution', {})

            # Convert indicators to legacy sequence
            indicator_sequence = []
            indicator_configs = {}

            for indicator in indicators:
                ind_type = indicator.get('type')
                ind_timeframe = indicator.get('timeframe')
                ind_params = indicator.get('params', {})

                # Create unique key for multi-timeframe support
                ind_key = f"{ind_type}_{ind_timeframe}" if len(indicators) > 1 else ind_type

                indicator_sequence.append(ind_key)
                indicator_configs[ind_key] = ind_params

            # Convert risk settings
            risk_percent = risk.get('position_sizing', {}).get('value', 0.01)
            stop_loss_pips = risk.get('stop_loss', {}).get('pips', 25)
            take_profit_pips = risk.get('take_profit', {}).get('pips', 50)

            # Convert timing settings
            setup_timeout = timing.get('setup_timeout', 30)
            signal_timeout = timing.get('signal_timeout', 20)

            # Build legacy format
            legacy_config = {
                'name': name,
                'indicator_sequence': indicator_sequence,
                'required_confirmations': execution.get('confirmation_required', len(indicators)),
                'timeouts': {
                    'default': setup_timeout,
                    **{seq: signal_timeout for seq in indicator_sequence}
                },
                'stop_loss_pips': stop_loss_pips,
                'take_profit_pips': take_profit_pips,
                'risk_percent': risk_percent,
                'indicator_configs': indicator_configs,
                'primary_timeframe': execution.get('primary_timeframe', 'H1'),
                'json_source': True  # Flag to indicate JSON origin
            }

            logger.info(f"Successfully converted JSON strategy '{name}' to legacy format")
            return legacy_config

        except Exception as e:
            raise ConversionError(f"Failed to convert JSON strategy to legacy format: {e}")

    def create_example_files(self, output_dir: Union[str, Path]) -> List[Path]:
        """
        Create example strategy JSON files.

        Args:
            output_dir: Directory where to create example files

        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        created_files = []

        # Multi-timeframe example
        multi_tf_example = get_example_strategy()
        multi_tf_path = output_path / "multi_timeframe_strategy.json"
        self.save_strategy_to_file(multi_tf_example, multi_tf_path)
        created_files.append(multi_tf_path)

        # Single timeframe example
        single_tf_example = get_single_timeframe_example()
        single_tf_path = output_path / "single_timeframe_strategy.json"
        self.save_strategy_to_file(single_tf_example, single_tf_path)
        created_files.append(single_tf_path)

        logger.info(f"Created {len(created_files)} example strategy files in {output_dir}")
        return created_files

    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about supported schemas and their properties.

        Returns:
            Dictionary with schema information
        """
        return {
            'version': '1.0.0',
            'supported_timeframes': [tf.value for tf in TimeframeType],
            'supported_indicators': [ind.value for ind in IndicatorType],
            'execution_modes': [mode.value for mode in ExecutionMode],
            'signal_directions': [dir.value for dir in SignalDirection],
            'schema_keys': list(self.schemas.keys()),
            'required_fields': STRATEGY_SCHEMA['required']
        }

    def merge_strategies(self, strategies: List[Dict[str, Any]],
                        merge_name: str = "Merged Strategy") -> Dict[str, Any]:
        """
        Merge multiple strategies into a single strategy.

        Args:
            strategies: List of strategy configurations to merge
            merge_name: Name for the merged strategy

        Returns:
            Merged strategy configuration

        Raises:
            JSONProcessingError: If strategies cannot be merged
        """
        if not strategies:
            raise JSONProcessingError("Cannot merge empty list of strategies")

        try:
            # Start with first strategy as base
            merged = strategies[0].copy()
            merged['name'] = merge_name
            merged['version'] = "1.0.0"
            merged['description'] = f"Merged strategy from {len(strategies)} source strategies"

            # Merge indicators from all strategies
            all_indicators = []
            for strategy in strategies:
                all_indicators.extend(strategy.get('indicators', []))

            # Remove duplicates and conflicts
            unique_indicators = []
            seen = set()
            for indicator in all_indicators:
                key = (indicator.get('type'), indicator.get('timeframe'))
                if key not in seen:
                    unique_indicators.append(indicator)
                    seen.add(key)

            merged['indicators'] = unique_indicators

            # Use most conservative risk settings
            risk_configs = [s.get('risk', {}) for s in strategies if s.get('risk')]
            if risk_configs:
                merged['risk'] = self._merge_risk_configs(risk_configs)

            # Update metadata
            merged['metadata'] = {
                'created_by': 'Strategy Merger',
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'source_strategies': [s.get('name', 'Unknown') for s in strategies],
                'merge_timestamp': datetime.utcnow().isoformat() + 'Z'
            }

            return merged

        except Exception as e:
            raise JSONProcessingError(f"Failed to merge strategies: {e}")

    def _merge_risk_configs(self, risk_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge risk configurations, choosing most conservative options.

        Args:
            risk_configs: List of risk configuration dictionaries

        Returns:
            Merged risk configuration
        """
        if not risk_configs:
            return {}

        # Use first as base
        merged_risk = risk_configs[0].copy()

        # Choose most conservative values
        for config in risk_configs[1:]:
            # Smallest position size
            if 'position_sizing' in config:
                current_size = merged_risk.get('position_sizing', {}).get('value', float('inf'))
                new_size = config['position_sizing'].get('value', float('inf'))
                if new_size < current_size:
                    merged_risk['position_sizing'] = config['position_sizing']

            # Tightest stop loss
            if 'stop_loss' in config:
                current_sl = merged_risk.get('stop_loss', {}).get('pips', float('inf'))
                new_sl = config['stop_loss'].get('pips', float('inf'))
                if new_sl < current_sl:
                    merged_risk['stop_loss'] = config['stop_loss']

        return merged_risk


# Utility functions for common operations
def load_and_validate_strategy(file_path: Union[str, Path]) -> tuple[Dict[str, Any], ValidationResult]:
    """
    Load and validate a strategy file in one operation.

    Args:
        file_path: Path to strategy JSON file

    Returns:
        Tuple of (strategy_dict, validation_result)

    Raises:
        JSONProcessingError: If loading fails
    """
    processor = JSONProcessor()
    strategy = processor.load_strategy_from_file(file_path)
    validation = processor.validate_strategy(strategy)
    return strategy, validation


def create_strategy_from_template(template_name: str = "multi_timeframe") -> Dict[str, Any]:
    """
    Create a strategy from a built-in template.

    Args:
        template_name: Name of template ('multi_timeframe' or 'single_timeframe')

    Returns:
        Strategy configuration dictionary

    Raises:
        ValueError: If template name is not recognized
    """
    templates = {
        'multi_timeframe': get_example_strategy,
        'single_timeframe': get_single_timeframe_example
    }

    if template_name not in templates:
        available = ', '.join(templates.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

    return templates[template_name]()