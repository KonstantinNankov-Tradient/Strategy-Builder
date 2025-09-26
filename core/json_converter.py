"""
JSON to object conversion layer for strategy configurations.

This module converts JSON strategy configurations into internal objects
compatible with the existing state machine and indicator system.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

from .state_types import (
    StrategyConfig,
    SignalDirection as StateSignalDirection
)
from .json_schemas import (
    IndicatorConfig,
    RiskConfig,
    TimingConfig,
    IndicatorType,
    TimeframeType,
    SignalDirection,
    ExecutionMode
)
from .json_processor import JSONProcessor, ConversionError


logger = logging.getLogger(__name__)


@dataclass
class MultiTimeframeStrategyConfig:
    """
    Multi-timeframe strategy configuration that extends the base StrategyConfig.

    This configuration supports multiple timeframes and provides JSON conversion
    capabilities while maintaining compatibility with existing systems.
    """
    name: str
    version: str
    description: str = ""

    # Multi-timeframe indicator configurations
    indicator_configs: Dict[str, Dict[str, Any]] = None  # indicator_key -> config
    timeframe_mapping: Dict[str, str] = None             # indicator_key -> timeframe

    # Execution configuration
    primary_timeframe: str = "H1"
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    required_confirmations: int = 1
    direction_filter: SignalDirection = SignalDirection.BOTH

    # Legacy compatibility fields
    indicator_sequence: List[str] = None
    timeouts: Dict[str, int] = None
    stop_loss_pips: float = 25.0
    take_profit_pips: float = 50.0
    risk_percent: float = 0.01

    # Risk management
    max_drawdown: Optional[float] = None
    max_concurrent_trades: int = 1
    position_timeout: Optional[int] = None

    # Session configuration
    trading_sessions: Optional[List[Dict[str, Any]]] = None

    # Metadata
    json_source: bool = False
    created_at: Optional[str] = None
    tags: List[str] = None

    def to_legacy_config(self) -> StrategyConfig:
        """
        Convert to legacy StrategyConfig for backward compatibility.

        Returns:
            StrategyConfig object compatible with existing state machine
        """
        return StrategyConfig(
            indicator_sequence=self.indicator_sequence or [],
            required_confirmations=self.required_confirmations,
            timeouts=self.timeouts or {'default': 30},
            stop_loss_pips=self.stop_loss_pips,
            take_profit_pips=self.take_profit_pips,
            risk_percent=self.risk_percent
        )

    def get_indicators_for_timeframe(self, timeframe: str) -> List[str]:
        """
        Get list of indicators configured for a specific timeframe.

        Args:
            timeframe: Timeframe to filter by (e.g., 'H1', 'H4')

        Returns:
            List of indicator keys for the given timeframe
        """
        if not self.timeframe_mapping:
            return []

        return [
            indicator_key for indicator_key, tf in self.timeframe_mapping.items()
            if tf == timeframe
        ]

    def get_all_timeframes(self) -> List[str]:
        """
        Get all timeframes used by this strategy.

        Returns:
            List of unique timeframes used
        """
        if not self.timeframe_mapping:
            return [self.primary_timeframe]

        timeframes = list(set(self.timeframe_mapping.values()))

        # Ensure primary timeframe is first
        if self.primary_timeframe in timeframes:
            timeframes.remove(self.primary_timeframe)
            timeframes.insert(0, self.primary_timeframe)

        return timeframes

    def is_multi_timeframe(self) -> bool:
        """
        Check if this strategy uses multiple timeframes.

        Returns:
            True if strategy uses more than one timeframe
        """
        return len(self.get_all_timeframes()) > 1


class JSONToObjectConverter:
    """
    Converter that transforms JSON strategy configurations into internal objects.

    Handles the conversion from JSON format to MultiTimeframeStrategyConfig
    and legacy StrategyConfig objects.
    """

    def __init__(self):
        """Initialize the converter."""
        self.json_processor = JSONProcessor()

    def convert_json_to_strategy(self, json_config: Dict[str, Any]) -> MultiTimeframeStrategyConfig:
        """
        Convert JSON strategy configuration to MultiTimeframeStrategyConfig.

        Args:
            json_config: JSON strategy configuration dictionary

        Returns:
            MultiTimeframeStrategyConfig object

        Raises:
            ConversionError: If conversion fails
        """
        try:
            # Validate JSON first
            validation = self.json_processor.validate_strategy(json_config)
            if not validation.is_valid:
                raise ConversionError(f"Invalid JSON configuration: {validation.errors}")

            # Extract basic information
            name = json_config.get('name', 'Unnamed Strategy')
            version = json_config.get('version', '1.0.0')
            description = json_config.get('description', '')

            # Process indicators
            indicators_data = json_config.get('indicators', [])
            indicator_configs, timeframe_mapping, indicator_sequence = self._process_indicators(indicators_data)

            # Process risk configuration
            risk_data = json_config.get('risk', {})
            risk_config = self._process_risk_config(risk_data)

            # Process timing configuration
            timing_data = json_config.get('timing', {})
            timing_config = self._process_timing_config(timing_data)

            # Process execution configuration
            execution_data = json_config.get('execution', {})
            execution_config = self._process_execution_config(execution_data)

            # Process metadata
            metadata = json_config.get('metadata', {})

            # Create the configuration object
            strategy_config = MultiTimeframeStrategyConfig(
                name=name,
                version=version,
                description=description,
                indicator_configs=indicator_configs,
                timeframe_mapping=timeframe_mapping,
                indicator_sequence=indicator_sequence,
                primary_timeframe=execution_config['primary_timeframe'],
                execution_mode=execution_config['execution_mode'],
                required_confirmations=execution_config['required_confirmations'],
                direction_filter=execution_config['direction_filter'],
                timeouts=timing_config['timeouts'],
                stop_loss_pips=risk_config['stop_loss_pips'],
                take_profit_pips=risk_config['take_profit_pips'],
                risk_percent=risk_config['risk_percent'],
                max_drawdown=risk_config.get('max_drawdown'),
                max_concurrent_trades=risk_config.get('max_concurrent_trades', 1),
                position_timeout=timing_config.get('position_timeout'),
                trading_sessions=timing_config.get('trading_sessions'),
                json_source=True,
                created_at=metadata.get('created_at'),
                tags=metadata.get('tags', [])
            )

            logger.info(f"Successfully converted JSON strategy '{name}' to MultiTimeframeStrategyConfig")
            return strategy_config

        except Exception as e:
            raise ConversionError(f"Failed to convert JSON to strategy config: {e}")

    def _process_indicators(self, indicators_data: List[Dict[str, Any]]) -> tuple[Dict, Dict, List]:
        """
        Process indicators from JSON configuration.

        Args:
            indicators_data: List of indicator configurations from JSON

        Returns:
            Tuple of (indicator_configs, timeframe_mapping, indicator_sequence)
        """
        indicator_configs = {}
        timeframe_mapping = {}
        indicator_sequence = []

        for i, indicator in enumerate(indicators_data):
            ind_type = indicator.get('type')
            timeframe = indicator.get('timeframe')
            params = indicator.get('params', {})
            required = indicator.get('required', True)

            # Create unique indicator key for multi-timeframe support
            if len(indicators_data) > 1:
                # Multi-indicator: use type_timeframe format
                indicator_key = f"{ind_type}_{timeframe}"
            else:
                # Single indicator: use just the type
                indicator_key = ind_type

            # Store configurations
            indicator_configs[indicator_key] = params
            timeframe_mapping[indicator_key] = timeframe

            if required:
                indicator_sequence.append(indicator_key)

        return indicator_configs, timeframe_mapping, indicator_sequence

    def _process_risk_config(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process risk configuration from JSON.

        Args:
            risk_data: Risk configuration dictionary from JSON

        Returns:
            Processed risk configuration
        """
        risk_config = {}

        # Position sizing
        position_sizing = risk_data.get('position_sizing', {})
        risk_config['risk_percent'] = position_sizing.get('value', 0.01)

        # Stop loss
        stop_loss = risk_data.get('stop_loss', {})
        risk_config['stop_loss_pips'] = stop_loss.get('pips', 25.0)

        # Take profit
        take_profit = risk_data.get('take_profit', {})
        risk_config['take_profit_pips'] = take_profit.get('pips', 50.0)

        # Additional risk parameters
        risk_config['max_drawdown'] = risk_data.get('max_drawdown')
        risk_config['max_concurrent_trades'] = risk_data.get('max_concurrent_trades', 1)

        return risk_config

    def _process_timing_config(self, timing_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process timing configuration from JSON.

        Args:
            timing_data: Timing configuration dictionary from JSON

        Returns:
            Processed timing configuration
        """
        timing_config = {}

        setup_timeout = timing_data.get('setup_timeout', 30)
        signal_timeout = timing_data.get('signal_timeout', 20)

        # Create timeouts dictionary
        timing_config['timeouts'] = {
            'default': setup_timeout,
            # Individual timeouts can be added here based on indicators
        }

        timing_config['position_timeout'] = timing_data.get('position_timeout')
        timing_config['trading_sessions'] = timing_data.get('trading_sessions')

        return timing_config

    def _process_execution_config(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process execution configuration from JSON.

        Args:
            execution_data: Execution configuration dictionary from JSON

        Returns:
            Processed execution configuration
        """
        execution_config = {}

        execution_config['primary_timeframe'] = execution_data.get('primary_timeframe', 'H1')

        mode_str = execution_data.get('mode', 'sequential')
        execution_config['execution_mode'] = ExecutionMode(mode_str)

        execution_config['required_confirmations'] = execution_data.get('confirmation_required', 1)

        direction_str = execution_data.get('direction_filter', 'both')
        execution_config['direction_filter'] = SignalDirection(direction_str)

        return execution_config

    def convert_to_legacy_format(self, multi_tf_config: MultiTimeframeStrategyConfig) -> StrategyConfig:
        """
        Convert MultiTimeframeStrategyConfig to legacy StrategyConfig.

        Args:
            multi_tf_config: Multi-timeframe strategy configuration

        Returns:
            Legacy StrategyConfig object
        """
        return multi_tf_config.to_legacy_config()

    def create_from_legacy(self, legacy_config: StrategyConfig,
                          name: str = "Legacy Strategy",
                          primary_timeframe: str = "H1") -> MultiTimeframeStrategyConfig:
        """
        Create MultiTimeframeStrategyConfig from legacy StrategyConfig.

        Args:
            legacy_config: Legacy strategy configuration
            name: Name for the new configuration
            primary_timeframe: Primary timeframe to use

        Returns:
            MultiTimeframeStrategyConfig object
        """
        # Create indicator configs and timeframe mapping
        indicator_configs = {}
        timeframe_mapping = {}

        for indicator_name in legacy_config.indicator_sequence:
            # Use primary timeframe for all indicators in legacy mode
            indicator_configs[indicator_name] = {}  # Empty config, will be filled later
            timeframe_mapping[indicator_name] = primary_timeframe

        return MultiTimeframeStrategyConfig(
            name=name,
            version="1.0.0",
            description=f"Converted from legacy configuration",
            indicator_configs=indicator_configs,
            timeframe_mapping=timeframe_mapping,
            indicator_sequence=legacy_config.indicator_sequence,
            primary_timeframe=primary_timeframe,
            execution_mode=ExecutionMode.SEQUENTIAL,
            required_confirmations=legacy_config.required_confirmations,
            direction_filter=SignalDirection.BOTH,
            timeouts=legacy_config.timeouts,
            stop_loss_pips=legacy_config.stop_loss_pips,
            take_profit_pips=legacy_config.take_profit_pips,
            risk_percent=legacy_config.risk_percent,
            max_concurrent_trades=1,
            json_source=False
        )

    def batch_convert_strategies(self, json_configs: List[Dict[str, Any]]) -> List[MultiTimeframeStrategyConfig]:
        """
        Convert multiple JSON strategies in batch.

        Args:
            json_configs: List of JSON strategy configurations

        Returns:
            List of MultiTimeframeStrategyConfig objects

        Raises:
            ConversionError: If any conversion fails
        """
        converted_strategies = []
        failed_conversions = []

        for i, json_config in enumerate(json_configs):
            try:
                strategy = self.convert_json_to_strategy(json_config)
                converted_strategies.append(strategy)
            except Exception as e:
                failed_conversions.append((i, str(e)))

        if failed_conversions:
            error_msg = "Failed conversions: " + ", ".join([
                f"Index {i}: {error}" for i, error in failed_conversions
            ])
            raise ConversionError(error_msg)

        return converted_strategies


# Utility functions for easy access
def convert_json_file_to_strategy(file_path: str) -> MultiTimeframeStrategyConfig:
    """
    Convert a JSON strategy file directly to MultiTimeframeStrategyConfig.

    Args:
        file_path: Path to JSON strategy file

    Returns:
        MultiTimeframeStrategyConfig object

    Raises:
        ConversionError: If loading or conversion fails
    """
    processor = JSONProcessor()
    converter = JSONToObjectConverter()

    json_config = processor.load_strategy_from_file(file_path)
    return converter.convert_json_to_strategy(json_config)


def create_example_multi_tf_config() -> MultiTimeframeStrategyConfig:
    """
    Create an example multi-timeframe strategy configuration.

    Returns:
        Example MultiTimeframeStrategyConfig object
    """
    from .json_schemas import get_example_strategy

    converter = JSONToObjectConverter()
    example_json = get_example_strategy()

    return converter.convert_json_to_strategy(example_json)