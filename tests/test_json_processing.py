"""
Test suite for JSON processing foundation.

Tests the JSON schema validation, processing utilities, conversion layer,
and error handling for multi-timeframe strategy configurations.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
import json
import tempfile
from typing import Dict, Any
import logging

from core.json_schemas import (
    IndicatorType,
    TimeframeType,
    SignalDirection,
    ExecutionMode,
    get_example_strategy,
    get_single_timeframe_example
)
from core.json_processor import (
    JSONProcessor,
    ValidationResult,
    JSONProcessingError,
    ValidationError,
    ConversionError
)
from core.json_converter import (
    JSONToObjectConverter,
    MultiTimeframeStrategyConfig,
    convert_json_file_to_strategy,
    create_example_multi_tf_config
)
from core.json_validation import (
    AdvancedJSONValidator,
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    ExtendedValidationResult,
    validate_strategy_file,
    auto_fix_strategy_file
)


# Suppress logging during tests
logging.getLogger('core.json_processor').setLevel(logging.CRITICAL)
logging.getLogger('core.json_converter').setLevel(logging.CRITICAL)
logging.getLogger('core.json_validation').setLevel(logging.CRITICAL)


class TestJSONSchemas(unittest.TestCase):
    """Test suite for JSON schema definitions."""

    def test_example_strategy_structure(self):
        """Test that example strategy has correct structure."""
        example = get_example_strategy()

        # Check required fields
        self.assertIn('name', example)
        self.assertIn('version', example)
        self.assertIn('indicators', example)
        self.assertIn('risk', example)
        self.assertIn('execution', example)

        # Check indicators structure
        indicators = example['indicators']
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)

        for indicator in indicators:
            self.assertIn('type', indicator)
            self.assertIn('timeframe', indicator)
            self.assertIn('params', indicator)

    def test_single_timeframe_example(self):
        """Test single timeframe example structure."""
        example = get_single_timeframe_example()

        self.assertIn('name', example)
        self.assertIn('indicators', example)

        indicators = example['indicators']
        self.assertEqual(len(indicators), 1)

        # Should use single timeframe
        timeframes = [ind.get('timeframe') for ind in indicators]
        unique_timeframes = set(timeframes)
        self.assertEqual(len(unique_timeframes), 1)

    def test_enum_values(self):
        """Test that enum values are properly defined."""
        # Test IndicatorType
        expected_indicators = [
            "liquidity_grab_detector",
            "choch_detector",
            "bos_detector",
            "fvg_detector",
            "order_block_detector"
        ]
        for indicator in expected_indicators:
            self.assertIn(indicator, [t.value for t in IndicatorType])

        # Test TimeframeType
        expected_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
        for timeframe in expected_timeframes:
            self.assertIn(timeframe, [t.value for t in TimeframeType])


class TestJSONProcessor(unittest.TestCase):
    """Test suite for JSON processor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = JSONProcessor()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_valid_strategy_file(self):
        """Test loading a valid strategy JSON file."""
        example = get_example_strategy()
        file_path = Path(self.temp_dir) / "test_strategy.json"

        # Save example to file
        with open(file_path, 'w') as f:
            json.dump(example, f, indent=2)

        # Load it back
        loaded_strategy = self.processor.load_strategy_from_file(file_path)
        self.assertEqual(loaded_strategy['name'], example['name'])
        self.assertEqual(loaded_strategy['version'], example['version'])

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises appropriate error."""
        with self.assertRaises(JSONProcessingError):
            self.processor.load_strategy_from_file("nonexistent.json")

    def test_load_invalid_json_file(self):
        """Test loading invalid JSON raises appropriate error."""
        file_path = Path(self.temp_dir) / "invalid.json"
        with open(file_path, 'w') as f:
            f.write("{invalid json content")

        with self.assertRaises(JSONProcessingError):
            self.processor.load_strategy_from_file(file_path)

    def test_load_non_json_file(self):
        """Test loading non-JSON file raises appropriate error."""
        file_path = Path(self.temp_dir) / "test.txt"
        with open(file_path, 'w') as f:
            f.write("This is not a JSON file")

        with self.assertRaises(JSONProcessingError):
            self.processor.load_strategy_from_file(file_path)

    def test_save_strategy_to_file(self):
        """Test saving strategy to file."""
        example = get_example_strategy()
        file_path = Path(self.temp_dir) / "saved_strategy.json"

        self.processor.save_strategy_to_file(example, file_path)

        # Verify file exists and has correct content
        self.assertTrue(file_path.exists())
        loaded_strategy = self.processor.load_strategy_from_file(file_path)
        self.assertEqual(loaded_strategy['name'], example['name'])

    def test_validate_valid_strategy(self):
        """Test validation of valid strategy."""
        example = get_example_strategy()
        validation = self.processor.validate_strategy(example)

        self.assertIsInstance(validation, ValidationResult)
        self.assertTrue(validation.is_valid)
        self.assertEqual(len(validation.errors), 0)

    def test_validate_invalid_strategy(self):
        """Test validation of invalid strategy."""
        invalid_strategy = {
            "name": "Test",
            # Missing required fields
        }

        validation = self.processor.validate_strategy(invalid_strategy)
        self.assertFalse(validation.is_valid)
        self.assertGreater(len(validation.errors), 0)

    def test_convert_to_legacy_format(self):
        """Test conversion to legacy format."""
        example = get_example_strategy()
        legacy_config = self.processor.convert_to_legacy_format(example)

        self.assertIn('name', legacy_config)
        self.assertIn('indicator_sequence', legacy_config)
        self.assertIn('indicator_configs', legacy_config)
        self.assertIn('stop_loss_pips', legacy_config)
        self.assertTrue(legacy_config['json_source'])

    def test_create_example_files(self):
        """Test creating example files."""
        created_files = self.processor.create_example_files(self.temp_dir)

        self.assertEqual(len(created_files), 2)
        for file_path in created_files:
            self.assertTrue(file_path.exists())
            self.assertTrue(file_path.name.endswith('.json'))

    def test_get_schema_info(self):
        """Test getting schema information."""
        schema_info = self.processor.get_schema_info()

        self.assertIn('version', schema_info)
        self.assertIn('supported_timeframes', schema_info)
        self.assertIn('supported_indicators', schema_info)
        self.assertIn('execution_modes', schema_info)

    def test_merge_strategies(self):
        """Test merging multiple strategies."""
        strategy1 = get_example_strategy()
        strategy2 = get_single_timeframe_example()

        merged = self.processor.merge_strategies([strategy1, strategy2], "Merged Test")

        self.assertEqual(merged['name'], "Merged Test")
        self.assertIn('indicators', merged)

        # Should have indicators from both strategies
        merged_indicators = merged['indicators']
        self.assertGreater(len(merged_indicators), 1)


class TestJSONConverter(unittest.TestCase):
    """Test suite for JSON to object conversion."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = JSONToObjectConverter()

    def test_convert_json_to_strategy(self):
        """Test converting JSON to MultiTimeframeStrategyConfig."""
        example = get_example_strategy()
        strategy_config = self.converter.convert_json_to_strategy(example)

        self.assertIsInstance(strategy_config, MultiTimeframeStrategyConfig)
        self.assertEqual(strategy_config.name, example['name'])
        self.assertEqual(strategy_config.version, example['version'])
        self.assertTrue(strategy_config.json_source)

    def test_convert_invalid_json(self):
        """Test converting invalid JSON raises error."""
        invalid_json = {"name": "Test"}  # Missing required fields

        with self.assertRaises(ConversionError):
            self.converter.convert_json_to_strategy(invalid_json)

    def test_multi_timeframe_detection(self):
        """Test multi-timeframe detection."""
        example = get_example_strategy()
        strategy_config = self.converter.convert_json_to_strategy(example)

        self.assertTrue(strategy_config.is_multi_timeframe())

        timeframes = strategy_config.get_all_timeframes()
        self.assertGreater(len(timeframes), 1)

    def test_single_timeframe_conversion(self):
        """Test single timeframe strategy conversion."""
        example = get_single_timeframe_example()
        strategy_config = self.converter.convert_json_to_strategy(example)

        self.assertFalse(strategy_config.is_multi_timeframe())

        timeframes = strategy_config.get_all_timeframes()
        self.assertEqual(len(timeframes), 1)

    def test_to_legacy_config(self):
        """Test conversion to legacy StrategyConfig."""
        example = get_example_strategy()
        multi_tf_config = self.converter.convert_json_to_strategy(example)
        legacy_config = multi_tf_config.to_legacy_config()

        from core.state_types import StrategyConfig
        self.assertIsInstance(legacy_config, StrategyConfig)
        self.assertIsInstance(legacy_config.indicator_sequence, list)
        self.assertGreater(legacy_config.stop_loss_pips, 0)

    def test_get_indicators_for_timeframe(self):
        """Test getting indicators for specific timeframe."""
        example = get_example_strategy()
        strategy_config = self.converter.convert_json_to_strategy(example)

        # Get indicators for each timeframe
        all_timeframes = strategy_config.get_all_timeframes()
        for timeframe in all_timeframes:
            indicators = strategy_config.get_indicators_for_timeframe(timeframe)
            self.assertIsInstance(indicators, list)

    def test_create_from_legacy(self):
        """Test creating MultiTimeframeStrategyConfig from legacy StrategyConfig."""
        from core.state_types import StrategyConfig

        legacy_config = StrategyConfig(
            indicator_sequence=['liquidity_grab_detector', 'choch_detector'],
            required_confirmations=2,
            timeouts={'default': 30},
            stop_loss_pips=25,
            take_profit_pips=50,
            risk_percent=0.01
        )

        multi_tf_config = self.converter.create_from_legacy(legacy_config, "Legacy Test")

        self.assertEqual(multi_tf_config.name, "Legacy Test")
        self.assertFalse(multi_tf_config.json_source)
        self.assertEqual(len(multi_tf_config.indicator_sequence), 2)

    def test_batch_convert_strategies(self):
        """Test batch conversion of multiple strategies."""
        strategies = [
            get_example_strategy(),
            get_single_timeframe_example()
        ]

        converted = self.converter.batch_convert_strategies(strategies)

        self.assertEqual(len(converted), 2)
        for config in converted:
            self.assertIsInstance(config, MultiTimeframeStrategyConfig)


class TestAdvancedJSONValidator(unittest.TestCase):
    """Test suite for advanced JSON validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = AdvancedJSONValidator()

    def test_validate_valid_strategy(self):
        """Test validation of valid strategy."""
        example = get_example_strategy()
        result = self.validator.validate_strategy_comprehensive(example)

        self.assertIsInstance(result, ExtendedValidationResult)
        self.assertTrue(result.is_valid)
        self.assertGreater(result.validation_time, 0)

    def test_validate_invalid_strategy(self):
        """Test validation of invalid strategy."""
        invalid_strategy = {
            "name": "Invalid",
            # Missing required fields
        }

        result = self.validator.validate_strategy_comprehensive(invalid_strategy)
        self.assertFalse(result.is_valid)

        errors = result.get_errors()
        self.assertGreater(len(errors), 0)

    def test_validation_issue_categories(self):
        """Test validation issue categorization."""
        # Create strategy with various issues
        problematic_strategy = get_example_strategy()
        problematic_strategy['risk']['position_sizing']['value'] = 0.5  # Too high
        del problematic_strategy['version']  # Missing version

        result = self.validator.validate_strategy_comprehensive(problematic_strategy)

        # Check that different categories are represented
        issues_by_category = {}
        for issue in result.issues:
            category = issue.category
            if category not in issues_by_category:
                issues_by_category[category] = []
            issues_by_category[category].append(issue)

        # Should have at least some categories
        self.assertGreater(len(issues_by_category), 0)

    def test_auto_fix_functionality(self):
        """Test automatic fixing of issues."""
        # Create strategy with auto-fixable issues
        fixable_strategy = get_single_timeframe_example()
        del fixable_strategy['version']  # Missing version (auto-fixable)

        fixed_strategy, applied_fixes = self.validator.auto_fix_strategy(fixable_strategy)

        self.assertGreater(len(applied_fixes), 0)
        self.assertIn('version', fixed_strategy)

    def test_risk_reward_validation(self):
        """Test risk-reward ratio validation."""
        strategy = get_example_strategy()
        strategy['risk']['stop_loss']['pips'] = 50
        strategy['risk']['take_profit']['pips'] = 10  # Poor RR ratio

        result = self.validator.validate_strategy_comprehensive(strategy)

        warnings = result.get_warnings()
        rr_warnings = [w for w in warnings if 'risk-reward' in w.message.lower()]
        self.assertGreater(len(rr_warnings), 0)

    def test_timeframe_logic_validation(self):
        """Test timeframe logic validation."""
        strategy = get_example_strategy()
        # Set primary timeframe to one not used by any indicator
        strategy['execution']['primary_timeframe'] = 'W1'

        result = self.validator.validate_strategy_comprehensive(strategy)

        errors = result.get_errors()
        tf_errors = [e for e in errors if 'primary_timeframe' in e.message.lower() or 'timeframe' in e.path]
        self.assertGreater(len(tf_errors), 0)

    def test_performance_analysis(self):
        """Test performance implication analysis."""
        strategy = get_example_strategy()

        # Add many indicators across many timeframes
        strategy['indicators'].extend([
            {
                "type": "order_block_detector",
                "timeframe": "M1",
                "params": {}
            },
            {
                "type": "fvg_detector",
                "timeframe": "M5",
                "params": {}
            },
            {
                "type": "bos_detector",
                "timeframe": "D1",
                "params": {}
            }
        ])

        result = self.validator.validate_strategy_comprehensive(strategy)

        performance_issues = result.get_by_category(ValidationCategory.PERFORMANCE)
        # Should have some performance warnings
        self.assertGreaterEqual(len(performance_issues), 0)


class TestJSONValidationUtilities(unittest.TestCase):
    """Test suite for JSON validation utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_validate_strategy_file(self):
        """Test validating strategy from file."""
        example = get_example_strategy()
        file_path = Path(self.temp_dir) / "test_strategy.json"

        # Save example to file
        with open(file_path, 'w') as f:
            json.dump(example, f, indent=2)

        result = validate_strategy_file(file_path)
        self.assertIsInstance(result, ExtendedValidationResult)
        self.assertTrue(result.is_valid)

    def test_auto_fix_strategy_file(self):
        """Test auto-fixing strategy file."""
        # Create strategy with fixable issues
        strategy = get_single_timeframe_example()
        del strategy['version']

        file_path = Path(self.temp_dir) / "fixable_strategy.json"
        with open(file_path, 'w') as f:
            json.dump(strategy, f, indent=2)

        applied_fixes, final_validation = auto_fix_strategy_file(file_path)

        self.assertGreater(len(applied_fixes), 0)
        self.assertIsInstance(final_validation, ExtendedValidationResult)

        # Check that backup was created
        backup_path = file_path.with_suffix('.bak.json')
        self.assertTrue(backup_path.exists())


class TestIntegration(unittest.TestCase):
    """Integration tests for JSON processing components."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = JSONProcessor()
        self.converter = JSONToObjectConverter()
        self.validator = AdvancedJSONValidator()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_json_workflow(self):
        """Test complete JSON workflow: create -> save -> load -> validate -> convert."""
        # 1. Create strategy
        strategy = get_example_strategy()

        # 2. Save to file
        file_path = Path(self.temp_dir) / "workflow_test.json"
        self.processor.save_strategy_to_file(strategy, file_path)

        # 3. Load from file
        loaded_strategy = self.processor.load_strategy_from_file(file_path)

        # 4. Validate
        validation = self.validator.validate_strategy_comprehensive(loaded_strategy)
        self.assertTrue(validation.is_valid)

        # 5. Convert to object
        strategy_config = self.converter.convert_json_to_strategy(loaded_strategy)
        self.assertIsInstance(strategy_config, MultiTimeframeStrategyConfig)

        # 6. Convert to legacy
        legacy_config = strategy_config.to_legacy_config()
        from core.state_types import StrategyConfig
        self.assertIsInstance(legacy_config, StrategyConfig)

    def test_error_propagation(self):
        """Test that errors propagate correctly through the system."""
        # Create invalid strategy
        invalid_strategy = {"name": "Invalid"}

        # Validation should fail
        validation = self.validator.validate_strategy_comprehensive(invalid_strategy)
        self.assertFalse(validation.is_valid)

        # Conversion should fail
        with self.assertRaises(ConversionError):
            self.converter.convert_json_to_strategy(invalid_strategy)

    def test_utility_function_integration(self):
        """Test utility functions work with full workflow."""
        # Create strategy file
        example = get_example_strategy()
        file_path = Path(self.temp_dir) / "utility_test.json"

        with open(file_path, 'w') as f:
            json.dump(example, f, indent=2)

        # Test utility function
        strategy_config = convert_json_file_to_strategy(file_path)
        self.assertIsInstance(strategy_config, MultiTimeframeStrategyConfig)

    def test_example_config_creation(self):
        """Test example configuration creation utility."""
        example_config = create_example_multi_tf_config()
        self.assertIsInstance(example_config, MultiTimeframeStrategyConfig)
        self.assertTrue(example_config.json_source)
        self.assertTrue(example_config.is_multi_timeframe())


if __name__ == '__main__':
    unittest.main()