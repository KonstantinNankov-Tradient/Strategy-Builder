"""
Extended JSON validation and error handling for strategy configurations.

This module provides comprehensive validation, error reporting, and
recovery mechanisms for JSON strategy configurations.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import traceback
from pathlib import Path

from .json_processor import ValidationResult, JSONProcessingError
from .json_schemas import (
    IndicatorType,
    TimeframeType,
    SignalDirection,
    ExecutionMode,
    STRATEGY_SCHEMA
)


logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Blocks execution
    WARNING = "warning"  # Should be addressed but doesn't block
    INFO = "info"        # Informational only


class ValidationCategory(str, Enum):
    """Categories of validation issues."""
    SCHEMA = "schema"                    # JSON schema violations
    BUSINESS_LOGIC = "business_logic"    # Business rule violations
    COMPATIBILITY = "compatibility"     # Backward compatibility issues
    PERFORMANCE = "performance"         # Performance implications
    SECURITY = "security"              # Security concerns


@dataclass
class ValidationIssue:
    """Detailed information about a validation issue."""
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    path: str = ""                      # JSON path where issue occurred
    suggestion: Optional[str] = None    # Suggested fix
    auto_fixable: bool = False         # Can be automatically fixed
    error_code: Optional[str] = None   # Unique error identifier


@dataclass
class ExtendedValidationResult:
    """Extended validation result with detailed issue reporting."""
    is_valid: bool
    is_warning_free: bool
    issues: List[ValidationIssue]
    schema_version: str
    validation_time: float
    auto_fixes_available: bool = False

    def get_errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]

    def get_info(self) -> List[ValidationIssue]:
        """Get only info-level issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.INFO]

    def get_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        """Get issues by category."""
        return [issue for issue in self.issues if issue.category == category]

    def has_auto_fixes(self) -> bool:
        """Check if any issues can be automatically fixed."""
        return any(issue.auto_fixable for issue in self.issues)


class AdvancedJSONValidator:
    """
    Advanced JSON validator with comprehensive error reporting and auto-fix capabilities.

    Provides detailed validation beyond basic schema checking, including business
    logic validation, performance analysis, and compatibility checking.
    """

    def __init__(self):
        """Initialize the advanced validator."""
        self.validation_rules = self._initialize_validation_rules()

    def validate_strategy_comprehensive(self, strategy: Dict[str, Any]) -> ExtendedValidationResult:
        """
        Perform comprehensive validation of strategy configuration.

        Args:
            strategy: Strategy configuration to validate

        Returns:
            ExtendedValidationResult with detailed issue reporting
        """
        import time

        start_time = time.time()
        issues = []

        try:
            # Schema validation
            issues.extend(self._validate_schema(strategy))

            # Business logic validation
            issues.extend(self._validate_business_logic(strategy))

            # Performance analysis
            issues.extend(self._analyze_performance_implications(strategy))

            # Compatibility checking
            issues.extend(self._check_compatibility(strategy))

            # Security validation
            issues.extend(self._validate_security(strategy))

            # Additional validations
            issues.extend(self._validate_indicator_combinations(strategy))
            issues.extend(self._validate_timeframe_logic(strategy))
            issues.extend(self._validate_risk_parameters(strategy))

        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SCHEMA,
                message=f"Validation error: {str(e)}",
                error_code="VALIDATION_EXCEPTION"
            ))

        validation_time = time.time() - start_time
        errors = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        warnings = [issue for issue in issues if issue.severity == ValidationSeverity.WARNING]

        return ExtendedValidationResult(
            is_valid=len(errors) == 0,
            is_warning_free=len(warnings) == 0,
            issues=issues,
            schema_version="1.0.0",
            validation_time=validation_time,
            auto_fixes_available=any(issue.auto_fixable for issue in issues)
        )

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules and patterns."""
        return {
            'max_indicators': 10,
            'max_timeframes': 5,
            'min_risk_reward_ratio': 0.5,
            'max_position_size': 0.1,
            'min_timeout': 5,
            'max_timeout': 1000,
            'supported_indicator_combinations': {
                # Define which indicators work well together
                'liquidity_grab_detector': ['choch_detector', 'bos_detector'],
                'choch_detector': ['liquidity_grab_detector', 'fvg_detector'],
                'fvg_detector': ['order_block_detector', 'choch_detector']
            }
        }

    def _validate_schema(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate against JSON schema."""
        issues = []

        try:
            import jsonschema
            validator = jsonschema.Draft7Validator(STRATEGY_SCHEMA)

            for error in validator.iter_errors(strategy):
                path = " -> ".join(str(p) for p in error.path)
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.SCHEMA,
                    message=error.message,
                    path=path,
                    error_code="SCHEMA_VIOLATION",
                    suggestion=self._suggest_schema_fix(error)
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SCHEMA,
                message=f"Schema validation failed: {str(e)}",
                error_code="SCHEMA_ERROR"
            ))

        return issues

    def _validate_business_logic(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate business logic rules."""
        issues = []

        indicators = strategy.get('indicators', [])
        risk = strategy.get('risk', {})
        execution = strategy.get('execution', {})

        # Check indicator count
        if len(indicators) == 0:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.BUSINESS_LOGIC,
                message="Strategy must have at least one indicator",
                error_code="NO_INDICATORS",
                auto_fixable=True,
                suggestion="Add at least one indicator to the strategy"
            ))

        if len(indicators) > self.validation_rules['max_indicators']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.BUSINESS_LOGIC,
                message=f"Strategy has {len(indicators)} indicators (recommended max: {self.validation_rules['max_indicators']})",
                error_code="TOO_MANY_INDICATORS",
                suggestion="Consider reducing the number of indicators for better performance"
            ))

        # Validate risk-reward ratio
        try:
            sl_pips = risk.get('stop_loss', {}).get('pips', 0)
            tp_pips = risk.get('take_profit', {}).get('pips', 0)

            if sl_pips > 0 and tp_pips > 0:
                rr_ratio = tp_pips / sl_pips
                if rr_ratio < self.validation_rules['min_risk_reward_ratio']:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.BUSINESS_LOGIC,
                        message=f"Risk-reward ratio is {rr_ratio:.2f} (recommended min: {self.validation_rules['min_risk_reward_ratio']})",
                        path="risk",
                        error_code="LOW_RISK_REWARD",
                        suggestion=f"Consider increasing take profit to at least {sl_pips * self.validation_rules['min_risk_reward_ratio']:.1f} pips"
                    ))
        except (TypeError, ZeroDivisionError):
            pass  # Will be caught by schema validation

        # Validate execution mode vs indicator count
        mode = execution.get('mode', 'sequential')
        if mode == 'sequential' and len(indicators) == 1:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.BUSINESS_LOGIC,
                message="Sequential mode with single indicator - parallel mode might be more appropriate",
                path="execution.mode",
                error_code="UNNECESSARY_SEQUENTIAL",
                auto_fixable=True,
                suggestion="Change execution mode to 'parallel' for single indicator strategies"
            ))

        return issues

    def _analyze_performance_implications(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Analyze performance implications of the configuration."""
        issues = []

        indicators = strategy.get('indicators', [])
        timeframes = list(set(ind.get('timeframe') for ind in indicators))

        # Check timeframe diversity
        if len(timeframes) > self.validation_rules['max_timeframes']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Strategy uses {len(timeframes)} timeframes (recommended max: {self.validation_rules['max_timeframes']})",
                error_code="MANY_TIMEFRAMES",
                suggestion="Consider consolidating indicators to fewer timeframes for better performance"
            ))

        # Check for computationally expensive combinations
        indicator_types = [ind.get('type') for ind in indicators]
        if 'order_block_detector' in indicator_types and len(indicators) > 3:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.PERFORMANCE,
                message="Order block detection with many other indicators may impact performance",
                error_code="EXPENSIVE_COMBINATION",
                suggestion="Consider reducing the number of indicators when using order block detection"
            ))

        return issues

    def _check_compatibility(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Check backward compatibility and version issues."""
        issues = []

        # Check version format
        version = strategy.get('version', '')
        if not version:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.COMPATIBILITY,
                message="Strategy version is missing",
                path="version",
                error_code="MISSING_VERSION",
                auto_fixable=True,
                suggestion="Add version field (e.g., '1.0.0')"
            ))

        # Check for deprecated fields or patterns
        if 'legacy_mode' in strategy:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.COMPATIBILITY,
                message="Legacy mode flag detected - consider updating to current format",
                path="legacy_mode",
                error_code="LEGACY_USAGE",
                suggestion="Remove legacy_mode flag and use current configuration format"
            ))

        return issues

    def _validate_security(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate security aspects of the configuration."""
        issues = []

        risk = strategy.get('risk', {})

        # Check position sizing limits
        position_size = risk.get('position_sizing', {}).get('value', 0)
        if position_size > self.validation_rules['max_position_size']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SECURITY,
                message=f"Position size {position_size} exceeds maximum allowed {self.validation_rules['max_position_size']}",
                path="risk.position_sizing.value",
                error_code="EXCESSIVE_POSITION_SIZE",
                suggestion=f"Reduce position size to {self.validation_rules['max_position_size']} or lower"
            ))

        # Check for missing risk controls
        if not risk.get('max_drawdown'):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.SECURITY,
                message="No maximum drawdown limit specified",
                path="risk.max_drawdown",
                error_code="NO_DRAWDOWN_LIMIT",
                auto_fixable=True,
                suggestion="Add max_drawdown field (recommended: 0.15 for 15% limit)"
            ))

        return issues

    def _validate_indicator_combinations(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate indicator combinations for effectiveness."""
        issues = []

        indicators = strategy.get('indicators', [])
        indicator_types = [ind.get('type') for ind in indicators]

        # Check for conflicting indicators
        if 'liquidity_grab_detector' in indicator_types and 'order_block_detector' in indicator_types:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category=ValidationCategory.BUSINESS_LOGIC,
                message="Liquidity Grab and Order Block detectors may provide conflicting signals",
                error_code="CONFLICTING_INDICATORS",
                suggestion="Consider using these indicators on different timeframes or with additional filters"
            ))

        # Check for redundant indicators
        type_counts = {}
        for ind_type in indicator_types:
            type_counts[ind_type] = type_counts.get(ind_type, 0) + 1

        for ind_type, count in type_counts.items():
            if count > 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.BUSINESS_LOGIC,
                    message=f"Duplicate indicator type '{ind_type}' found {count} times",
                    error_code="DUPLICATE_INDICATOR",
                    suggestion="Ensure multiple instances of the same indicator type are on different timeframes"
                ))

        return issues

    def _validate_timeframe_logic(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate timeframe logic and relationships."""
        issues = []

        indicators = strategy.get('indicators', [])
        execution = strategy.get('execution', {})
        primary_tf = execution.get('primary_timeframe', 'H1')

        timeframes = [ind.get('timeframe') for ind in indicators]

        # Check if primary timeframe is used
        if primary_tf not in timeframes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.BUSINESS_LOGIC,
                message=f"Primary timeframe '{primary_tf}' is not used by any indicator",
                path="execution.primary_timeframe",
                error_code="UNUSED_PRIMARY_TIMEFRAME",
                auto_fixable=True,
                suggestion=f"Add an indicator on {primary_tf} timeframe or change primary_timeframe"
            ))

        # Check for logical timeframe progression
        tf_hierarchy = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1']
        used_tfs = list(set(timeframes))

        if len(used_tfs) > 1:
            # Check for large gaps in timeframe hierarchy
            tf_indices = [tf_hierarchy.index(tf) for tf in used_tfs if tf in tf_hierarchy]
            if tf_indices:
                tf_indices.sort()
                for i in range(1, len(tf_indices)):
                    gap = tf_indices[i] - tf_indices[i-1]
                    if gap > 3:  # More than 3 steps in hierarchy
                        issues.append(ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            category=ValidationCategory.BUSINESS_LOGIC,
                            message=f"Large timeframe gap between {tf_hierarchy[tf_indices[i-1]]} and {tf_hierarchy[tf_indices[i]]}",
                            error_code="TIMEFRAME_GAP",
                            suggestion="Consider using intermediate timeframes for better signal correlation"
                        ))

        return issues

    def _validate_risk_parameters(self, strategy: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate risk management parameters."""
        issues = []

        risk = strategy.get('risk', {})
        timing = strategy.get('timing', {})

        # Check timeout values
        setup_timeout = timing.get('setup_timeout', 30)
        signal_timeout = timing.get('signal_timeout', 20)

        if setup_timeout < self.validation_rules['min_timeout']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.BUSINESS_LOGIC,
                message=f"Setup timeout {setup_timeout} may be too short (min recommended: {self.validation_rules['min_timeout']})",
                path="timing.setup_timeout",
                error_code="SHORT_TIMEOUT",
                suggestion=f"Consider increasing setup_timeout to at least {self.validation_rules['min_timeout']}"
            ))

        if setup_timeout > self.validation_rules['max_timeout']:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Setup timeout {setup_timeout} may be too long (max recommended: {self.validation_rules['max_timeout']})",
                path="timing.setup_timeout",
                error_code="LONG_TIMEOUT",
                suggestion=f"Consider reducing setup_timeout to under {self.validation_rules['max_timeout']}"
            ))

        return issues

    def _suggest_schema_fix(self, error) -> Optional[str]:
        """Suggest fixes for schema validation errors."""
        if "required" in error.message.lower():
            return f"Add missing required field: {error.message}"
        elif "enum" in error.message.lower():
            return f"Use one of the allowed values: {error.message}"
        elif "type" in error.message.lower():
            return f"Check data type: {error.message}"
        else:
            return None

    def auto_fix_strategy(self, strategy: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Automatically fix issues that can be resolved programmatically.

        Args:
            strategy: Strategy configuration to fix

        Returns:
            Tuple of (fixed_strategy, list_of_applied_fixes)
        """
        fixed_strategy = strategy.copy()
        applied_fixes = []

        validation_result = self.validate_strategy_comprehensive(strategy)
        auto_fixable_issues = [issue for issue in validation_result.issues if issue.auto_fixable]

        for issue in auto_fixable_issues:
            if issue.error_code == "MISSING_VERSION":
                fixed_strategy['version'] = "1.0.0"
                applied_fixes.append("Added default version '1.0.0'")

            elif issue.error_code == "NO_DRAWDOWN_LIMIT":
                if 'risk' not in fixed_strategy:
                    fixed_strategy['risk'] = {}
                fixed_strategy['risk']['max_drawdown'] = 0.15
                applied_fixes.append("Added default max_drawdown of 15%")

            elif issue.error_code == "UNNECESSARY_SEQUENTIAL":
                if 'execution' not in fixed_strategy:
                    fixed_strategy['execution'] = {}
                fixed_strategy['execution']['mode'] = 'parallel'
                applied_fixes.append("Changed execution mode from sequential to parallel for single indicator")

        return fixed_strategy, applied_fixes


# Utility functions
def validate_strategy_file(file_path: Union[str, Path]) -> ExtendedValidationResult:
    """
    Validate a strategy JSON file comprehensively.

    Args:
        file_path: Path to strategy JSON file

    Returns:
        ExtendedValidationResult with detailed validation information

    Raises:
        JSONProcessingError: If file cannot be loaded
    """
    from .json_processor import JSONProcessor

    processor = JSONProcessor()
    validator = AdvancedJSONValidator()

    strategy = processor.load_strategy_from_file(file_path)
    return validator.validate_strategy_comprehensive(strategy)


def auto_fix_strategy_file(file_path: Union[str, Path],
                          backup: bool = True) -> Tuple[List[str], ExtendedValidationResult]:
    """
    Auto-fix a strategy JSON file and save the result.

    Args:
        file_path: Path to strategy JSON file
        backup: Whether to create a backup of the original file

    Returns:
        Tuple of (applied_fixes, validation_result_after_fix)

    Raises:
        JSONProcessingError: If file operations fail
    """
    from .json_processor import JSONProcessor

    processor = JSONProcessor()
    validator = AdvancedJSONValidator()
    path = Path(file_path)

    # Create backup if requested
    if backup and path.exists():
        backup_path = path.with_suffix('.bak.json')
        backup_path.write_text(path.read_text())

    # Load, fix, and save
    strategy = processor.load_strategy_from_file(file_path)
    fixed_strategy, applied_fixes = validator.auto_fix_strategy(strategy)

    if applied_fixes:
        processor.save_strategy_to_file(fixed_strategy, file_path)

    # Validate the fixed version
    final_validation = validator.validate_strategy_comprehensive(fixed_strategy)

    return applied_fixes, final_validation