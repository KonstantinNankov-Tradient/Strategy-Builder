"""
Multi-timeframe indicator coordination system.

This module manages the coordination of indicators across multiple timeframes,
handling the complex logic of when and how indicators should be executed
based on timeframe relationships and strategy requirements.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd

from .multi_timeframe_state import (
    TimeframeDetection,
    MultiTimeframeSetupContext,
    TimeframeSyncStatus
)
from .multi_timeframe_detection import (
    MultiTimeframeDetectionProcessor,
    DetectionCluster,
    ConfirmationRequest
)
from .data_loader import TimeframeConverter
from .json_converter import MultiTimeframeStrategyConfig


logger = logging.getLogger(__name__)


class IndicatorPriority(Enum):
    """Priority levels for indicator execution."""
    CRITICAL = 1    # Must execute immediately
    HIGH = 2        # Execute as soon as possible
    NORMAL = 3      # Execute in normal order
    LOW = 4         # Execute when convenient
    BACKGROUND = 5  # Execute when no other work


class CoordinationStrategy(Enum):
    """Strategies for coordinating indicators across timeframes."""
    SEQUENTIAL = "sequential"       # Execute indicators in strict order
    PARALLEL = "parallel"          # Execute all indicators simultaneously
    HIERARCHICAL = "hierarchical"  # Higher timeframes first, then lower
    ADAPTIVE = "adaptive"          # Adapt based on market conditions


@dataclass
class IndicatorTask:
    """Represents an indicator execution task."""
    indicator_name: str
    timeframe: str
    priority: IndicatorPriority
    data_context: Any  # TimeframeDataContext
    candle_index: int
    scheduled_time: datetime
    dependencies: List[str] = field(default_factory=list)

    # Execution state
    executed: bool = False
    execution_time: Optional[datetime] = None
    result: Optional[TimeframeDetection] = None
    error: Optional[str] = None

    # Performance tracking
    execution_duration: float = 0.0
    retry_count: int = 0

    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if this task can be executed based on dependencies."""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class CoordinationContext:
    """Context for coordinating indicators across timeframes."""
    strategy_config: MultiTimeframeStrategyConfig
    current_setup: Optional[MultiTimeframeSetupContext] = None

    # Task management
    pending_tasks: List[IndicatorTask] = field(default_factory=list)
    completed_tasks: Dict[str, IndicatorTask] = field(default_factory=dict)
    failed_tasks: Dict[str, IndicatorTask] = field(default_factory=dict)

    # Coordination state
    coordination_strategy: CoordinationStrategy = CoordinationStrategy.SEQUENTIAL
    active_timeframes: Set[str] = field(default_factory=set)
    blocked_timeframes: Set[str] = field(default_factory=set)

    # Performance metrics
    total_tasks_executed: int = 0
    avg_execution_time: float = 0.0
    coordination_efficiency: float = 1.0


class MultiTimeframeIndicatorCoordinator:
    """
    Coordinates indicator execution across multiple timeframes.

    Manages the complex orchestration of when and how indicators execute
    based on timeframe relationships, dependencies, and performance constraints.
    """

    def __init__(self, strategy_config: MultiTimeframeStrategyConfig):
        """
        Initialize the indicator coordinator.

        Args:
            strategy_config: Multi-timeframe strategy configuration
        """
        self.strategy_config = strategy_config
        self.timeframe_converter = TimeframeConverter()
        self.logger = logging.getLogger(__name__)

        # Coordination configuration
        self.coordination_context = CoordinationContext(strategy_config=strategy_config)
        self.max_concurrent_tasks = 5
        self.task_timeout_seconds = 30.0

        # Timeframe hierarchy for coordination
        self.timeframe_hierarchy = {
            'M1': 1, 'M5': 2, 'M15': 3, 'M30': 4,
            'H1': 5, 'H4': 6, 'D1': 7, 'W1': 8
        }

        # Performance tracking
        self.performance_stats = {
            'total_coordinations': 0,
            'successful_coordinations': 0,
            'failed_coordinations': 0,
            'avg_coordination_time': 0.0,
            'timeframe_utilization': {}
        }

    def coordinate_indicator_execution(
        self,
        timeframe_data: Dict[str, Any],  # timeframe -> TimeframeDataContext
        indicators: Dict[str, Any],      # indicator_name -> BaseIndicator
        current_setup: Optional[MultiTimeframeSetupContext] = None
    ) -> Dict[str, List[TimeframeDetection]]:
        """
        Coordinate the execution of indicators across timeframes.

        Args:
            timeframe_data: Data contexts for each timeframe
            indicators: Available indicator instances
            current_setup: Current multi-timeframe setup context

        Returns:
            Dictionary mapping timeframes to detected signals
        """
        start_time = datetime.now()
        self.performance_stats['total_coordinations'] += 1

        try:
            # Update coordination context
            self.coordination_context.current_setup = current_setup
            self.coordination_context.active_timeframes = set(timeframe_data.keys())

            # Create execution plan
            execution_plan = self._create_execution_plan(timeframe_data, indicators)

            # Execute plan based on coordination strategy
            results = self._execute_coordination_plan(execution_plan, timeframe_data, indicators)

            # Update performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(execution_time, True)

            self.logger.info(
                f"Indicator coordination completed: {len(results)} timeframes, "
                f"{sum(len(detections) for detections in results.values())} total detections"
            )

            self.performance_stats['successful_coordinations'] += 1
            return results

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(execution_time, False)
            self.performance_stats['failed_coordinations'] += 1

            self.logger.error(f"Indicator coordination failed: {e}")
            return {}

    def _create_execution_plan(
        self,
        timeframe_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> List[IndicatorTask]:
        """Create an optimized execution plan for indicators."""
        tasks = []
        current_time = datetime.now()

        # Create tasks for each indicator-timeframe combination
        for timeframe, context in timeframe_data.items():
            if not context.is_ready_to_process():
                continue

            # Get indicators for this timeframe
            timeframe_indicators = self.strategy_config.get_indicators_for_timeframe(timeframe)

            for indicator_name in timeframe_indicators:
                if indicator_name not in indicators:
                    continue

                # Determine task priority
                priority = self._calculate_indicator_priority(
                    indicator_name, timeframe, context
                )

                # Determine dependencies
                dependencies = self._get_indicator_dependencies(
                    indicator_name, timeframe
                )

                task = IndicatorTask(
                    indicator_name=indicator_name,
                    timeframe=timeframe,
                    priority=priority,
                    data_context=context,
                    candle_index=context.current_index,
                    scheduled_time=current_time,
                    dependencies=dependencies
                )

                tasks.append(task)

        # Sort tasks by coordination strategy
        return self._sort_tasks_by_strategy(tasks)

    def _calculate_indicator_priority(
        self,
        indicator_name: str,
        timeframe: str,
        context: Any
    ) -> IndicatorPriority:
        """Calculate priority for an indicator execution."""
        # Base priority on timeframe hierarchy
        tf_rank = self.timeframe_hierarchy.get(timeframe, 5)

        # Higher timeframes get higher priority in hierarchical mode
        if self.coordination_context.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
            if tf_rank >= 7:  # D1, W1
                return IndicatorPriority.HIGH
            elif tf_rank >= 5:  # H1, H4
                return IndicatorPriority.NORMAL
            else:  # M1-M30
                return IndicatorPriority.LOW

        # Check if this is the primary timeframe
        if timeframe == self.strategy_config.primary_timeframe:
            return IndicatorPriority.HIGH

        # Check indicator importance in strategy sequence
        indicator_sequence = self.strategy_config.indicator_sequence
        if indicator_name in indicator_sequence:
            sequence_pos = indicator_sequence.index(indicator_name)
            if sequence_pos == 0:  # First indicator
                return IndicatorPriority.CRITICAL
            elif sequence_pos == 1:  # Second indicator
                return IndicatorPriority.HIGH
            else:
                return IndicatorPriority.NORMAL

        return IndicatorPriority.LOW

    def _get_indicator_dependencies(self, indicator_name: str, timeframe: str) -> List[str]:
        """Get dependencies for an indicator execution."""
        dependencies = []

        # Sequential execution requires previous indicators to complete
        if self.coordination_context.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
            indicator_sequence = self.strategy_config.indicator_sequence
            if indicator_name in indicator_sequence:
                current_pos = indicator_sequence.index(indicator_name)
                for i in range(current_pos):
                    prev_indicator = indicator_sequence[i]
                    # Create dependency key
                    dep_key = f"{prev_indicator}_{timeframe}"
                    dependencies.append(dep_key)

        # Hierarchical execution requires higher timeframes to complete first
        elif self.coordination_context.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
            current_rank = self.timeframe_hierarchy.get(timeframe, 5)
            for tf, rank in self.timeframe_hierarchy.items():
                if rank > current_rank and tf in self.coordination_context.active_timeframes:
                    dep_key = f"{indicator_name}_{tf}"
                    dependencies.append(dep_key)

        return dependencies

    def _sort_tasks_by_strategy(self, tasks: List[IndicatorTask]) -> List[IndicatorTask]:
        """Sort tasks based on coordination strategy."""
        if self.coordination_context.coordination_strategy == CoordinationStrategy.SEQUENTIAL:
            # Sort by indicator sequence order, then by timeframe hierarchy
            def sort_key(task):
                sequence_order = 999
                if task.indicator_name in self.strategy_config.indicator_sequence:
                    sequence_order = self.strategy_config.indicator_sequence.index(task.indicator_name)

                tf_hierarchy = self.timeframe_hierarchy.get(task.timeframe, 5)
                return (sequence_order, task.priority.value, tf_hierarchy)

            tasks.sort(key=sort_key)

        elif self.coordination_context.coordination_strategy == CoordinationStrategy.HIERARCHICAL:
            # Sort by timeframe hierarchy (higher first), then by priority
            def sort_key(task):
                tf_hierarchy = -self.timeframe_hierarchy.get(task.timeframe, 5)  # Negative for reverse order
                return (tf_hierarchy, task.priority.value)

            tasks.sort(key=sort_key)

        elif self.coordination_context.coordination_strategy == CoordinationStrategy.PARALLEL:
            # Sort by priority only
            tasks.sort(key=lambda task: task.priority.value)

        elif self.coordination_context.coordination_strategy == CoordinationStrategy.ADAPTIVE:
            # Adaptive sorting based on current conditions
            tasks = self._adaptive_task_sorting(tasks)

        return tasks

    def _adaptive_task_sorting(self, tasks: List[IndicatorTask]) -> List[IndicatorTask]:
        """Adaptively sort tasks based on current market conditions."""
        # Analyze current setup state
        if not self.coordination_context.current_setup:
            # No active setup - prioritize primary timeframe
            primary_tasks = [t for t in tasks if t.timeframe == self.strategy_config.primary_timeframe]
            other_tasks = [t for t in tasks if t.timeframe != self.strategy_config.primary_timeframe]

            primary_tasks.sort(key=lambda t: t.priority.value)
            other_tasks.sort(key=lambda t: (t.priority.value, self.timeframe_hierarchy.get(t.timeframe, 5)))

            return primary_tasks + other_tasks

        # Active setup - prioritize based on what's needed
        needed_indicators = set(self.strategy_config.indicator_sequence) - set(
            self.coordination_context.current_setup.detected_indicators
        )

        priority_tasks = [t for t in tasks if t.indicator_name in needed_indicators]
        other_tasks = [t for t in tasks if t.indicator_name not in needed_indicators]

        priority_tasks.sort(key=lambda t: t.priority.value)
        other_tasks.sort(key=lambda t: t.priority.value)

        return priority_tasks + other_tasks

    def _execute_coordination_plan(
        self,
        tasks: List[IndicatorTask],
        timeframe_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Dict[str, List[TimeframeDetection]]:
        """Execute the coordination plan."""
        results = {tf: [] for tf in timeframe_data.keys()}
        completed_task_keys = set()

        # Execute tasks in order, respecting dependencies
        for task in tasks:
            if not task.can_execute(completed_task_keys):
                self.logger.debug(f"Skipping {task.indicator_name}@{task.timeframe} - dependencies not met")
                continue

            # Execute indicator
            detection = self._execute_indicator_task(task, indicators)

            if detection:
                results[task.timeframe].append(detection)
                task.result = detection
                task.executed = True
                self.logger.debug(f"Detection: {task.indicator_name}@{task.timeframe} -> {detection.direction.value}")

            # Mark task as completed
            task_key = f"{task.indicator_name}_{task.timeframe}"
            completed_task_keys.add(task_key)

            # Update coordination context
            self.coordination_context.completed_tasks[task_key] = task
            self.coordination_context.total_tasks_executed += 1

        return results

    def _execute_indicator_task(
        self,
        task: IndicatorTask,
        indicators: Dict[str, Any]
    ) -> Optional[TimeframeDetection]:
        """Execute a single indicator task."""
        start_time = datetime.now()

        try:
            indicator = indicators[task.indicator_name]
            context = task.data_context

            # Prepare data for indicator
            candle_data = context.data.iloc[:context.current_index + 1]

            # Execute indicator
            legacy_detection = indicator.check(candle_data, context.current_index)

            if legacy_detection:
                # Convert to timeframe detection
                from .multi_timeframe_state import convert_legacy_detection_to_timeframe
                tf_detection = convert_legacy_detection_to_timeframe(
                    legacy_detection, task.timeframe
                )
                tf_detection.timeframe_candle_index = context.current_index

                # Calculate execution duration
                task.execution_duration = (datetime.now() - start_time).total_seconds()
                task.execution_time = datetime.now()

                self.logger.debug(
                    f"Indicator {task.indicator_name}@{task.timeframe} detected signal at {tf_detection.price:.5f}"
                )

                return tf_detection

        except Exception as e:
            task.error = str(e)
            task.retry_count += 1
            self.logger.error(f"Indicator execution failed: {task.indicator_name}@{task.timeframe} - {e}")

        finally:
            task.execution_duration = (datetime.now() - start_time).total_seconds()
            task.execution_time = datetime.now()

        return None

    def optimize_coordination_strategy(self) -> CoordinationStrategy:
        """Analyze performance and optimize coordination strategy."""
        # Analyze recent performance
        if self.performance_stats['total_coordinations'] < 10:
            return self.coordination_context.coordination_strategy

        success_rate = (
            self.performance_stats['successful_coordinations'] /
            self.performance_stats['total_coordinations']
        )

        avg_time = self.performance_stats['avg_coordination_time']

        # Decision logic for strategy optimization
        if success_rate < 0.7:
            # Poor success rate - try sequential for reliability
            return CoordinationStrategy.SEQUENTIAL
        elif avg_time > 2.0:
            # Slow execution - try parallel for speed
            return CoordinationStrategy.PARALLEL
        elif success_rate > 0.9 and avg_time < 1.0:
            # Good performance - try adaptive for optimization
            return CoordinationStrategy.ADAPTIVE

        return self.coordination_context.coordination_strategy

    def get_coordination_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coordination statistics."""
        stats = {
            'performance_stats': self.performance_stats.copy(),
            'current_strategy': self.coordination_context.coordination_strategy.value,
            'active_timeframes': list(self.coordination_context.active_timeframes),
            'blocked_timeframes': list(self.coordination_context.blocked_timeframes),
            'total_tasks_executed': self.coordination_context.total_tasks_executed,
            'avg_execution_time': self.coordination_context.avg_execution_time,
            'coordination_efficiency': self.coordination_context.coordination_efficiency
        }

        # Task statistics
        completed_count = len(self.coordination_context.completed_tasks)
        failed_count = len(self.coordination_context.failed_tasks)

        stats['task_statistics'] = {
            'completed_tasks': completed_count,
            'failed_tasks': failed_count,
            'success_rate': completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0.0
        }

        # Timeframe utilization
        timeframe_usage = {}
        for tf in self.coordination_context.active_timeframes:
            tf_tasks = [task for task in self.coordination_context.completed_tasks.values()
                       if task.timeframe == tf]
            timeframe_usage[tf] = len(tf_tasks)

        stats['timeframe_utilization'] = timeframe_usage

        return stats

    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance tracking metrics."""
        # Update average coordination time
        total_coords = self.performance_stats['total_coordinations']
        current_avg = self.performance_stats['avg_coordination_time']

        self.performance_stats['avg_coordination_time'] = (
            (current_avg * (total_coords - 1) + execution_time) / total_coords
        )

        # Update context metrics
        if self.coordination_context.total_tasks_executed > 0:
            total_exec_time = sum(
                task.execution_duration
                for task in self.coordination_context.completed_tasks.values()
            )
            self.coordination_context.avg_execution_time = (
                total_exec_time / self.coordination_context.total_tasks_executed
            )

        # Calculate efficiency
        expected_time = len(self.coordination_context.active_timeframes) * 0.1  # 100ms per timeframe
        if execution_time > 0:
            self.coordination_context.coordination_efficiency = min(1.0, expected_time / execution_time)

    def reset_coordination_context(self):
        """Reset coordination context for new setup."""
        self.coordination_context.pending_tasks.clear()
        self.coordination_context.completed_tasks.clear()
        self.coordination_context.failed_tasks.clear()
        self.coordination_context.current_setup = None
        self.coordination_context.total_tasks_executed = 0

    def set_coordination_strategy(self, strategy: CoordinationStrategy):
        """Manually set coordination strategy."""
        old_strategy = self.coordination_context.coordination_strategy
        self.coordination_context.coordination_strategy = strategy

        self.logger.info(f"Coordination strategy changed: {old_strategy.value} -> {strategy.value}")


# Utility functions for coordination
def create_indicator_coordinator(config: MultiTimeframeStrategyConfig) -> MultiTimeframeIndicatorCoordinator:
    """
    Factory function to create an indicator coordinator.

    Args:
        config: Multi-timeframe strategy configuration

    Returns:
        Configured MultiTimeframeIndicatorCoordinator instance
    """
    return MultiTimeframeIndicatorCoordinator(config)


def analyze_coordination_performance(coordinator: MultiTimeframeIndicatorCoordinator) -> Dict[str, Any]:
    """
    Analyze coordination performance and provide recommendations.

    Args:
        coordinator: Coordinator instance to analyze

    Returns:
        Performance analysis with recommendations
    """
    stats = coordinator.get_coordination_statistics()

    analysis = {
        'current_performance': stats,
        'recommendations': [],
        'optimization_opportunities': []
    }

    # Performance analysis
    success_rate = stats['task_statistics']['success_rate']
    avg_time = stats['performance_stats']['avg_coordination_time']
    efficiency = stats['coordination_efficiency']

    # Generate recommendations
    if success_rate < 0.8:
        analysis['recommendations'].append(
            "Consider using SEQUENTIAL coordination strategy for better reliability"
        )

    if avg_time > 1.5:
        analysis['recommendations'].append(
            "Consider using PARALLEL coordination strategy for better performance"
        )

    if efficiency < 0.7:
        analysis['optimization_opportunities'].append(
            "Optimize indicator execution order for better efficiency"
        )

    # Timeframe analysis
    tf_usage = stats['timeframe_utilization']
    if tf_usage:
        max_usage = max(tf_usage.values())
        min_usage = min(tf_usage.values())

        if max_usage > min_usage * 3:
            analysis['optimization_opportunities'].append(
                "Timeframe usage is unbalanced - consider redistributing indicators"
            )

    return analysis