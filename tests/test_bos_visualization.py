"""
Test script for BOS Detector visualization.

This script tests the BosDetector with EURUSD 1H data for 1 month,
creating a comprehensive plot showing swing points, BOS lines,
and trend continuation markers for validation purposes.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.data_loader import DataLoader
from indicators.bos_detector import BosDetector


class BosVisualizer:
    """
    Visualizer for BOS detection results.

    Creates comprehensive plots showing swing points, BOS signals,
    and trend continuations for validation purposes.
    """

    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.detector = None
        self.data = None
        self.swing_points = []
        self.bos_signals = []

    def load_data(self, start_date: str = '2024-02-01', end_date: str = '2024-02-05',
                  timeframe: str = 'H1', data_file: str = None) -> pd.DataFrame:
        """
        Load EURUSD data for the specified period.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe for analysis
            data_file: Optional specific data file path

        Returns:
            Loaded and filtered DataFrame
        """
        # Use provided data file or default path
        if data_file:
            data_path = Path(data_file)
        else:
            data_path = Path(__file__).parent.parent / 'data' / f'{self.symbol}_20200101_20250809.csv'

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"📊 Loading {self.symbol} data from {data_path}")

        # Load the CSV data
        self.data = self.data_loader.load_csv(data_path, symbol=self.symbol)

        # Get resampled data for the timeframe and date range
        self.data = self.data_loader.get_data(
            symbol=self.symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        print(f"✅ Loaded {len(self.data)} candles for {self.symbol}")
        print(f"📅 Date range: {self.data.index[0]} to {self.data.index[-1]}")

        # Reset index to have datetime as column for indicator
        self.data = self.data.reset_index()

        return self.data

    def setup_detector(self, config: dict = None) -> None:
        """
        Setup the BOS detector with configuration.

        Args:
            config: Configuration dictionary for the detector
        """
        default_config = {
            'symbol': self.symbol,
            'base_strength': 5,
            'min_gap': 3
        }

        if config:
            default_config.update(config)

        self.detector = BosDetector(
            name='bos_test',
            config=default_config
        )

        print(f"🔧 Setup detector with config: {default_config}")

    def run_analysis(self) -> None:
        """
        Run the BOS analysis on the loaded data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.detector is None:
            raise ValueError("Detector not setup. Call setup_detector() first.")

        print("🔍 Running BOS analysis...")

        # Get all swing points and BOS signals for visualization
        data_with_datetime_index = self.data.set_index('datetime')
        self.swing_points, self.bos_signals = self.detector.get_all_swings_and_bos(data_with_datetime_index)

        print(f"📊 Found {len(self.swing_points)} swing points")
        print(f"🎯 Detected {len(self.bos_signals)} BOS signals")

        # Also test live detection by running detector on each candle
        live_detections = []
        lookback = self.detector.get_lookback_period()

        for i in range(lookback, len(self.data)):
            # Get data window for current candle
            window_data = self.data.iloc[max(0, i - lookback):i + 1].copy()

            # Run detector
            detection = self.detector.check(window_data, i)

            if detection:
                live_detections.append({
                    'timestamp': detection.timestamp,
                    'candle_index': detection.candle_index,
                    'price': detection.price,
                    'direction': detection.direction,
                    'metadata': detection.metadata
                })

        print(f"🔴 Live detection found {len(live_detections)} BOS signals")

    def create_visualization(self) -> go.Figure:
        """
        Create comprehensive visualization of BOS analysis.

        Returns:
            Plotly figure with candlesticks, swing points, and BOS signals
        """
        if self.data is None:
            raise ValueError("No data to visualize")

        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=self.data['datetime'],
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name="Price",
            increasing_line_color="#1b5e20",
            decreasing_line_color="#b2b5be",
            increasing_fillcolor="#1b5e20",
            decreasing_fillcolor="#b2b5be"
        ))

        # Add swing points
        self._add_swing_points(fig)

        # Add BOS signals
        self._add_bos_signals(fig)

        # Configure layout with consistent styling
        period = f"{self.data['datetime'].iloc[0].strftime('%Y-%m-%d')} to {self.data['datetime'].iloc[-1].strftime('%Y-%m-%d')}"
        timeframe = "H1"  # Default to H1 as per the test

        fig.update_layout(
            showlegend=False,
            title=f"<b>Pair:</b> {self.symbol} | <b>Interval:</b> {timeframe} | <b>Period:</b> {period}",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="ggplot2",
            xaxis=dict(
                showgrid=True,
                zeroline=False,
                title_standoff=15,
                type="date",
                tickangle=0,
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                title_standoff=15,
                side="right",
                showline=True,
                linewidth=2,
                linecolor="black",
            ),
            width=1900,
            height=1100,
        )

        return fig

    def _add_swing_points(self, fig: go.Figure) -> None:
        """Add swing high/low markers."""
        swing_high_color = "#8A2BE2"  # Blue Violet for swing highs
        swing_low_color = "#FF4500"   # Orange Red for swing lows

        for swing in self.swing_points:
            if swing['type'] == 'high':
                color = swing_high_color
                symbol = "star"
                name = "Swing High"
            else:
                color = swing_low_color
                symbol = "star"
                name = "Swing Low"

            fig.add_trace(go.Scatter(
                x=[swing['time']],
                y=[swing['price']],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=14,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                showlegend=False,  # Remove from legend
                name=name,
                legendgroup=f"swings_{swing['type']}"
            ))

    def _add_bos_signals(self, fig: go.Figure) -> None:
        """Add BOS lines and markers."""
        bullish_bos_color = "#00ff00"  # Bright green for bullish BOS
        bearish_bos_color = "#ff0000"  # Bright red for bearish BOS

        for bos in self.bos_signals:
            swing_time = bos['swing_time']
            swing_price = bos['swing_price']
            breakout_time = bos['breakout_time']
            trend_direction = bos['trend_direction']
            trend_name = bos['trend_name']

            # Choose color based on trend
            color = bullish_bos_color if trend_direction == 1 else bearish_bos_color

            # Add horizontal line from swing to breakout
            fig.add_trace(go.Scatter(
                x=[swing_time, breakout_time],
                y=[swing_price, swing_price],
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=2,
                    dash='solid'
                ),
                marker=dict(
                    size=[8, 12],  # Smaller at swing, larger at breakout
                    color=color,
                    symbol=['circle', 'diamond'],  # Circle at swing, diamond at breakout
                    line=dict(width=2, color='white')
                ),
                showlegend=True,
                name=f"{trend_name} BOS",
                legendgroup=f"bos_{trend_name.lower()}",
                opacity=0.9
            ))

            # Add annotation for BOS
            fig.add_annotation(
                x=breakout_time,
                y=swing_price,
                text="BOS",
                showarrow=False,
                font=dict(
                    size=10,
                    color=color
                ),
                xshift=15,
                yshift=10 if trend_direction == 1 else -10
            )

    def show_plot(self) -> None:
        """Display the BOS analysis plot."""
        fig = self.create_visualization()
        # Don't show plot in headless environment to avoid browser errors
        # fig.show()

    def save_plot(self, filename: str = None, output_dir: str = None,
                  timeframe: str = 'H1', start_date: str = None, end_date: str = None) -> None:
        """Save the plot as HTML file."""
        if output_dir is None:
            # Save in outputs/plots directory at project root
            output_dir = Path(__file__).parent.parent / 'outputs' / 'plots'
        else:
            output_dir = Path(output_dir)

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            date_range = ""
            if start_date and end_date:
                date_range = f"_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
            filename = f"bos_analysis_{self.symbol}_{timeframe}{date_range}_{timestamp}.html"

        # Full file path
        file_path = output_dir / filename

        fig = self.create_visualization()
        fig.write_html(file_path)
        print(f"💾 Plot saved as: {file_path}")


def main():
    """Main function to run the BOS visualization test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BOS Detector Visualization Test')
    parser.add_argument('--timeframe', default='H1',
                       choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
                       help='Timeframe for analysis (default: H1)')
    parser.add_argument('--start-date', default='2024-02-01',
                       help='Start date in YYYY-MM-DD format (default: 2024-02-01)')
    parser.add_argument('--end-date', default='2024-02-05',
                       help='End date in YYYY-MM-DD format (default: 2024-02-05)')
    parser.add_argument('--symbol', default='EURUSD',
                       help='Currency pair symbol (default: EURUSD)')
    parser.add_argument('--data-file',
                       help='Optional: specific data file path')
    parser.add_argument('--base-strength', type=int, default=5,
                       help='BOS base strength parameter (default: 5)')
    parser.add_argument('--min-gap', type=int, default=3,
                       help='Minimum gap between swings (default: 3)')

    args = parser.parse_args()

    # Validate dates
    try:
        from datetime import datetime
        start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
        if start_dt >= end_dt:
            print("❌ Error: Start date must be before end date")
            return
    except ValueError:
        print("❌ Error: Invalid date format. Use YYYY-MM-DD")
        return

    try:
        print("🚀 Starting BOS Visualization Test")
        print("=" * 50)
        print(f"📊 Configuration:")
        print(f"   • Symbol: {args.symbol}")
        print(f"   • Timeframe: {args.timeframe}")
        print(f"   • Date Range: {args.start_date} to {args.end_date}")
        print(f"   • Base Strength: {args.base_strength}")
        print(f"   • Min Gap: {args.min_gap}")
        if args.data_file:
            print(f"   • Data File: {args.data_file}")

        # Initialize visualizer
        visualizer = BosVisualizer(args.symbol)

        # Load data with provided parameters
        visualizer.load_data(
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            data_file=args.data_file
        )

        # Setup detector with configuration
        config = {
            'symbol': args.symbol,
            'base_strength': args.base_strength,
            'min_gap': args.min_gap
        }
        visualizer.setup_detector(config)

        # Run analysis
        visualizer.run_analysis()

        # Show results
        print("\n📊 Analysis Results:")
        print(f"   • Swing Points: {len(visualizer.swing_points)}")
        print(f"   • BOS Signals: {len(visualizer.bos_signals)}")

        if visualizer.bos_signals:
            print("\n🎯 Detected BOS Signals (Trend Continuations):")
            for i, bos in enumerate(visualizer.bos_signals):
                print(f"   {i+1}. {bos['breakout_time']} | "
                      f"{bos['trend_name']} BOS | "
                      f"Swing: {bos['swing_price']:.5f} | "
                      f"From: {bos['swing_time']}")

        if visualizer.swing_points:
            swing_highs = [s for s in visualizer.swing_points if s['type'] == 'high']
            swing_lows = [s for s in visualizer.swing_points if s['type'] == 'low']
            print(f"\n📈 Swing Points:")
            print(f"   • Swing Highs: {len(swing_highs)}")
            print(f"   • Swing Lows: {len(swing_lows)}")

        # Create and save visualization
        print("\n📈 Creating visualization...")

        # Save to outputs/plots directory with parameters
        visualizer.save_plot(
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Don't show plot in headless environment
        # visualizer.show_plot()

        print("\n✅ BOS visualization test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        raise


if __name__ == "__main__":
    main()