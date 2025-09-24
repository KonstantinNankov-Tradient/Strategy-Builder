"""
Test script for CHoCH Detector visualization.

This script tests the ChochDetector with EURUSD 1H data for 1 month,
creating a comprehensive plot showing swing points, CHoCH lines,
and trend change markers for validation purposes.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.data_loader import DataLoader
from indicators.choch_detector import ChochDetector


class ChochVisualizer:
    """
    Visualizer for CHoCH detection results.

    Creates comprehensive plots showing swing points, CHoCH signals,
    and trend changes for validation purposes.
    """

    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.detector = None
        self.data = None
        self.swing_points = []
        self.choch_signals = []

    def load_data(self, start_date: str = '2024-01-01', end_date: str = '2024-02-01',
                  timeframe: str = 'H1') -> pd.DataFrame:
        """
        Load EURUSD data for the specified period.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe for analysis

        Returns:
            Loaded and filtered DataFrame
        """
        # Load data from demo_strategy_builder data directory
        data_path = Path(__file__).parent.parent / 'demo_strategy_builder' / 'data' / 'EURUSD_20200101_20250809.csv'

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
        Setup the CHoCH detector with configuration.

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

        self.detector = ChochDetector(
            name='choch_test',
            config=default_config
        )

        print(f"🔧 Setup detector with config: {default_config}")

    def run_analysis(self) -> None:
        """
        Run the CHoCH analysis on the loaded data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.detector is None:
            raise ValueError("Detector not setup. Call setup_detector() first.")

        print("🔍 Running CHoCH analysis...")

        # Get all swing points and CHoCH signals for visualization
        data_with_datetime_index = self.data.set_index('datetime')
        self.swing_points, self.choch_signals = self.detector.get_all_swings_and_chochs(data_with_datetime_index)

        print(f"📊 Found {len(self.swing_points)} swing points")
        print(f"🎯 Detected {len(self.choch_signals)} CHoCH signals")

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

        print(f"🔴 Live detection found {len(live_detections)} CHoCH signals")

    def create_visualization(self) -> go.Figure:
        """
        Create comprehensive visualization of CHoCH analysis.

        Returns:
            Plotly figure with candlesticks, swing points, and CHoCH signals
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
            increasing_line_color="cyan",
            decreasing_line_color="gray",
        ))

        # Add swing points
        self._add_swing_points(fig)

        # Add CHoCH signals
        self._add_choch_signals(fig)

        # Configure layout
        title = f"<b>CHoCH Analysis</b> | {self.symbol} 1H | {self.data['datetime'].iloc[0].strftime('%Y-%m-%d')} to {self.data['datetime'].iloc[-1].strftime('%Y-%m-%d')}"

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            showlegend=False,  # Remove legend completely
            width=1400,
            height=800,
            xaxis=dict(
                showgrid=True,
                zeroline=False,
                type="date",
                tickangle=0,
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=True,
                side="right",
                showline=True,
                linewidth=2,
                linecolor="black",
            )
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

    def _add_choch_signals(self, fig: go.Figure) -> None:
        """Add CHoCH lines and markers."""
        bullish_color = "#00ff00"  # Bright green for bullish CHoCH
        bearish_color = "#ff0000"  # Bright red for bearish CHoCH

        for choch in self.choch_signals:
            swing_time = choch['swing_time']
            swing_price = choch['swing_price']
            breakout_time = choch['breakout_time']
            trend_direction = choch['trend_direction']
            trend_name = choch['trend_name']
            is_first_trend = choch.get('is_first_trend', False)

            # Choose color based on trend
            color = bullish_color if trend_direction == 1 else bearish_color

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
                name=f"{trend_name} CHoCH",
                legendgroup=f"choch_{trend_name.lower()}",
                opacity=0.8
            ))

            # Annotations removed as requested

    def show_plot(self) -> None:
        """Display the CHoCH analysis plot."""
        fig = self.create_visualization()
        fig.show()

    def save_plot(self, filename: str = None, output_dir: str = None) -> None:
        """Save the plot as HTML file."""
        if output_dir is None:
            # Save in outputs/plots directory
            output_dir = Path(__file__).parent / 'outputs' / 'plots'
        else:
            output_dir = Path(output_dir)

        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"choch_analysis_{self.symbol}_{timestamp}.html"

        # Full file path
        file_path = output_dir / filename

        fig = self.create_visualization()
        fig.write_html(file_path)
        print(f"💾 Plot saved as: {file_path}")


def main():
    """Main function to run the CHoCH visualization test."""
    try:
        print("🚀 Starting CHoCH Visualization Test")
        print("=" * 50)

        # Initialize visualizer
        visualizer = ChochVisualizer('EURUSD')

        # Load 1 month of EURUSD 1H data
        visualizer.load_data(
            start_date='2024-01-01',
            end_date='2024-02-01',
            timeframe='H1'
        )

        # Setup detector with configuration
        config = {
            'symbol': 'EURUSD',
            'base_strength': 5,
            'min_gap': 3
        }
        visualizer.setup_detector(config)

        # Run analysis
        visualizer.run_analysis()

        # Show results
        print("\n📊 Analysis Results:")
        print(f"   • Swing Points: {len(visualizer.swing_points)}")
        print(f"   • CHoCH Signals: {len(visualizer.choch_signals)}")

        if visualizer.choch_signals:
            print("\n🎯 Detected CHoCH Signals:")
            for i, choch in enumerate(visualizer.choch_signals):
                print(f"   {i+1}. {choch['breakout_time']} | "
                      f"{choch['trend_name']} CHoCH | "
                      f"Swing: {choch['swing_price']:.5f} | "
                      f"From: {choch['swing_time']} | "
                      f"{'(First Trend)' if choch.get('is_first_trend') else ''}")

        if visualizer.swing_points:
            swing_highs = [s for s in visualizer.swing_points if s['type'] == 'high']
            swing_lows = [s for s in visualizer.swing_points if s['type'] == 'low']
            print(f"\n📈 Swing Points:")
            print(f"   • Swing Highs: {len(swing_highs)}")
            print(f"   • Swing Lows: {len(swing_lows)}")

        # Create and save visualization
        print("\n📈 Creating visualization...")

        # Save to outputs/plots directory
        visualizer.save_plot()

        # Don't show plot in headless environment
        # visualizer.show_plot()

        print("\n✅ CHoCH visualization test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        raise


if __name__ == "__main__":
    main()