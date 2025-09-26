"""
Test script for Order Block Detector visualization.

This script tests the OrderBlockDetector with real market data,
creating comprehensive plots showing bullish/bearish order blocks,
mitigation status, and volume-based strength indicators.
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
from indicators.order_block_detector import OrderBlockDetector


class OrderBlockVisualizer:
    """
    Visualizer for Order Block detection results.

    Creates comprehensive plots showing:
    - Bullish Order Blocks (blue rectangles)
    - Bearish Order Blocks (orange rectangles)
    - Mitigated Order Blocks (lower opacity, end at mitigation)
    - Active Order Blocks (extend to chart end)
    - Volume-based strength indicators
    """

    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.detector = None
        self.data = None
        self.order_blocks = []

    def load_data(self, start_date: str = '2024-02-01', end_date: str = '2024-03-01',
                  timeframe: str = 'H1', data_file: str = None) -> pd.DataFrame:
        """
        Load market data for the specified period.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe for analysis
            data_file: Optional custom data file path

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Loading {self.symbol} data from {start_date} to {end_date} ({timeframe})...")

        # Use provided data file or default path
        if data_file:
            data_path = Path(data_file)
        else:
            data_path = Path(__file__).parent.parent / 'data' / f'{self.symbol}_20200101_20250809.csv'

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Load the CSV data
        self.data = self.data_loader.load_csv(data_path, symbol=self.symbol)

        # Get resampled data for the timeframe and date range
        self.data = self.data_loader.get_data(
            symbol=self.symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )

        # Reset index to have datetime as column
        self.data = self.data.reset_index()

        if self.data is None or self.data.empty:
            raise ValueError("Failed to load data")

        print(f"Loaded {len(self.data)} candles")
        print(f"Date range: {self.data['datetime'].iloc[0]} to {self.data['datetime'].iloc[-1]}")

        return self.data

    def detect_order_blocks(self, base_strength: int = 5, min_gap: int = 3,
                          close_mitigation: bool = False, track_volume: bool = True,
                          max_blocks: int = 10):
        """
        Detect Order Blocks in the loaded data.

        Args:
            base_strength: Swing detection strength
            min_gap: Minimum gap between swings
            close_mitigation: Use close price for mitigation
            track_volume: Calculate volume-based strength
            max_blocks: Maximum blocks to track
        """
        print(f"\nDetecting Order Blocks with base_strength={base_strength}...")

        # Initialize detector
        self.detector = OrderBlockDetector(
            name="ob_visualizer",
            config={
                'base_strength': base_strength,
                'min_gap': min_gap,
                'close_mitigation': close_mitigation,
                'track_volume': track_volume,
                'max_blocks': max_blocks,
                'symbol': self.symbol
            }
        )

        # Get all order blocks
        self.order_blocks = self.detector.get_all_order_blocks(self.data)

        # Print statistics
        bullish_obs = [ob for ob in self.order_blocks if ob['type'] == 'bullish']
        bearish_obs = [ob for ob in self.order_blocks if ob['type'] == 'bearish']
        mitigated_obs = [ob for ob in self.order_blocks if ob['is_mitigated']]
        active_obs = [ob for ob in self.order_blocks if not ob['is_mitigated']]

        print(f"Found {len(self.order_blocks)} total Order Blocks:")
        print(f"  - Bullish: {len(bullish_obs)}")
        print(f"  - Bearish: {len(bearish_obs)}")
        print(f"  - Active: {len(active_obs)}")
        print(f"  - Mitigated: {len(mitigated_obs)}")

        if track_volume:
            avg_strength = np.mean([ob['strength_percentage'] for ob in self.order_blocks])
            print(f"  - Average Strength: {avg_strength:.1f}%")

    def create_plot(self) -> go.Figure:
        """
        Create a Plotly figure with candlestick chart and Order Blocks.

        Returns:
            Plotly figure object
        """
        # Set data index to datetime if needed
        plot_data = self.data.copy()
        if 'datetime' in plot_data.columns:
            plot_data = plot_data.set_index('datetime')

        # Create figure with candlestick
        fig = go.Figure()

        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=plot_data.index,
            open=plot_data['open'],
            high=plot_data['high'],
            low=plot_data['low'],
            close=plot_data['close'],
            name='Price',
            increasing_line_color='#1b5e20',
            decreasing_line_color='#b2b5be',
            increasing_fillcolor='#1b5e20',
            decreasing_fillcolor='#b2b5be'
        ))

        # Add Order Blocks as shapes
        for ob in self.order_blocks:
            # Determine rectangle boundaries
            x0 = ob['formation_time']  # Start from formation time

            # End point depends on mitigation status
            if ob['is_mitigated']:
                x1 = ob['mitigation_time']
            else:
                x1 = plot_data.index[-1]  # Extend to end of chart

            # Y boundaries
            y0 = ob['bottom']
            y1 = ob['top']

            # Determine color and opacity based on type and status
            if ob['type'] == 'bullish':
                base_color = 'rgba(74, 111, 165, {opacity})'  # Blue
                border_color = 'rgb(74, 111, 165)'
            else:  # bearish
                base_color = 'rgba(255, 165, 0, {opacity})'  # Orange
                border_color = 'rgb(255, 165, 0)'

            # Set opacity based on status
            if ob['is_mitigated']:
                opacity = 0.15  # Lower opacity for mitigated
                line_width = 1
                line_dash = 'dot'
            else:
                opacity = 0.3  # Higher opacity for active
                line_width = 2
                line_dash = 'solid'

            fill_color = base_color.format(opacity=opacity)

            # Add the Order Block rectangle
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                line=dict(
                    color=border_color,
                    width=line_width,
                    dash=line_dash
                ),
                fillcolor=fill_color,
                layer='below'
            )

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

    def save_plot(self, fig: go.Figure, filename: str = None):
        """
        Save the plot to an HTML file.

        Args:
            fig: Plotly figure to save
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timeframe = self.detector.config.get('timeframe', 'H1')
            start_date = self.data['datetime'].iloc[0].strftime("%Y%m%d")
            end_date = self.data['datetime'].iloc[-1].strftime("%Y%m%d")
            filename = f"order_block_analysis_{self.symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.html"

        output_dir = project_root / "outputs" / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        fig.write_html(str(filepath))
        print(f"\nPlot saved to: {filepath}")
        return filepath

    def show_plot(self) -> None:
        """Display the Order Block analysis plot."""
        fig = self.create_plot()
        # Don't show plot in headless environment to avoid browser errors
        # fig.show()


def main():
    """Main function to run Order Block visualization."""
    parser = argparse.ArgumentParser(description='Order Block Detector Visualization')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Trading symbol (default: EURUSD)')
    parser.add_argument('--start-date', type=str, default='2024-02-01',
                       help='Start date YYYY-MM-DD (default: 2024-02-01)')
    parser.add_argument('--end-date', type=str, default='2024-03-01',
                       help='End date YYYY-MM-DD (default: 2024-03-01)')
    parser.add_argument('--timeframe', type=str, default='H1',
                       help='Timeframe (default: H1)')
    parser.add_argument('--base-strength', type=int, default=5,
                       help='Swing detection strength (default: 5)')
    parser.add_argument('--min-gap', type=int, default=3,
                       help='Minimum gap between swings (default: 3)')
    parser.add_argument('--close-mitigation', action='store_true',
                       help='Use close price for mitigation')
    parser.add_argument('--no-volume', action='store_true',
                       help='Disable volume-based strength calculation')
    parser.add_argument('--max-blocks', type=int, default=10,
                       help='Maximum blocks to track (default: 10)')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Custom data file path')

    args = parser.parse_args()

    # Create visualizer
    visualizer = OrderBlockVisualizer(symbol=args.symbol)

    try:
        # Load data
        visualizer.load_data(
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            data_file=args.data_file
        )

        # Detect Order Blocks
        visualizer.detect_order_blocks(
            base_strength=args.base_strength,
            min_gap=args.min_gap,
            close_mitigation=args.close_mitigation,
            track_volume=not args.no_volume,
            max_blocks=args.max_blocks
        )

        # Create and save plot
        fig = visualizer.create_plot()
        visualizer.save_plot(fig)

        # Don't show plot to avoid browser errors
        # visualizer.show_plot()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())