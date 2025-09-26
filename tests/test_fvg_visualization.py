"""
Test script for FVG Detector visualization.

This script tests the FvgDetector with real market data,
creating a comprehensive plot showing bullish/bearish FVGs,
mitigated gaps, shrunk gaps, and unmitigated gaps.
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
from indicators.fvg_detector import FvgDetector


class FvgVisualizer:
    """
    Visualizer for FVG detection results.

    Creates comprehensive plots showing:
    - Bullish FVGs (green rectangles)
    - Bearish FVGs (red rectangles)
    - Mitigated FVGs (lower opacity, end at mitigation candle)
    - Shrunk FVGs (show current boundaries)
    - Unmitigated FVGs (extend to chart end)
    """

    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.detector = None
        self.data = None
        self.fvgs = []

    def load_data(self, start_date: str = '2024-02-01', end_date: str = '2024-02-10',
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

    def detect_fvgs(self, min_gap_size: float = 0.0,
                    track_mitigation: bool = True,
                    track_shrinking: bool = True):
        """
        Detect FVGs in the loaded data.

        Args:
            min_gap_size: Minimum gap size to consider
            track_mitigation: Track gap filling
            track_shrinking: Track partial gap fills
        """
        print(f"\nDetecting FVGs with min_gap_size={min_gap_size}...")

        # Initialize detector
        self.detector = FvgDetector(
            name="fvg_visualizer",
            config={
                'min_gap_size': min_gap_size,
                'track_mitigation': track_mitigation,
                'track_shrinking': track_shrinking,
                'symbol': self.symbol
            }
        )

        # Get all FVGs
        self.fvgs = self.detector.get_all_fvgs(self.data)

        # Print statistics
        bullish_fvgs = [f for f in self.fvgs if f['type'] == 'bullish']
        bearish_fvgs = [f for f in self.fvgs if f['type'] == 'bearish']
        mitigated_fvgs = [f for f in self.fvgs if f['is_mitigated']]
        shrunk_fvgs = [f for f in self.fvgs if len(f['shrink_history']) > 0]

        print(f"Found {len(self.fvgs)} total FVGs:")
        print(f"  - Bullish: {len(bullish_fvgs)}")
        print(f"  - Bearish: {len(bearish_fvgs)}")
        print(f"  - Mitigated: {len(mitigated_fvgs)}")
        print(f"  - Shrunk: {len(shrunk_fvgs)}")

    def create_plot(self, show_original_boundaries: bool = True) -> go.Figure:
        """
        Create a Plotly figure with candlestick chart and FVGs.

        Args:
            show_original_boundaries: Show original gap boundaries for shrunk FVGs

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

        # Add FVGs as shapes
        for fvg in self.fvgs:
            # Determine rectangle boundaries
            x0 = fvg['candle1_time']  # Start from first candle of formation

            # End point depends on mitigation status
            if fvg['is_mitigated']:
                x1 = fvg['mitigation_time']
            else:
                x1 = plot_data.index[-1]  # Extend to end of chart

            # Y boundaries (current boundaries after shrinking)
            y0 = fvg['current_bottom']
            y1 = fvg['current_top']

            # Determine color and opacity based on type and status
            if fvg['type'] == 'bullish':
                base_color = 'green'
                fill_color_base = 'rgba(0, 255, 0, {opacity})'
            else:  # bearish
                base_color = 'red'
                fill_color_base = 'rgba(255, 0, 0, {opacity})'

            # Set opacity based on status
            if fvg['is_mitigated']:
                opacity = 0.1  # Low opacity for mitigated
                line_width = 1
                line_dash = 'dot'
            elif len(fvg['shrink_history']) > 0:
                opacity = 0.2  # Medium opacity for shrunk
                line_width = 1.5
                line_dash = 'dash'
            else:
                opacity = 0.3  # Full opacity for untouched
                line_width = 2
                line_dash = 'solid'

            fill_color = fill_color_base.format(opacity=opacity)

            # Add the FVG rectangle
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                line=dict(
                    color=base_color,
                    width=line_width,
                    dash=line_dash
                ),
                fillcolor=fill_color,
                layer='below'
            )

            # Add original boundaries for shrunk FVGs (optional)
            if show_original_boundaries and len(fvg['shrink_history']) > 0:
                fig.add_shape(
                    type="rect",
                    x0=x0, x1=x1,
                    y0=fvg['original_bottom'], y1=fvg['original_top'],
                    line=dict(
                        color=base_color,
                        width=0.5,
                        dash='dot'
                    ),
                    fillcolor='rgba(0, 0, 0, 0)',  # No fill, just outline
                    layer='below'
                )

        # Configure layout with consistent styling
        period = f'{self.data.iloc[0]["datetime"].date()} to {self.data.iloc[-1]["datetime"].date()}'
        timeframe = self.detector.config.get('timeframe', 'H1')

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

    def _generate_stats_text(self) -> str:
        """Generate statistics text for the plot."""
        bullish_fvgs = [f for f in self.fvgs if f['type'] == 'bullish']
        bearish_fvgs = [f for f in self.fvgs if f['type'] == 'bearish']
        mitigated_fvgs = [f for f in self.fvgs if f['is_mitigated']]
        shrunk_fvgs = [f for f in self.fvgs if len(f['shrink_history']) > 0 and not f['is_mitigated']]
        active_fvgs = [f for f in self.fvgs if not f['is_mitigated'] and len(f['shrink_history']) == 0]

        stats = f"""FVG Statistics:
Total: {len(self.fvgs)}
Bullish: {len(bullish_fvgs)}
Bearish: {len(bearish_fvgs)}
Active: {len(active_fvgs)}
Shrunk: {len(shrunk_fvgs)}
Mitigated: {len(mitigated_fvgs)}"""

        return stats

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
            start_date = self.data.iloc[0]['datetime'].strftime("%Y%m%d")
            end_date = self.data.iloc[-1]['datetime'].strftime("%Y%m%d")
            filename = f"fvg_analysis_{self.symbol}_{timeframe}_{start_date}_{end_date}_{timestamp}.html"

        output_dir = project_root / "outputs" / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        fig.write_html(str(filepath))
        print(f"\nPlot saved to: {filepath}")
        return filepath


def main():
    """Main function to run FVG visualization."""
    parser = argparse.ArgumentParser(description='FVG Detector Visualization')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Trading symbol (default: EURUSD)')
    parser.add_argument('--start-date', type=str, default='2024-02-01',
                       help='Start date YYYY-MM-DD (default: 2024-02-01)')
    parser.add_argument('--end-date', type=str, default='2024-03-01',
                       help='End date YYYY-MM-DD (default: 2024-03-01)')
    parser.add_argument('--timeframe', type=str, default='H1',
                       help='Timeframe (default: H1)')
    parser.add_argument('--min-gap-size', type=float, default=0.0,
                       help='Minimum gap size (default: 0.0)')
    parser.add_argument('--no-mitigation', action='store_true',
                       help='Disable mitigation tracking')
    parser.add_argument('--no-shrinking', action='store_true',
                       help='Disable shrinking tracking')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Custom data file path')
    parser.add_argument('--show-original', action='store_true',
                       help='Show original boundaries for shrunk FVGs')

    args = parser.parse_args()

    # Create visualizer
    visualizer = FvgVisualizer(symbol=args.symbol)

    try:
        # Load data
        visualizer.load_data(
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            data_file=args.data_file
        )

        # Detect FVGs
        visualizer.detect_fvgs(
            min_gap_size=args.min_gap_size,
            track_mitigation=not args.no_mitigation,
            track_shrinking=not args.no_shrinking
        )

        # Create and save plot
        fig = visualizer.create_plot(show_original_boundaries=args.show_original)
        visualizer.save_plot(fig)

        # Don't show plot in headless environment to avoid browser errors
        # fig.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())