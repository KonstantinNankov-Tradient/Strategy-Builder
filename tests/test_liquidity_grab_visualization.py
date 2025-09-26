"""
Test script for Liquidity Grab Detector visualization.

This script tests the LiquidityGrabDetector with EURUSD 1H data for 1 month,
creating a comprehensive plot showing session highs/lows, liquidity grab lines,
and grab markers for validation purposes.
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
from indicators.liquidity_grab_detector import LiquidityGrabDetector


class LiquidityGrabVisualizer:
    """
    Visualizer for liquidity grab detection results.

    Creates comprehensive plots showing session levels, liquidity grabs,
    and all related market structure for validation.
    """

    def __init__(self, symbol: str = 'EURUSD'):
        self.symbol = symbol
        self.data_loader = DataLoader()
        self.detector = None
        self.data = None
        self.session_levels = []
        self.liquidity_grabs = []

    def load_data(self, start_date: str = '2024-01-01', end_date: str = '2024-02-01',
                  timeframe: str = 'H1', data_file: str = None) -> pd.DataFrame:
        """
        Load EURUSD data for the specified period.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            timeframe: Timeframe for analysis

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
        Setup the liquidity grab detector with configuration.

        Args:
            config: Configuration dictionary for the detector
        """
        default_config = {
            'symbol': self.symbol,
            'enable_wick_extension_filter': True,
            'min_wick_extension_pips': 3.0,
            'detect_same_session': True
        }

        if config:
            default_config.update(config)

        self.detector = LiquidityGrabDetector(
            name='liquidity_grab_test',
            config=default_config
        )

        print(f"🔧 Setup detector with config: {default_config}")

    def run_analysis(self) -> None:
        """
        Run the liquidity grab analysis on the loaded data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if self.detector is None:
            raise ValueError("Detector not setup. Call setup_detector() first.")

        print("🔍 Running liquidity grab analysis...")

        # Get session levels directly from detector's internal method
        data_with_datetime_index = self.data.set_index('datetime')
        self.session_levels = self.detector._get_session_ranges(data_with_datetime_index)

        print(f"📊 Found {len(self.session_levels)} session levels")

        # Simulate running detector on each candle to find grabs
        lookback = self.detector.get_lookback_period()

        for i in range(lookback, len(self.data)):
            # Get data window for current candle
            window_data = self.data.iloc[max(0, i - lookback):i + 1].copy()

            # Run detector
            detection = self.detector.check(window_data, i)

            if detection:
                self.liquidity_grabs.append({
                    'timestamp': detection.timestamp,
                    'candle_index': detection.candle_index,
                    'price': detection.price,
                    'direction': detection.direction,
                    'metadata': detection.metadata
                })

        print(f"🎯 Detected {len(self.liquidity_grabs)} liquidity grabs")

    def create_visualization(self) -> go.Figure:
        """
        Create comprehensive visualization of liquidity grab analysis.

        Returns:
            Plotly figure with candlesticks, session levels, and grabs
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

        # Add session levels
        self._add_session_levels(fig)

        # Add liquidity grab lines and markers
        self._add_liquidity_grabs(fig)

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

    def _add_session_levels(self, fig: go.Figure) -> None:
        """Add session high/low markers and session boundaries."""
        # Session colors
        session_colors = {
            'asian': 'rgba(255, 165, 0, 0.8)',     # Orange
            'european': 'rgba(0, 255, 0, 0.8)',    # Green
            'ny': 'rgba(0, 0, 255, 0.8)'           # Blue
        }

        sessions_seen = set()

        for level_info in self.session_levels:
            session = level_info['session']
            level_type = level_info['type']
            color = session_colors.get(session, 'rgba(128, 128, 128, 0.8)')

            # Add session level marker
            if level_type == 'high':
                marker_symbol = "triangle-down"
                marker_name = f"{session.title()} High"
            else:
                marker_symbol = "triangle-up"
                marker_name = f"{session.title()} Low"

            fig.add_trace(go.Scatter(
                x=[level_info['time']],
                y=[level_info['level']],
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=12,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=marker_name,
                showlegend=True,
                legendgroup=f"{session}_levels"
            ))

            # Session boundary lines and annotations removed as requested

    def _add_liquidity_grabs(self, fig: go.Figure) -> None:
        """Add liquidity grab lines and markers."""
        grab_colors = {
            'high_grab': '#ff6b6b',    # Red for high grabs
            'low_grab': '#4ecdc4'      # Teal for low grabs
        }

        for grab in self.liquidity_grabs:
            metadata = grab['metadata']
            grab_type = metadata.get('grab_type', 'unknown_grab')
            color = grab_colors.get(grab_type, '#888888')

            level_time = metadata.get('level_time')
            grab_time = grab['timestamp']
            level_price = grab['price']

            # Add horizontal line from level creation to grab
            fig.add_trace(go.Scatter(
                x=[level_time, grab_time],
                y=[level_price, level_price],
                mode="lines",
                line=dict(color=color, width=2, dash='solid'),
                opacity=0.8,
                name=f"Liquidity Grab ({grab_type.replace('_', ' ').title()})",
                showlegend=True,
                legendgroup="liquidity_grabs"
            ))

            # Add grab marker
            if grab_type == 'high_grab':
                marker_symbol = "triangle-down"
            else:
                marker_symbol = "triangle-up"

            fig.add_trace(go.Scatter(
                x=[grab_time],
                y=[level_price],
                mode="markers",
                marker=dict(
                    symbol=marker_symbol,
                    size=14,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=f"Grab Point",
                showlegend=False
            ))

            # Add session relationship annotation (F/N/S only)
            session_relationship = metadata.get('session_relationship', 'following')

            # Map session relationship to display letter
            relationship_map = {
                'same': 'S',
                'next': 'N',
                'following': 'F'
            }
            relationship_letter = relationship_map.get(session_relationship, 'F')

            fig.add_annotation(
                x=grab_time,
                y=level_price,
                text=relationship_letter,
                showarrow=False,
                font=dict(size=12, color="white", family="Arial Black"),
                bgcolor=color,
                bordercolor="white",
                borderwidth=2,
                yshift=20 if grab_type == 'low_grab' else -20,
                xanchor="center",
                yanchor="middle"
            )

    def show_plot(self) -> None:
        """Display the liquidity grab analysis plot."""
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
            filename = f"liquidity_grab_analysis_{self.symbol}_{timeframe}{date_range}_{timestamp}.html"

        # Full file path
        file_path = output_dir / filename

        fig = self.create_visualization()
        fig.write_html(file_path)
        print(f"💾 Plot saved as: {file_path}")


def main():
    """Main function to run the liquidity grab visualization test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Liquidity Grab Detector Visualization Test')
    parser.add_argument('--timeframe', default='H1',
                       choices=['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
                       help='Timeframe for analysis (default: H1)')
    parser.add_argument('--start-date', default='2024-01-01',
                       help='Start date in YYYY-MM-DD format (default: 2024-01-01)')
    parser.add_argument('--end-date', default='2024-02-01',
                       help='End date in YYYY-MM-DD format (default: 2024-02-01)')
    parser.add_argument('--symbol', default='EURUSD',
                       help='Currency pair symbol (default: EURUSD)')
    parser.add_argument('--data-file',
                       help='Optional: specific data file path')
    parser.add_argument('--min-wick-pips', type=float, default=3.0,
                       help='Minimum wick extension in pips (default: 3.0)')

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
        print("🚀 Starting Liquidity Grab Visualization Test")
        print("=" * 50)
        print(f"📊 Configuration:")
        print(f"   • Symbol: {args.symbol}")
        print(f"   • Timeframe: {args.timeframe}")
        print(f"   • Date Range: {args.start_date} to {args.end_date}")
        print(f"   • Min Wick Pips: {args.min_wick_pips}")
        if args.data_file:
            print(f"   • Data File: {args.data_file}")

        # Initialize visualizer
        visualizer = LiquidityGrabVisualizer(args.symbol)

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
            'enable_wick_extension_filter': True,
            'min_wick_extension_pips': args.min_wick_pips,
            'detect_same_session': True
        }
        visualizer.setup_detector(config)

        # Run analysis
        visualizer.run_analysis()

        # Show results
        print("\n📊 Analysis Results:")
        print(f"   • Session Levels: {len(visualizer.session_levels)}")
        print(f"   • Liquidity Grabs: {len(visualizer.liquidity_grabs)}")

        if visualizer.liquidity_grabs:
            print("\n🎯 Detected Grabs:")
            for i, grab in enumerate(visualizer.liquidity_grabs):
                metadata = grab['metadata']
                print(f"   {i+1}. {grab['timestamp']} | "
                      f"{metadata.get('grab_type', 'unknown').replace('_', ' ').title()} | "
                      f"Level: {grab['price']:.5f} | "
                      f"Session: {metadata.get('session', 'unknown').title()} | "
                      f"Wick: {metadata.get('wick_extension_pips', 0):.1f} pips")

        # Create and save visualization
        print("\n📈 Creating visualization...")

        # Save to outputs/plots directory with parameters
        visualizer.save_plot(
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Alternative: Save to custom location
        # visualizer.save_plot(output_dir="/path/to/custom/directory")

        # Don't show plot in headless environment
        # visualizer.show_plot()

        print("\n✅ Liquidity Grab visualization test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        raise


if __name__ == "__main__":
    main()