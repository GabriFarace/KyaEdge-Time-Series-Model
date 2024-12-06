import numpy as np
import pandas as pd
from enum import Enum

class TrendType(Enum):
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"

class TimeSeriesAttributes:
    def __init__(self):
        self.num_days = None,
        self.baseline_value = 100,
        self.trend_change_prob = 0,
        self.max_trend_slope = 0.5,
        self.allow_seasonality = False,
        self.seasonality_intervals = None,
        self.seasonality_frequencies = None,
        self.seasonality_amplitude_pattern = None,
        self.seasonality_base_amplitude = None,
        self.noise_std = 2,
        self.inactivity_prob = 0.05,
        self.spike_prob = 0.02,
        self.spike_multiplier_range = (1.5, 3.0)



class TimeSeriesGenerator:
    ''' Builder of time series using a component approach'''
    def __init__(self):
        self.ts = None

    def build_baseline(self, num_units, baseline_value):
        ''' Build the baseline component of the time series'''
        # Baseline
        self.ts = np.full(num_units, baseline_value, dtype=float)
        return self

    def build_trend(self, trend_change_points_list, trend_type_list, trend_params_list):
        ''' Build the trend component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        trend = np.array([])

        # Build the trend iteratively for each trend period (defined by the change point list)
        for i in range(len(trend_change_points_list) - 1):
            range_length = trend_change_points_list[i+1] - trend_change_points_list[i]
            t = np.arange(range_length)
            trend_piece = np.zeros(range_length)
            trend_type = trend_type_list[i]
            trend_params = trend_params_list[i]
            if trend_type == TrendType.POLYNOMIAL:
                # Polynomial trend
                coefficients = trend_params.get('coefficients', [0.01, -0.1, 2])
                trend_piece = np.polyval(coefficients, t)
            elif trend_type == TrendType.EXPONENTIAL:
                # Exponential trend
                a = trend_params.get('a', 1)
                b = trend_params.get('b', 0.01)
                trend_piece = a * np.exp(b * t)
            elif trend_type == TrendType.LOGARITHMIC:
                # Logarithmic trend
                a = trend_params.get('a', 10)
                c = trend_params.get('c', 1)
                trend_piece = a * np.log(t + 1) + c
            else:
                raise ValueError('Trend type not supported')
            trend = np.concatenate((trend, trend_piece))
        if trend.size != self.ts.size:
            raise ValueError('Trend size does not match')
        self.ts = self.ts + trend
        return self

    def build_seasonality(self, seasonality_params):
        ''' Build the seasonal component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        '''
        seasonality_params = [{"seasonality"},...]
        '''

        '''    # Handle seasonality intervals
        if seasonality_intervals_list is None:
            seasonality_intervals_list = [(0, self.ts.size)]  # Entire time series
        for start, end in seasonality_intervals_list:
            for freq in seasonality_frequencies_list:
                # Calculate amplitude pattern
                if seasonality_amplitude_pattern == 'constant':
                    amplitude = seasonality_base_amplitude
                elif seasonality_amplitude_pattern == 'increasing':
                    amplitude = np.linspace(0, seasonality_base_amplitude, end - start)
                elif seasonality_amplitude_pattern == 'decreasing':
                    amplitude = np.linspace(seasonality_base_amplitude, 0, end - start)
                elif seasonality_amplitude_pattern == 'wave':
                    amplitude = seasonality_base_amplitude * (
                            1 + np.sin(2 * np.pi * np.arange(end - start) / (2 * (end - start)))
                    )

                # Generate seasonality for the interval
                seasonal_component = amplitude * np.sin(2 * np.pi * np.arange(start, end) / freq)
                seasonality[start:end] += seasonal_component

        return self'''

def generate_energy_series_v3(
        num_days=365,
        daily_baseline=100,
        trend_change_prob=0.1,
        max_trend_slope=0.5,
        allow_seasonality=True,
        seasonality_intervals=None,  # Intervals where seasonality is present
        seasonality_frequencies=[7],  # List of frequencies (e.g., 7 days, 30 days)
        seasonality_amplitude_pattern='constant',  # Options: 'constant', 'increasing', 'decreasing', 'wave'
        seasonality_base_amplitude=10,
        noise_std=2,
        inactivity_prob=0.05,
        spike_prob=0.02,
        spike_multiplier_range=(1.5, 3.0)
):
    """
    Generate simulated daily energy consumption with advanced seasonal behaviors.
    """
    # Time index
    t = np.arange(num_days)

    # Baseline
    baseline = np.full(num_days, daily_baseline, dtype=float)

    # Trend with changes
    trend = np.zeros(num_days)
    current_slope = np.random.uniform(-max_trend_slope, max_trend_slope)
    for i in range(1, num_days):
        if np.random.rand() < trend_change_prob:  # Change trend slope
            current_slope = np.random.uniform(-max_trend_slope, max_trend_slope)
        trend[i] = trend[i - 1] + current_slope

    # Seasonality
    seasonality = np.zeros(num_days)
    if allow_seasonality:
        # Handle seasonality intervals
        if seasonality_intervals is None:
            seasonality_intervals = [(0, num_days)]  # Entire time series
        for start, end in seasonality_intervals:
            for freq in seasonality_frequencies:
                # Calculate amplitude pattern
                if seasonality_amplitude_pattern == 'constant':
                    amplitude = seasonality_base_amplitude
                elif seasonality_amplitude_pattern == 'increasing':
                    amplitude = np.linspace(0, seasonality_base_amplitude, end - start)
                elif seasonality_amplitude_pattern == 'decreasing':
                    amplitude = np.linspace(seasonality_base_amplitude, 0, end - start)
                elif seasonality_amplitude_pattern == 'wave':
                    amplitude = seasonality_base_amplitude * (
                            1 + np.sin(2 * np.pi * np.arange(end - start) / (2 * (end - start)))
                    )

                # Generate seasonality for the interval
                seasonal_component = amplitude * np.sin(2 * np.pi * np.arange(start, end) / freq)
                seasonality[start:end] += seasonal_component

    # Noise
    noise = np.random.normal(0, noise_std, num_days)

    # Combine components (baseline, trend, seasonality, noise)
    energy_consumption = baseline + trend + seasonality + noise

    # Simulate inactivity (set energy to zero randomly)
    for i in range(num_days):
        if np.random.rand() < inactivity_prob:
            energy_consumption[i] = 0

    # Add spikes
    for i in range(num_days):
        if np.random.rand() < spike_prob:
            spike_multiplier = np.random.uniform(*spike_multiplier_range)
            energy_consumption[i] += spike_multiplier * daily_baseline

    # Ensure non-negative values
    energy_consumption = np.maximum(energy_consumption, 0)

    # Create a pandas DataFrame
    data = pd.DataFrame({
        'day': t,
        'energy_consumption': energy_consumption
    })

    return data


# Example: Generate and visualize advanced seasonal behaviors
simulated_data_v3 = generate_energy_series_v3(
    num_days=365 * 2,
    daily_baseline=120,
    trend_change_prob=0.05,
    max_trend_slope=0.3,
    allow_seasonality=True,
    seasonality_intervals=[(30, 180), (200, 300)],  # Seasonality active only in these intervals
    seasonality_frequencies=[7, 30],  # Weekly and monthly seasonality
    seasonality_amplitude_pattern='wave',  # Amplitude follows a wave pattern
    seasonality_base_amplitude=20,
    noise_std=5,
    inactivity_prob=0.1,
    spike_prob=0.05,
    spike_multiplier_range=(2, 4)
)

simulated_data_v3.plot(x='day', y='energy_consumption', title="Advanced Seasonal Behaviors")
