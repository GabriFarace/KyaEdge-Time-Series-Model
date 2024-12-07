import numpy as np
import pandas as pd
from enum import Enum
import json

class TrendType(Enum):
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"

class SeasonalityType(Enum):
    SINUSOIDAL = "sinusoidal"
    TRIANGULAR = "triangular"
    SQUARE = "square"
    SAWTOOTH = "sawtooth"
    SPIKE = "spike"

class SeasonalityAmplitudePattern(Enum):
    CONSTANT = "constant"
    INCREASING = "increasing"
    DECREASING = "decreasing"

class TrendAttributes:
    def __init__(self, trend_interval : tuple[int, int], trend_type : TrendType, trend_params : dict):
        self.trend_interval = trend_interval
        self.trend_type = trend_type
        self.trend_params = trend_params


class SeasonalityAttributes:
    def __init__(self, seasonality_frequency : int, seasonality_intervals : list, seasonality_type : SeasonalityType, seasonality_base_amplitude : float, seasonality_amplitude_pattern : SeasonalityAmplitudePattern):
        self.seasonality_intervals = seasonality_intervals,
        self.seasonality_frequency = seasonality_frequency,
        self.seasonality_type = seasonality_type,
        self.seasonality_base_amplitude = seasonality_base_amplitude,
        self.seasonality_amplitude_pattern = seasonality_amplitude_pattern



class TimeSeriesGenerator:
    ''' Builder of time series using a component approach'''
    def __init__(self):
        self.ts = None

    def build_baseline(self, num_units, baseline_value):
        ''' Build the baseline component of the time series'''
        # Baseline
        self.ts = np.full(num_units, baseline_value, dtype=float)
        return self

    def build_trend(self, trend_attributes_list):
        ''' Build the trend component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        ts_length = self.ts.size
        trend = np.zeros(ts_length)

        # Build the trend iteratively for each trend period (defined by the change point list)
        for trend_attributes in trend_attributes_list:
            a, b = trend_attributes.trend_interval[0], trend_attributes.trend_interval[1]
            t = np.arange(b - a)
            trend_p = np.zeros(ts_length)
            trend_type = trend_attributes.trend_type
            trend_params = trend_attributes.trend_params
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
            trend_p[a:b] = trend_piece
            trend = trend + trend_p
        if trend.size != ts_length:
            raise ValueError('Trend size does not match')
        self.ts = self.ts + trend
        return self

    def build_seasonality(self, seasonality_attributes_list):
        ''' Build the seasonal component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')


        ts_length = self.ts.size
        seasonality = np.zeros(ts_length)

        for seasonality_attributes in seasonality_attributes_list:

            freq = seasonality_attributes.seasonality_frequency
            type = seasonality_attributes.seasonality_type
            base_amplitude = seasonality_attributes.seasonality_base_amplitude
            amplitude_pattern = seasonality_attributes.seasonality_amplitude_pattern
            seasonality_intervals = seasonality_attributes.seasonality_intervals

            # Handle seasonality intervals
            if len(seasonality_intervals) == 0:
                seasonality_intervals = [(0, self.ts.size)]  # Entire time series
            for start, end in seasonality_intervals:
                # Define seasonality amplitude
                if amplitude_pattern == SeasonalityAmplitudePattern.CONSTANT:
                    amplitude = base_amplitude
                elif amplitude_pattern == SeasonalityAmplitudePattern.INCREASING:
                    amplitude = np.linspace(0, base_amplitude, end - start)
                elif amplitude_pattern == SeasonalityAmplitudePattern.DECREASING:
                    amplitude = np.linspace(base_amplitude, 0, end - start)
                else:
                    raise ValueError('Amplitude pattern not supported')

                # Define seasonality type
                if type == SeasonalityType.SINUSOIDAL:
                    seasonal_component = amplitude * np.sin(2 * np.pi * np.arange(start, end) / freq)
                elif type == SeasonalityType.TRIANGULAR:
                    seasonal_component = amplitude * np.abs(1 - 2 * (np.arange(start, end) % freq) / freq)
                elif type == SeasonalityType.SQUARE:
                    seasonal_component = amplitude * np.sign(np.sin(2 * np.pi * np.arange(start, end) / freq))
                elif type == SeasonalityType.SAWTOOTH:
                    seasonal_component = amplitude * ((np.arange(start, end) % freq) / freq)
                elif type == SeasonalityType.SPIKE:
                    spike_days = [0, int(freq / 2)]  # Spikes at start and midpoint of the period
                    seasonal_component = amplitude if (np.arange(start, end) % freq) in spike_days else 0
                else:
                    raise ValueError(f"Unknown seasonality type: {type}")
                seasonality[start:end] += seasonal_component
        self.ts = self.ts + seasonality
        return self

    def build_noise(self, noise_std, baseline_value):
        """Build the noise component of the time series """
        # Noise
        noise = np.random.normal(0, noise_std, self.ts.size)
        self.ts = self.ts + noise * baseline_value
        return self

    def build_inactivity(self, inactivity_prob):
        ''' Build the inactivity component of the time series '''
        # Simulate inactivity (set energy to zero randomly)
        for i in range(self.ts.size):
            if np.random.rand() < inactivity_prob:
                self.ts[i] = 0
        return self

    def build_spikes(self, spike_prob, spike_multiplier_range, baseline):
        ''' Build the spike component of the time series '''

        # Add spikes
        for i in range(self.ts.size):
            if np.random.rand() < spike_prob:
                spike_multiplier = np.random.uniform(*spike_multiplier_range)
                self.ts[i] += spike_multiplier * baseline
        return self

    def build_max(self):
        ''' Saturate to 0 the time series values '''
        self.ts = np.maximum(self.ts, 0)
        return self

    def generate(self):
        return self.ts

    def reset(self):
        self.ts = None
        return self

class TimeSeriesFlags:
    def __init__(self, trend : bool, seasonal : bool, noise : bool, spike : bool, inactivity : bool,  max : bool):
        self.trend = trend
        self.seasonal = seasonal
        self.noise = noise
        self.spike = spike
        self.inactivity = inactivity
        self.max = max


class TimeSeriesDirector:
    def __init__(self, time_series_generator: TimeSeriesGenerator):
        self.time_series_generator = time_series_generator
        with open("tsg_config.json", "w") as config_file:
            self.config = json.load(config_file)

    def make_ts(self, ts_flags : TimeSeriesFlags):
        self.time_series_generator.reset()








simulated_data_v3.plot(x='day', y='energy_consumption', title="Advanced Seasonal Behaviors")
