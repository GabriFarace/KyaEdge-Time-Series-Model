from logging import Logger
import numpy as np

class TimeSeriesFlags:
    def __init__(self, trend : bool, seasonal : bool, noise : bool, spike : bool, inactivity : bool,  max : bool):
        self.trend = trend
        self.seasonal = seasonal
        self.noise = noise
        self.spikes = spike
        self.inactivity = inactivity
        self.max = max

class SeasonalityAttributesProphet:
    def __init__(self, seasonality_frequency : int, paramaters_number : int, coefficients : list[tuple[float, float]]):
        self.seasonality_frequency = seasonality_frequency
        self.parameters_number = paramaters_number
        self.coefficients = coefficients


class TimeSeriesGeneratorProphet:
    ''' Builder of time series using a component approach'''
    def __init__(self):
        self.ts = None

    def build_baseline(self, num_units : int, baseline_value : float):
        ''' Build the baseline component of the time series'''
        # Baseline
        self.ts = np.full(num_units, baseline_value, dtype=float)
        return self

    def build_trend(self, trend_intervals, trend_changes, m_base):
        ''' Build the trend component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        ts_length = self.ts.size
        trend = np.zeros(ts_length)

        # Build the trend iteratively for each trend period (defined by the change point list)
        cumulate_k_change = 0
        current_m = m_base
        for i,trend_interval in enumerate(trend_intervals):
            min_a, min_b = trend_interval[0], trend_interval[1]
            t = np.arange(1, min_b - min_a + 1)
            trend_p = np.zeros(ts_length)

            # Polynomial trend
            cumulate_k_change += trend_changes[i]
            trend_piece = cumulate_k_change * t + current_m
            current_m = trend_piece[-1]

            trend_p[min_a:min_b] = trend_piece
            trend = trend + trend_p
        if trend.size != ts_length:
            raise ValueError('Trend size does not match')
        self.ts = self.ts + trend
        return self

    def build_seasonality(self, seasonality_attributes_list : list[SeasonalityAttributesProphet]):
        ''' Build the seasonal component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')


        ts_length = self.ts.size

        for seasonality_attributes in seasonality_attributes_list:

            freq = seasonality_attributes.seasonality_frequency
            parameters_number = seasonality_attributes.parameters_number
            coefficients = seasonality_attributes.coefficients
            t = np.arange(ts_length)

            fourier_sum = np.sum(
                [coefficients[n][0] * np.cos((2 * np.pi * (n + 1) * t) / freq) + coefficients[n][1] * np.sin((2 * np.pi * (n + 1) * t) / freq)
                 for n in range(parameters_number)], axis=0
            )

            self.ts = self.ts + fourier_sum
        return self

    def build_noise(self, noise_std):
        """Build the noise component of the time series """
        # Noise
        noise = np.random.normal(0, noise_std, self.ts.size)
        self.ts = self.ts + noise
        return self

    def build_holidays(self, holidays_intervals : list, holiday_std):
        ''' Build the spike component of the time series '''

        # Add holidays
        holidays_prior = np.random.normal(0, holiday_std, len(holidays_intervals))
        timestamps = np.arange(self.ts.size)
        holidays = np.zeros(self.ts.size)
        for t in timestamps:
            for i,holiday_interval in enumerate(holidays_intervals):
                if t in holiday_interval:
                    holidays[t] += holidays_prior[i]
        self.ts += holidays
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




class TimeSeriesDirectorProphet:
    def __init__(self, time_series_generator: TimeSeriesGeneratorProphet, logger : Logger, config: dict):
        self.time_series_generator = time_series_generator
        self.config = config
        self.logger = logger

    def _baseline_parameters_generation(self):
        ''' Generate the parameters for the baseline component using the configuration file'''
        baseline_config = self.config["baseline"]
        num_units = np.random.choice(np.arange(1, baseline_config["n_years_max"] + 1)) * 365
        baseline_value = np.random.uniform(baseline_config["baseline_min"], baseline_config["baseline_max"])
        self.logger.info("Baseline")
        self.logger.info(f"Baseline : Units -> {num_units}, Baseline Value -> {baseline_value}")
        return num_units, baseline_value

    def _trend_parameters_generation(self, num_units, baseline_value):
        ''' Generate the parameters for the trend component using the configuration file'''
        trend_config = self.config["trend"]
        num_shifts = np.sum(np.random.choice(np.arange(1,trend_config["max_shift_year"] + 1), num_units // 365) )
        change_points = np.random.choice(np.arange(1, num_units), num_shifts, replace=False)
        change_points = np.sort(change_points)
        change_points = np.concatenate(([0], change_points, [num_units]))
        trend_intervals = [(int(change_points[i]), int(change_points[i + 1])) for i in range(change_points.size - 1)]
        m_base = 0
        trend_changes = []
        self.logger.info("\n\n TREND")
        self.logger.info(f"Trend : Number of shifts ->{num_shifts}")
        current_rate = 0
        for i, trend_interval in enumerate(trend_intervals):
            value_change = np.random.uniform(- trend_config["value_change_ratio"], trend_config["value_change_ratio"]) * baseline_value
            interval_rate = value_change / (trend_interval[1] - trend_interval[0])
            trend_change = interval_rate - current_rate
            current_rate = interval_rate
            if i == 0:
                self.logger.info(
                    f"Trend : Base k Rate -> {interval_rate}, Base m -> {m_base}, Value change -> {value_change}")
            else:
                self.logger.info(
                    f"Trend : Interval -> {trend_interval}, Rate change -> {trend_change}, Value change -> {value_change}")
            trend_changes.append(trend_change)
        return trend_intervals, trend_changes, m_base

    def _seasonal_parameters_generation(self, baseline_value):
        ''' Generate the parameters for the seasonal component using the configuration file'''
        seasonal_config = self.config["seasonal"]
        seasonality_attributes_list = []
        frequencies = seasonal_config["frequencies"]
        self.logger.info("\n\n SEASONALITY")
        for seasonality_frequency in frequencies:
            if np.random.choice([True, False], p=[seasonality_frequency["prob"],1 - seasonality_frequency["prob"]]):
                frequency = seasonality_frequency["value"]
                parameters_number = seasonality_frequency["params_number"]
                coefficients = []
                for i in range(parameters_number):
                    a = np.random.normal(0, seasonality_frequency["coeff_ratio_std"] * baseline_value)
                    b = np.random.normal(0, seasonality_frequency["coeff_ratio_std"] * baseline_value)
                    coefficients.append((a, b))
                self.logger.info(
                    f"Seasonality : Frequency -> {frequency}, N -> {parameters_number} , Coefficients -> {coefficients}")
                seasonality_attributes_list.append(SeasonalityAttributesProphet(frequency, parameters_number, coefficients))
        return seasonality_attributes_list

    def _noise_parameters_generation(self, baseline_value):
        ''' Generate the parameters for the noise component using the configuration file'''
        noise_config = self.config["noise"]
        noise_std = np.random.uniform(0, noise_config["std_max"]) * baseline_value
        self.logger.info("\n\n NOISE")
        self.logger.info(f"Noise : Standard Deviation -> {noise_std}")
        return noise_std

    def _holidays_parameters_generation(self, num_units, baseline_value):
        ''' Generate the parameters for the holidays component using the configuration file'''
        holidays_config = self.config["holidays"]
        holidays_std = np.random.uniform(0, holidays_config["std_max"]) * baseline_value
        self.logger.info("\n\n HOLIDAYS")
        self.logger.info(f"Holidays : Standard Deviation -> {holidays_std}")
        holidays_per_year = np.random.choice(np.arange(0, holidays_config["max_number_of_holidays_year"] + 1))
        holidays_windows = np.random.choice(np.arange(1, holidays_config["holidays_max_window"] + 1), holidays_per_year)
        holidays_timestamps = [np.random.choice(np.arange(holidays_windows[i], 365 - int(holidays_windows[i]))) for i in range(holidays_per_year)]
        yearly_intervals = [[holidays_timestamps[i] + j for j in range(-holidays_windows[i], holidays_windows[i] + 1)] for i in range(holidays_per_year)]

        holidays_intervals = []
        for i in range(len(yearly_intervals)):
            holiday_intervals = []
            for j in range(len(yearly_intervals[i])):
                current = yearly_intervals[i][j]
                while current < num_units:
                    holiday_intervals.append(current)
                    current += 365
            holiday_intervals = sorted(holiday_intervals)
            holidays_intervals.append(holiday_intervals)
            self.logger.info(f"Holidays : Interval -> {holiday_intervals}")
        return holidays_intervals, holidays_std

    def make_ts(self, include_max : bool):
        '''
        Use the builder to make the time series with all the components and generating randomly
        the parameters using the configurations
        '''
        self.time_series_generator.reset()

        # Build the baseline
        num_units, baseline_value = self._baseline_parameters_generation()
        self.time_series_generator.build_baseline(num_units, baseline_value)

        # Build the trend
        self.time_series_generator.build_trend(*self._trend_parameters_generation(num_units, baseline_value))

        # Build the seasonality
        self.time_series_generator.build_seasonality(self._seasonal_parameters_generation(baseline_value))

        # Build the noise
        self.time_series_generator.build_noise(self._noise_parameters_generation(baseline_value))

        # Build the holidays
        self.time_series_generator.build_holidays(*self._holidays_parameters_generation(num_units, baseline_value))

        # Add max
        if include_max:
            self.time_series_generator.build_max()

    def make_ts_conditional(self, ts_flags: TimeSeriesFlags):
        '''
        Use the builder to make the time series with the components specified by ts_flags and generating randomly
        the parameters using the configurations
        '''
        self.time_series_generator.reset()

        # Build the baseline
        num_units, baseline_value = self._baseline_parameters_generation()
        self.time_series_generator.build_baseline(num_units, baseline_value)

        # Build the trend
        if ts_flags.trend:
            self.time_series_generator.build_trend(*self._trend_parameters_generation(num_units, baseline_value))

        # Build the seasonality
        if ts_flags.seasonal:
            self.time_series_generator.build_seasonality(self._seasonal_parameters_generation(baseline_value))

        # Build the noise
        if ts_flags.noise:
            self.time_series_generator.build_noise(self._noise_parameters_generation(baseline_value))

        # Build the holidays
        if ts_flags.spikes:
            self.time_series_generator.build_holidays(*self._holidays_parameters_generation(num_units, baseline_value))
        # Add max
        if ts_flags.max:
            self.time_series_generator.build_max()

    def set_config(self, new_config):
        self.config = new_config



