from logging import Logger
import numpy as np
from enum import Enum




class SeasonalityAttributesProphet:
    def __init__(self, seasonality_frequency : int, seasonality_base_amplitude : float, coefficients):
        self.seasonality_frequency = seasonality_frequency
        self.seasonality_base_amplitude = seasonality_base_amplitude


class TimeSeriesGeneratorProphet:
    ''' Builder of time series using a component approach'''
    def __init__(self):
        self.ts = None

    def build_baseline(self, num_units : int, baseline_value : float):
        ''' Build the baseline component of the time series'''
        # Baseline
        self.ts = np.full(num_units, 0., dtype=float)
        return self

    def build_trend(self, trend_intervals, trend_changes, k_base, m_base):
        ''' Build the trend component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        ts_length = self.ts.size
        trend = np.zeros(ts_length)

        # Build the trend iteratively for each trend period (defined by the change point list)
        for i,trend_interval in enumerate(trend_intervals):
            min_a, min_b = trend_interval[0], trend_interval[1]
            t = np.arange(min_b - min_a)
            trend_p = np.zeros(ts_length)

            # Polynomial trend
            coefficients = [k_base + trend_changes[i], m_base - k_base * min_a]
            trend_piece = np.polyval(coefficients, t)

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
        seasonality = np.zeros(ts_length)

        for seasonality_attributes in seasonality_attributes_list:

            freq = seasonality_attributes.seasonality_frequency
            base_amplitude = seasonality_attributes.seasonality_base_amplitude

            # Handle seasonality intervals
            for start, end in seasonality_intervals:

                # Define seasonality type
                if type == SeasonalityType.SINUSOIDAL:
                    seasonal_component = amplitude * np.sin(2 * np.pi * np.arange(start, end) / freq)
                elif type == SeasonalityType.TRIANGULAR:
                    seasonal_component = amplitude * np.abs(1 - 2 * (np.arange(start, end) % freq) / freq)
                elif type == SeasonalityType.SQUARE:
                    seasonal_component = amplitude * np.sign(np.sin(2 * np.pi * np.arange(start, end) / freq))
                elif type == SeasonalityType.SAWTOOTH:
                    seasonal_component = amplitude * ((np.arange(start, end) % freq) / freq)
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

    def build_spikes(self, spike_prob, spike_min_multiplier, spike_max_multiplier, baseline):
        ''' Build the spike component of the time series '''

        # Add spikes
        for i in range(self.ts.size):
            if np.random.rand() < spike_prob:
                spike_multiplier = np.random.uniform(spike_min_multiplier, spike_max_multiplier)
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
        self.spikes = spike
        self.inactivity = inactivity
        self.max = max


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
        num_shifts = (np.random.choice(np.arange(1,trend_config["max_shift_year"] + 1)) ) * (num_units // 365)
        change_points = np.random.choice(np.arange(1, num_units + 1), num_shifts, replace=False)
        change_points = np.sort(change_points)
        trend_intervals = [(int(change_points[i]), int(change_points[i + 1])) for i in range(change_points.size - 1)]
        trend_attributes_list = []
        self.logger.info("\n\n TREND")
        for trend_interval in trend_intervals:
            trend_type = np.random.choice([TrendType.POLYNOMIAL, TrendType.EXPONENTIAL, TrendType.LOGARITHMIC],
                                          p=trend_config["prob_poly_exp_log"])
            if trend_type == TrendType.POLYNOMIAL:
                degree = np.random.choice(np.arange(len(trend_config["poly_params"]["prob_num_degree"])),
                                          p=trend_config["poly_params"]["prob_num_degree"])
                coeff_0 = 0 if not degree == 0 else np.random.uniform(0.5 * baseline_value, 1.5 * baseline_value)
                coefficients = [coeff_0]
                for i in range(1, degree + 1):
                    coefficients.append(baseline_value * np.random.uniform(
                        trend_config["poly_params"]["multiplicative_coeff_range_ratio"][0],
                        trend_config["poly_params"]["multiplicative_coeff_range_ratio"][1]))
                trend_params = {"coefficients": coefficients[::-1]}
            elif trend_type == TrendType.EXPONENTIAL:
                a = baseline_value * np.random.uniform(
                    trend_config["exp_params"]["multiplicative_coeff_range_ratio"][0],
                    trend_config["exp_params"]["multiplicative_coeff_range_ratio"][1])
                b = baseline_value * np.random.uniform(trend_config["exp_params"]["exp_coeff_range_ratio"][0],
                                                       trend_config["exp_params"]["exp_coeff_range_ratio"][1])
                trend_params = {"a": a, "b": b}
            elif trend_type == TrendType.LOGARITHMIC:
                a = baseline_value * np.random.uniform(
                    trend_config["log_params"]["multiplicative_coeff_range_ratio"][0],
                    trend_config["log_params"]["multiplicative_coeff_range_ratio"][1])
                c = baseline_value * np.random.uniform(trend_config["log_params"]["additive_coeff_range_ratio"][0],
                                                       trend_config["log_params"]["additive_coeff_range_ratio"][1])
                trend_params = {"a": a, "c": c}
            else:
                raise ValueError(f"Unknown trend type: {trend_type}")
            trend_attributes_list.append(TrendAttributes(trend_interval, trend_type, trend_params))
            self.logger.info(f"Trend : Interval -> {trend_interval}, Type -> {trend_type.value},Params -> {trend_params}")
        return trend_attributes_list

    def _seasonal_parameters_generation(self, num_units, baseline_value):
        ''' Generate the parameters for the seasonal component using the configuration file'''
        seasonal_config = self.config["seasonal"]
        seasonality_attributes_list = []
        num_frequencies = np.random.choice(np.arange(1,seasonal_config["max_number_frequencies"] + 1))
        frequencies = np.random.choice(seasonal_config["frequencies"], num_frequencies, replace=False)
        self.logger.info("\n\n SEASONALITY")
        for seasonality_frequency in frequencies:
            all_unit = np.random.choice([True, False], p=seasonal_config["prob_all_partial"])
            seasonality_intervals = []
            if not all_unit:
                seasonality_frequency["duration_list"] = [duration for duration in seasonality_frequency["duration_list"] if duration < num_units]
                if len(seasonality_frequency["duration_list"]) == 0:
                    seasonality_frequency["duration_list"] = [num_units - 1]
                duration = np.random.choice(seasonality_frequency["duration_list"])
                current_point = 0
                go = True
                while (num_units - current_point > duration) and go:
                    starting_point = np.random.choice(np.arange(current_point, num_units - duration))
                    seasonality_intervals.append((starting_point, starting_point + duration))
                    current_point = starting_point + duration
                    go = np.random.choice([True, False])
            else:
                seasonality_intervals = [(0, num_units)]  # Entire time series

            seasonality_type = np.random.choice([SeasonalityType.SINUSOIDAL, SeasonalityType.TRIANGULAR, SeasonalityType.SQUARE, SeasonalityType.SAWTOOTH],
                                        p=seasonal_config["prob_type_si_tr_sq_sa"])
            seasonality_base_amplitude = baseline_value * np.random.uniform(
                        seasonal_config["amplitude_coeff_range_ratio"][0],
                        seasonal_config["amplitude_coeff_range_ratio"][1])

            seasonality_amplitude_pattern = np.random.choice([SeasonalityAmplitudePattern.CONSTANT, SeasonalityAmplitudePattern.INCREASING, SeasonalityAmplitudePattern.DECREASING],
                                        p=seasonal_config["prob_pattern_c_i_d"])

            seasonality_attributes_list.append(SeasonalityAttributes(seasonality_frequency["value"], seasonality_intervals, seasonality_type, seasonality_base_amplitude, seasonality_amplitude_pattern))
            self.logger.info(
                f"Seasonality : Frequency -> {seasonality_frequency}, Intervals -> {seasonality_intervals} , Type -> {seasonality_type.value}, Amplitude -> {seasonality_base_amplitude}, Amplitude Pattern -> {seasonality_amplitude_pattern.value}")
        return seasonality_attributes_list

    def _noise_parameters_generation(self, baseline):
        ''' Generate the parameters for the noise component using the configuration file'''
        noise_config = self.config["noise"]
        noise_std = np.random.uniform(0, noise_config["std_max"])
        baseline_value = baseline * np.random.uniform(
                                        noise_config["baseline_range_ratio"][0],
                                        noise_config["baseline_range_ratio"][1])
        self.logger.info("\n\n NOISE")
        self.logger.info(f"Noise : Standard Deviation -> {noise_std}, Baseline -> {baseline_value}")
        return noise_std, baseline_value

    def _inactivity_parameters_generation(self):
        ''' Generate the parameters for the inactivity component using the configuration file'''
        self.logger.info("\n\n INACTIVITY")
        inactivity_prob = np.random.uniform(0, self.config["inactivity"]["max_prob"])
        self.logger.info(f"Inactivity : Probability -> {inactivity_prob}")
        return inactivity_prob

    def _spikes_parameters_generation(self, baseline):
        ''' Generate the parameters for the spikes component using the configuration file'''
        spikes_config = self.config["spikes"]
        spikes_prob = np.random.uniform(0, spikes_config["max_prob"])
        baseline_value = baseline * np.random.uniform(
                                        spikes_config["baseline_range_ratio"][0],
                                        spikes_config["baseline_range_ratio"][1])
        self.logger.info("\n\n SPIKES")
        self.logger.info(f"Spikes : Probability -> {spikes_prob}, Baseline -> {baseline_value}")
        return spikes_prob, spikes_config["min_range"], spikes_config["max_range"], baseline_value

    def make_ts_conditional(self, ts_flags : TimeSeriesFlags):
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
            self.time_series_generator.build_trend(self._trend_parameters_generation(num_units, baseline_value))

        # Build the seasonality
        if ts_flags.seasonal:
            self.time_series_generator.build_seasonality(self._seasonal_parameters_generation(num_units, baseline_value))

        # Build the noise
        if ts_flags.noise:
            self.time_series_generator.build_noise(*self._noise_parameters_generation(baseline_value))

        # Build the spikes
        if ts_flags.spikes:
            self.time_series_generator.build_spikes(*self._spikes_parameters_generation(baseline_value))

        # Build the inactivity
        if ts_flags.inactivity:
            self.time_series_generator.build_inactivity(self._inactivity_parameters_generation())

        # Add max
        if ts_flags.max:
            self.time_series_generator.build_max()

    def make_ts_all(self):
        '''
        Use the builder to make the time series with all the components and generating randomly
        the parameters using the configurations
        '''
        self.time_series_generator.reset()

        # Build the baseline
        num_units, baseline_value = self._baseline_parameters_generation()
        self.time_series_generator.build_baseline(num_units, baseline_value)

        # Build the trend
        self.time_series_generator.build_trend(self._trend_parameters_generation(num_units, baseline_value))

        # Build the seasonality
        self.time_series_generator.build_seasonality(self._seasonal_parameters_generation(num_units, baseline_value))

        # Build the noise
        self.time_series_generator.build_noise(*self._noise_parameters_generation(baseline_value))

        # Build the spikes
        self.time_series_generator.build_spikes(*self._spikes_parameters_generation(baseline_value))

        # Build the inactivity
        self.time_series_generator.build_inactivity(self._inactivity_parameters_generation())

        # Add max
        self.time_series_generator.build_max()


