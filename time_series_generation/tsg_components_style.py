import logging
import numpy as np


# Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format for log messages
    datefmt='%Y-%m-%d %H:%M:%S'  # Format for timestamps
)
logger = logging.getLogger("Tsg Components Logger")

class ComponentsFlags:
    def __init__(self, trend : bool, seasonal : bool, noise : bool, inactivity : bool, autoregression : bool, interval_constraint : bool, sum_constraint : bool):
        self.trend = trend
        self.seasonal = seasonal
        self.noise = noise
        self.inactivity = inactivity
        self.autoregression = autoregression
        self.interval_constraint = interval_constraint
        self.sum_constraint = sum_constraint

class SeasonalityAttributes:
    def __init__(self, seasonality_frequency : int, paramaters_number : int, coefficients : list[tuple[float, float]]):
        self.seasonality_frequency = seasonality_frequency
        self.parameters_number = paramaters_number
        self.coefficients = coefficients

class TimeSeriesComponents:
    ''' Builder of time series using a component approach'''
    def __init__(self):
        self.ts = None
        self.components = {}

    # TODO fix the component returned by the generator (normalization is needed when scaling)

    def build_baseline(self, num_units : int, baseline_value : float):
        ''' Build the baseline component of the time series'''
        # Baseline
        self.ts = np.full(num_units, baseline_value, dtype=float)
        self.components["baseline"] = self.ts.copy()
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
        self.components['trend'] = trend
        self.ts = self.ts + trend
        return self

    def build_seasonality(self, seasonality_attributes_list : list[SeasonalityAttributes]):
        ''' Build the seasonal component of the time series '''
        # Trend with changes
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')


        ts_length = self.ts.size
        self.components["seasonality"] = {}
        for seasonality_attributes in seasonality_attributes_list:

            freq = seasonality_attributes.seasonality_frequency
            parameters_number = seasonality_attributes.parameters_number
            coefficients = seasonality_attributes.coefficients
            t = np.arange(ts_length)

            fourier_sum = np.sum(
                [coefficients[n][0] * np.cos((2 * np.pi * (n + 1) * t) / freq) + coefficients[n][1] * np.sin((2 * np.pi * (n + 1) * t) / freq)
                 for n in range(parameters_number)], axis=0
            )
            self.components["seasonality"][freq] =  fourier_sum[:freq]
            self.ts = self.ts + fourier_sum
        return self

    def build_noise(self, noise_std):
        """Build the noise component of the time series """


        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        # Noise
        noise = np.random.normal(0, noise_std, self.ts.size)
        self.components['noise'] = noise
        self.ts = self.ts + noise
        return self

    def build_inactivity(self, inactivity_prob):
        ''' Build the inactivity component of the time series '''

        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        # Simulate inactivity (set energy to zero randomly)
        for i in range(self.ts.size):
            if np.random.rand() < inactivity_prob:
                self.ts[i] = 0
        return self

    def build_autoregression(self, number_of_lags : int, coefficients : list[float]):
        ''' Build the autoregressive component of the time series. AR order = number_of_lags, coefficients = coefficients'''
        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        if number_of_lags != len(coefficients):
            raise ValueError("The number of coefficients does not match with the number of lags")

        coefficients_np_array = np.array(coefficients)

        autoregressive_component = [0] * number_of_lags

        for i in range(number_of_lags, self.ts.size):
            autoregressive_component.append(np.dot(self.ts[i - number_of_lags : i], coefficients_np_array))

        self.components['autoregressive_component'] = autoregressive_component
        self.ts = self.ts + autoregressive_component
        return self

    def build_min_max(self, max_value, normalization_value, percentage=0.3):
        ''' Scale the time series in the interval [0, max_value] and then apply mean-scaling with value normalization_value plus/minus percentage'''

        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        # First min-max scaling
        old_min = self.ts.min()
        old_max = self.ts.max()
        self.ts = (self.ts - old_min) * (max_value - 0) / (old_max - old_min)

        # Normalization by subtracting the mean and adding normalization_value
        min_mean = max(0, normalization_value * (1 - percentage))
        max_mean = min(normalization_value * (1 + percentage), max_value)
        normalization = np.random.uniform(min_mean, max_mean)

        mean = self.ts.mean()
        self.ts = self.ts - mean + normalization

        # Consider also modifying the standard deviation for the limits
        current_std = self.ts.std()
        current_ratio_std = 1
        final_ts = self.ts
        while current_ratio_std > 0.2 and (final_ts.min() < 0 or final_ts.max() > max_value):
            final_ts = self.ts * (current_std * current_ratio_std/current_std)
            final_ts = final_ts - final_ts.mean() + normalization
            current_ratio_std -= 0.1
        self.ts = final_ts


        # Truncate the values to the specified range [truncate_min, truncate_max]
        self.ts = np.clip(self.ts, 0, max_value)

        return self

    def build_sum(self, sum_value):
        '''' Constraint the sum of the time series to be sum_value by truncating to 0 the series from the time step where the sum is reached'''

        if self.ts is None:
            raise ValueError('Time series baseline has not been built')

        # Step 1: Compute the cumulative sum
        cumulative_sum = np.cumsum(self.ts)

        # Step 2: Find the index where cumulative sum exceeds the limit
        truncation_index = np.argmax(cumulative_sum > sum_value) if np.any(cumulative_sum > sum_value) else -1

        # Step 3: Truncate the series
        if truncation_index != -1:  # Only truncate if the limit is surpassed
            self.ts[truncation_index:] = 0

        return self

    def get_series(self):
        return self.ts

    def get_components(self):
        return self.components

    def reset(self):
        self.ts = None
        self.components = {}
        return self

class TimeSeriesGeneratorComponents:
    def __init__(self, config: dict):
        self.time_series_components = TimeSeriesComponents()
        self.config = config

    def _baseline_parameters_generation(self):
        ''' Generate the parameters for the baseline component using the configuration file'''
        baseline_config = self.config["baseline"]
        num_units = np.random.choice(np.arange(baseline_config["n_units_min"], baseline_config["n_units_max"] + 1))
        if baseline_config["unit_is_energy"]:
            baseline_value = np.random.uniform(baseline_config["baseline_min"], baseline_config["baseline_max"])
            daily_hours = np.random.uniform(0, 24)
            max_value = (baseline_value / daily_hours) * 24
            sum_value = baseline_value * num_units
        else:
            baseline_value = np.random.uniform(0, 24)
            max_value = 24
            sum_value = baseline_value * num_units
        logger.info("Baseline")
        logger.info(f"Baseline : Units -> {num_units}, Baseline Value -> {baseline_value}, Max Value -> {max_value}, Sum Value -> {sum_value}")
        return num_units, baseline_value, max_value, sum_value

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
        logger.info("\n\n TREND")
        logger.info(f"Trend : Number of shifts ->{num_shifts}")
        current_rate = 0
        for i, trend_interval in enumerate(trend_intervals):
            value_change = np.random.uniform(- trend_config["value_change_ratio"], trend_config["value_change_ratio"]) * baseline_value
            interval_rate = value_change / (trend_interval[1] - trend_interval[0])
            trend_change = interval_rate - current_rate
            current_rate = interval_rate
            if i == 0:
                logger.info(
                    f"Trend : Base k Rate -> {interval_rate}, Base m -> {m_base}, Value change -> {value_change}")
            else:
                logger.info(
                    f"Trend : Interval -> {trend_interval}, Rate change -> {trend_change}, Value change -> {value_change}")
            trend_changes.append(trend_change)
        return trend_intervals, trend_changes, m_base

    def _seasonal_parameters_generation(self, baseline_value):
        ''' Generate the parameters for the seasonal component using the configuration file'''
        seasonal_config = self.config["seasonal"]
        seasonality_attributes_list = []
        frequencies = seasonal_config["frequencies"]
        logger.info("\n\n SEASONALITY")
        for seasonality_frequency in frequencies:
            if np.random.choice([True, False], p=[seasonality_frequency["prob"],1 - seasonality_frequency["prob"]]):
                frequency = seasonality_frequency["value"]
                parameters_number = seasonality_frequency["params_number"]
                coefficients = []
                for i in range(parameters_number):
                    a = np.random.normal(0, seasonality_frequency["coeff_ratio_std"] * baseline_value)
                    b = np.random.normal(0, seasonality_frequency["coeff_ratio_std"] * baseline_value)
                    coefficients.append((a, b))
                logger.info(
                    f"Seasonality : Frequency -> {frequency}, N -> {parameters_number} , Coefficients -> {coefficients}")
                seasonality_attributes_list.append(SeasonalityAttributes(frequency, parameters_number, coefficients))
        return seasonality_attributes_list

    def _noise_parameters_generation(self, baseline_value):
        ''' Generate the parameters for the noise component using the configuration file'''
        noise_config = self.config["noise"]
        noise_std = np.random.uniform(0, noise_config["std_max"]) * baseline_value
        logger.info("\n\n NOISE")
        logger.info(f"Noise : Standard Deviation -> {noise_std}")
        return noise_std

    def _inactivity_parameters_generation(self):
        ''' Generate the parameters for the noise component using the configuration file'''
        inactivity_config = self.config["inactivity"]
        inactivity_prob = np.random.uniform(0, inactivity_config["max_prob"])
        logger.info("\n\n INACTIVITY")
        logger.info(f"Inactivity : Probability -> {inactivity_prob}")
        return inactivity_prob

    def _autoregression_parameters_generation(self):
        ''' Generate the parameters for the autoregressive component using the configuration file'''
        autoregressive_config = self.config["autoregression"]
        number_of_lags = np.random.randint(0, autoregressive_config["max_number_of_lags"] + 1 )
        coefficients = []
        for i in range(number_of_lags):
            coefficients.append(np.random.normal(0, autoregressive_config["max_coefficient"]))
        logger.info("\n\n AUTOREGRESSION")
        logger.info(f"Autoregression : Number of lags -> {number_of_lags}, Coefficients -> {coefficients}")
        return number_of_lags, coefficients

    def generate(self, ts_flags: ComponentsFlags, baseline=None):
        '''
        Use the builder to make the time series with the components specified by ts_flags and generating randomly
        the parameters using the configurations
        '''
        self.time_series_components.reset()

        # Build the baseline
        if baseline is None:
            num_units, baseline_value, max_value, sum_value = self._baseline_parameters_generation()
        else:
            num_units = baseline["num_units"]
            baseline_value = baseline["baseline_value"]
            max_value = baseline["max_value"]
            sum_value = baseline["sum_value"]


        self.time_series_components.build_baseline(num_units, baseline_value)

        # Build the trend
        if ts_flags.trend:
            self.time_series_components.build_trend(*self._trend_parameters_generation(num_units, baseline_value))

        # Build the seasonality
        if ts_flags.seasonal:
            self.time_series_components.build_seasonality(self._seasonal_parameters_generation(baseline_value))

        # Build the noise
        if ts_flags.noise:
            self.time_series_components.build_noise(self._noise_parameters_generation(baseline_value))

        # Build the autoregression component
        if ts_flags.autoregression:
            self.time_series_components.build_autoregression(*self._autoregression_parameters_generation())

        # Add the scale of min and max
        if ts_flags.interval_constraint:
            self.time_series_components.build_min_max(max_value, baseline_value)

        # Add inactivities
        if ts_flags.inactivity:
            self.time_series_components.build_inactivity(self._inactivity_parameters_generation())

        # Add the constraints of the sum
        if ts_flags.sum_constraint:
            self.time_series_components.build_sum(sum_value)

        return self.time_series_components.get_series().tolist()


