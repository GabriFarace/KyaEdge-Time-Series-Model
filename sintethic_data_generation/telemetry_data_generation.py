from sintethic_data_generation.utils import days_between_month
from time_series_generation.neural_prophet_style.tsg_neural_prophet import TimeSeriesDirectorNP, TimeSeriesGeneratorNP, ParametersGenerationConfigsNP
from time_series_generation.pattern_style.tsg_conditions import TimeSeriesGeneratorConditions, Weekday


class TelemetryDataGeneratorWrapper:
    ''' Wrap the neural prophet time series generator and build the telemetry'''

    def __init__(self):

        self.time_series_generator = TimeSeriesGeneratorNP()
        self.config = ParametersGenerationConfigsNP()
        self.tsd = TimeSeriesDirectorNP(self.time_series_generator, self.config)
        self.tsg_conditions = TimeSeriesGeneratorConditions()


    def generate_telemetry_data(self, asset_data):
        ''' Generate sintethic telemetry data using the generator, assume hours as unit of the telemetry'''
        self.time_series_generator.reset()

        # Build the baseline
        num_units = 365 * asset_data["category_data"]["useful_life_years"]
        baseline_value = asset_data["category_data"]["useful_life_hours"] / num_units
        max_value = 24
        sum_value = asset_data["category_data"]["useful_life_hours"]

        num_units = days_between_month(asset_data["start_date"], asset_data["contract_data"]["contract_months"])

        print(f" NUMBER OF DAYS {num_units}, BASELINE VALUE {baseline_value}")

        # Build the baseline
        self.time_series_generator.build_baseline(num_units, baseline_value)

        # Build the trend
        self.time_series_generator.build_trend(*self.tsd._trend_parameters_generation(num_units, baseline_value))

        # Build the seasonality
        self.time_series_generator.build_seasonality(self.tsd._seasonal_parameters_generation(baseline_value))

        # Build the noise
        self.time_series_generator.build_noise(self.tsd._noise_parameters_generation(baseline_value))

        # Normalize (constraint of the max)
        self.time_series_generator.build_min_max(max_value, baseline_value)

        # Add the constraints of the sum
        self.time_series_generator.build_sum(sum_value)

        return self.time_series_generator.generate().tolist()

    def generate_telemetry_data_c(self, asset_data):
        ''' Generate sintethic telemetry data using the generator, assume hours as unit of the telemetry'''
        self.tsg_conditions.reset()

        # Build the baseline
        num_units = 365 * asset_data["category_data"]["useful_life_years"]
        baseline_value = int(asset_data["category_data"]["useful_life_hours"] / num_units)
        max_value = 24
        sum_value = asset_data["category_data"]["useful_life_hours"]

        num_units = days_between_month(asset_data["start_date"], asset_data["contract_data"]["contract_months"])

        print(f" NUMBER OF DAYS {num_units}, BASELINE VALUE {baseline_value}")

        # Build the baseline
        self.tsg_conditions.build_baseline(num_units, baseline_value, 0., max_value, 0.001, asset_data["start_date"])

        # Build the conditions
        self.tsg_conditions.apply_func_condition(24, lambda x : x * 1.2)
        self.tsg_conditions.apply_func_condition(25, lambda x : x * 1.2)
        self.tsg_conditions.apply_func_condition(26, lambda x : x * 1.2)
        self.tsg_conditions.apply_func_condition(27, lambda x : x * 1.2)

        # Build the conditions
        self.tsg_conditions.apply_func_condition(1, lambda x : x * 0.7)
        self.tsg_conditions.apply_func_condition(2, lambda x : x * 0.7)
        self.tsg_conditions.apply_func_condition(3, lambda x : x * 0.7)
        self.tsg_conditions.apply_func_condition(4, lambda x : x * 0.7)

        # Build the conditions
        self.tsg_conditions.apply_func_condition(Weekday.SUNDAY, lambda x : 0)

        # Build the conditions
        self.tsg_conditions.apply_func_condition(Weekday.SATURDAY, lambda x : x/2)


        # todo also for the other generator self.time_series_generator.build_sum(sum_value)

        return self.tsg_conditions.ts["y"].tolist()