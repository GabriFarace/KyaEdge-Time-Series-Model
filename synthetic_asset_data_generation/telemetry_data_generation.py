from synthetic_asset_data_generation.pd_date_utils import days_between_month
from time_series_generation.tsg_components_style import TimeSeriesGeneratorComponents, ComponentsFlags
from time_series_generation.tsg_functional_style import TimeSeriesGeneratorConditions, Weekday


class TelemetryDataGeneratorWrapper:
    ''' Wrap the neural prophet time series generator and build the telemetry'''

    def __init__(self, time_series_generator_components: TimeSeriesGeneratorComponents, time_series_generator_functional : TimeSeriesGeneratorConditions):

        self.time_series_generator_components = time_series_generator_components
        self.time_series_generator_functional = time_series_generator_functional


    def generate_telemetry_data(self, asset_data, components = bool):
        ''' Generate sintethic telemetry data using the generator, assume hours as unit of the telemetry'''

        # Build the baseline
        num_units = 365 * asset_data["category_data"]["useful_life_years"]
        baseline_value = asset_data["category_data"]["useful_life_hours"] / num_units
        max_value = 24
        sum_value = asset_data["category_data"]["useful_life_hours"]

        num_units = days_between_month(asset_data["start_date"], asset_data["contract_data"]["contract_months"])

        print(f" NUMBER OF DAYS {num_units}, BASELINE VALUE {baseline_value}")

        baseline = {
            "num_units" : num_units,
            "baseline_value" : baseline_value,
            "max_value" : max_value,
            "sum_value" : sum_value
        }


        return self.time_series_generator_components.generate(ComponentsFlags(trend=True, seasonal=True, noise=True, inactivity=False, autoregression=False, interval_constraint=True, sum_constraint=True), baseline=baseline)

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
        self.tsg_conditions.build_baseline(num_units=num_units, baseline_value=baseline_value, min_value=0., max_value=max_value, sum_value=sum_value ,noise_ratio_std=0.001, start_date=asset_data["start_date"])

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

        return self.tsg_conditions.ts["y"].tolist()