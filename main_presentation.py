from synthetic_asset_data_generation.asset_data_generation import AssetDataGenerator
from synthetic_asset_data_generation.estimators import AssetScoresEstimator, get_forecasted_telemetry
import json
import pandas as pd

from synthetic_asset_data_generation.pd_date_utils import days_between_dates, days_between_month
from synthetic_asset_data_generation.plotting_utils import plot_differences_telemetry
from time_series_generation.tsg_components_style import TimeSeriesGeneratorComponents
from time_series_generation.tsg_functional_style import TimeSeriesGeneratorFunctional, TimeSeriesFunctional, Weekday


def generate_data(name_output):
    ''' Main loop that generates sinthetic asset data with daily and monthly granularity'''

    with open(f"json_files/cities_data.json", "r") as f:
        cities_data = json.load(f)

    categories = [{
        "id": "1",
        "name": "Professional Multi Function Printer",
        "cost": 5000,
        "useful_life_years": 8,
        "useful_life_hours": 3000,
        "power_kw": 1.5,
        "residual_value": 0.
    }]

    with open(f"json_files/config_generator_functional_style.json", "r") as f:
        config = json.load(f)
    with open(f"json_files/config_generator_components_style.json", "r") as f:
        config2 = json.load(f)


    asset_data_generator = AssetDataGenerator(cities_data=cities_data, categories=categories, time_series_generator_functional=TimeSeriesGeneratorFunctional(config=config), time_series_generator_components=TimeSeriesGeneratorComponents(config=config2))


    asset_data = asset_data_generator.generate_new_asset(components=True)

    telemetry_generator = TimeSeriesFunctional()
    num_units = 365 * asset_data["category_data"]["useful_life_years"]
    baseline_value = int(asset_data["category_data"]["useful_life_hours"] / num_units)
    max_value = 24
    sum_value = asset_data["category_data"]["useful_life_hours"]

    num_units = days_between_month(asset_data["start_date"], asset_data["contract_data"]["contract_months"])

    print(f" NUMBER OF DAYS {num_units}, BASELINE VALUE {baseline_value}")

    telemetry_generator.build_baseline(num_units=num_units, baseline_value=baseline_value, max_value=max_value, min_value=0, sum_value=sum_value, noise_ratio_std=0.01, start_date=asset_data["start_date"])

    for i in range(24,32):
        telemetry_generator.apply_func_condition(i, lambda x : x * 1.5)

    telemetry_generator.apply_func_condition(("2020-08-01", "2020-08-31"), lambda x : x * 0.3)

    telemetry_generator.apply_func_condition(("2020-12-01", "2020-12-15"), lambda x : x * 1.5)

    telemetry_generator.apply_func_condition(("2020-12-16", "2020-12-31"), lambda x : 0.)

    telemetry_generator.apply_func_condition(Weekday.SATURDAY, lambda x : 0.)

    telemetry_generator.apply_func_condition(Weekday.SUNDAY, lambda x : 0.)

    telemetry_data = telemetry_generator.ts["y"].tolist()

    today = pd.Timestamp.today().strftime('%Y-%m-%d')

    number_of_units = min(len(telemetry_data), days_between_dates(asset_data["start_date"], today))
    asset_data["asset_specific_expected_yearly_usage"] = AssetScoresEstimator.get_asset_expected_usage(asset_data, telemetry_data[:number_of_units])
    asset_data["standard_asset_average_expected_yearly_usage"] = AssetScoresEstimator.get_standard_average_asset_expected_usage(asset_data)

    if len(telemetry_data) == number_of_units :
        print("NO FORECASTING \n\n")
        telemetry_input = {
            "lower_bound_curve": telemetry_data,
            "upper_bound_curve": telemetry_data,
            "mean_curve": telemetry_data
        }
    else:
        print("FORECASTING \n\n")
        future_periods = len(telemetry_data) - number_of_units
        telemetry_input = get_forecasted_telemetry(telemetry_data[:number_of_units], future_periods, asset_data["category_data"]["useful_life_hours"], today, asset_data["start_date"])
        plot_differences_telemetry(telemetry_data, telemetry_input, today, asset_data["start_date"])

    asset_data["true_telemetry"] = telemetry_data
    asset_data["forecasted_telemetry"] = telemetry_input["mean_curve"]

    asset_data["scores"] = AssetScoresEstimator.get_scores(asset_data, telemetry_input, number_of_units)
    asset_data.pop("telemetry")


    with open(f'json_files/{name_output}.json', 'w') as json_file:
        json.dump(asset_data, json_file, indent=4)






if __name__ == '__main__':
    generate_data(name_output="data_presentation")

