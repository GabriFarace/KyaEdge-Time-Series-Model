import json
import pandas as pd
from waitress.adjustments import asset

from sintethic_data_generation.asset_data_generation import AssetDataGenerator
from sintethic_data_generation.estimators import AssetScoresEstimator, get_forecasted_telemetry
from sintethic_data_generation.plotting_data import plot_differences_telemetry
from sintethic_data_generation.telemetry_data_generation import TelemetryDataGeneratorWrapper
from sintethic_data_generation.utils import compact_into_months, months_between_inclusive, days_between_dates


def reduction_monthly(name_output):
    with open(f"{name_output}.json", "r") as f:
        data = json.load(f)

    for asset_data in data:
        scores = asset_data["scores"]

        asset_data["true_telemetry"] = compact_into_months(asset_data["true_telemetry"], asset_data["start_date"])
        asset_data["forecasted_telemetry"] = compact_into_months(asset_data["forecasted_telemetry"], asset_data["start_date"])

        # ASSET QUALITY
        scores["asset_quality_rating"]["quality_rating_curve"]["upper_bound_curve"] = compact_into_months(
            scores["asset_quality_rating"]["quality_rating_curve"]["upper_bound_curve"], asset_data["start_date"])
        scores["asset_quality_rating"]["quality_rating_curve"]["lower_bound_curve"] = compact_into_months(
            scores["asset_quality_rating"]["quality_rating_curve"]["lower_bound_curve"], asset_data["start_date"])
        scores["asset_quality_rating"]["quality_rating_curve"]["mean_curve"] = compact_into_months(
            scores["asset_quality_rating"]["quality_rating_curve"]["mean_curve"], asset_data["start_date"])

        scores["asset_quality_rating"]["operational_use_curve"]["upper_bound_curve"] = compact_into_months(
            scores["asset_quality_rating"]["operational_use_curve"]["upper_bound_curve"], asset_data["start_date"])
        scores["asset_quality_rating"]["operational_use_curve"]["lower_bound_curve"] = compact_into_months(
            scores["asset_quality_rating"]["operational_use_curve"]["lower_bound_curve"], asset_data["start_date"])
        scores["asset_quality_rating"]["operational_use_curve"]["mean_curve"] = compact_into_months(
            scores["asset_quality_rating"]["operational_use_curve"]["mean_curve"], asset_data["start_date"])

        # LEASING RISK
        scores["leasing_risk"]["remarketing_value_curve"]["upper_bound_curve"] = compact_into_months(
            scores["leasing_risk"]["remarketing_value_curve"]["upper_bound_curve"], asset_data["start_date"])
        scores["leasing_risk"]["remarketing_value_curve"]["lower_bound_curve"] = compact_into_months(
            scores["leasing_risk"]["remarketing_value_curve"]["lower_bound_curve"], asset_data["start_date"])
        scores["leasing_risk"]["remarketing_value_curve"]["mean_curve"] = compact_into_months(
            scores["leasing_risk"]["remarketing_value_curve"]["mean_curve"], asset_data["start_date"])

        scores["leasing_risk"]["gap_curve"]["upper_bound_curve"] = compact_into_months(
            scores["leasing_risk"]["gap_curve"]["upper_bound_curve"], asset_data["start_date"])
        scores["leasing_risk"]["gap_curve"]["lower_bound_curve"] = compact_into_months(
            scores["leasing_risk"]["gap_curve"]["lower_bound_curve"], asset_data["start_date"])
        scores["leasing_risk"]["gap_curve"]["mean_curve"] = compact_into_months(
            scores["leasing_risk"]["gap_curve"]["mean_curve"], asset_data["start_date"])

        scores["leasing_risk"]["residual_debt"]["curve"] = compact_into_months(
            scores["leasing_risk"]["residual_debt"]["curve"], asset_data["start_date"])

        # ESG RATING
        scores["esg_rating"]["footprint_curve"]["upper_bound_curve"] = compact_into_months(
            scores["esg_rating"]["footprint_curve"]["upper_bound_curve"], asset_data["start_date"])
        scores["esg_rating"]["footprint_curve"]["lower_bound_curve"] = compact_into_months(
            scores["esg_rating"]["footprint_curve"]["lower_bound_curve"], asset_data["start_date"])
        scores["esg_rating"]["footprint_curve"]["mean_curve"] = compact_into_months(
            scores["esg_rating"]["footprint_curve"]["mean_curve"], asset_data["start_date"])

        scores["esg_rating"]["energy_consumed"]["upper_bound_curve"] = compact_into_months(
            scores["esg_rating"]["energy_consumed"]["upper_bound_curve"], asset_data["start_date"])
        scores["esg_rating"]["energy_consumed"]["lower_bound_curve"] = compact_into_months(
            scores["esg_rating"]["energy_consumed"]["lower_bound_curve"], asset_data["start_date"])
        scores["esg_rating"]["energy_consumed"]["mean_curve"] = compact_into_months(
            scores["esg_rating"]["energy_consumed"]["mean_curve"], asset_data["start_date"])

        scores["number_of_units"] = min(months_between_inclusive(asset_data["start_date"]), len(scores["esg_rating"]["energy_consumed"]["mean_curve"]))
    with open(f'{name_output}_months.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def generate_loop(num_generation, name_output):
    ''' Main loop that generates sinthetic asset data with daily and monthly granularity'''

    asset_data_generator = AssetDataGenerator()
    telemetry_data_generator = TelemetryDataGeneratorWrapper()
    data = []


    for i in range(num_generation):

        asset_data = asset_data_generator.generate_new_asset()

        # todo use a variable to decide which generator
        telemetry_data = telemetry_data_generator.generate_telemetry_data(asset_data)

        '''# Plot the generated time series
        fig, ax = plt.subplots()
        ax.plot(telemetry_data, label="Generated Time Series")
        ax.set_title("Time Series")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.legend()

        plt.show()'''

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
        asset_data.pop("category_data")
        asset_data.pop("city_data")
        data.append(asset_data)

    with open(f'{name_output}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    reduction_monthly(name_output)


if __name__ == '__main__':
    generate_loop(num_generation=1, name_output="data_t2")
