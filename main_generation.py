import json
import pandas as pd
import os

from syntethic_data_generation.asset_data_generation import AssetDataGenerator
from syntethic_data_generation.estimators import AssetScoresEstimator, get_forecasted_telemetry, \
    AggregateScoresEstimator
from syntethic_data_generation.pd_date_utils import days_between_dates, compact_into_months, months_between_inclusive
from syntethic_data_generation.plotting_utils import plot_differences_telemetry, plot_differences_telemetry_months, \
    plot_leasing_risk, plot_lower_upper, plot_quality_rating, plot_esg_rating
from syntethic_data_generation.telemetry_data_generation import TelemetryDataGeneratorWrapper
from time_series_generation.tsg_conditions import TimeSeriesGeneratorConditions
from time_series_generation.tsg_neural_prophet import TimeSeriesGeneratorNP


def aggregates_scores_main():
    ''' Read the file data_months and compute the aggregation using the estimators.AggregateScoresEstimator'''
    with open("json_files/data_months.json", "r") as f:
        data = json.load(f)

    os.makedirs("json_files/json_data", exist_ok=True)
    lessors_assets_scores_list = {}

    for asset_data in data:

        lessor_id = asset_data["lessor_id"]
        if lessor_id not in lessors_assets_scores_list.keys():
            lessors_assets_scores_list[lessor_id] = []

        lessors_assets_scores_list[lessor_id].append(asset_data)

    for lessor_id in lessors_assets_scores_list.keys():
        print(len(lessors_assets_scores_list[lessor_id]))
        aggregates_lessor = AggregateScoresEstimator.aggregate_scores_lessor(lessors_assets_scores_list[lessor_id], lessor_id)

        os.makedirs(f"json_files/json_data/lessor{lessor_id}_data/", exist_ok=True)
        with open(f'json_files/json_data/lessor{lessor_id}_data/aggregates_scores_{lessor_id}.json', 'w') as json_file:
            json.dump(aggregates_lessor, json_file, indent=4)

        with open(f'json_files/json_data/lessor{lessor_id}_data/data_{lessor_id}.json', 'w') as json_file:
            json.dump(lessors_assets_scores_list[lessor_id], json_file, indent=4)

def plot_main(number_of_asset, index_asset, name_input):
    ''' Plot the data for a number_of_asset from the data.json file'''
    with open(f"json_files/{name_input}.json", "r") as f:
        data = json.load(f)

    counter = 0

    i = 0
    for asset_data in data:
        i = i + 1
        if i != index_asset:
            continue



        scores = asset_data["scores"]

        plot_differences_telemetry_months(asset_data["true_telemetry"], asset_data["forecasted_telemetry"], asset_data["start_date"])

        plot_leasing_risk(scores["leasing_risk"]["remarketing_value_curve"], scores["leasing_risk"]["residual_debt"], scores["leasing_risk"]["gap_curve"], asset_data["start_date"])
        plot_lower_upper(scores["leasing_risk"]["remarketing_value_curve"], asset_data["start_date"], "Market Value")

        plot_quality_rating(scores["asset_quality_rating"]["quality_rating_curve"], scores["asset_quality_rating"]["operational_use_curve"], asset_data["start_date"])
        plot_lower_upper(scores["asset_quality_rating"]["quality_rating_curve"], asset_data["start_date"], "Quality Value")
        plot_lower_upper(scores["asset_quality_rating"]["operational_use_curve"], asset_data["start_date"],
                         "Operational Use")

        plot_esg_rating(scores["esg_rating"]["footprint_curve"], scores["esg_rating"]["energy_consumed"], asset_data["start_date"])
        plot_lower_upper(scores["esg_rating"]["footprint_curve"], asset_data["start_date"], "Footprint Value")
        plot_lower_upper(scores["esg_rating"]["energy_consumed"], asset_data["start_date"],
                         "Energy Consumed")

        asset_data.pop("scores")
        print(asset_data)
        counter +=1
        if counter >= number_of_asset:
            break

def reduction_monthly(name_output):
    with open(f"json_files/{name_output}.json", "r") as f:
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
    with open(f'json_files/{name_output}_months.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

def generate_loop(num_generation, name_output):
    ''' Main loop that generates sinthetic asset data with daily and monthly granularity'''

    with open(f"json_files/cities_data.json", "r") as f:
        cities_data = json.load(f)
    with open(f"json_files/categories.json", "r") as f:
        categories = json.load(f)
    with open(f"json_files/tsg_config_neural_prophet.json", "r") as f:
        config = json.load(f)

    asset_data_generator = AssetDataGenerator(cities_data=cities_data, categories=categories)

    telemetry_data_generator = TelemetryDataGeneratorWrapper(time_series_generator=TimeSeriesGeneratorNP(), config=config, tsg_conditions=TimeSeriesGeneratorConditions())
    data = []


    for i in range(num_generation):

        asset_data = asset_data_generator.generate_new_asset()

        # todo use a variable to decide which generator
        telemetry_data = telemetry_data_generator.generate_telemetry_data(asset_data)


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

    with open(f'json_files/{name_output}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

    reduction_monthly(name_output)


if __name__ == '__main__':
    generate_loop(num_generation=1, name_output="data_t2")
    #plot_main(1, 1, name_input="data_t2")
    #aggregates_scores_main()
