import json
import pandas as pd
from datetime import datetime
import pandas as pd

from sintethic_data_generation.asset_data_generation import AssetDataGenerator
from sintethic_data_generation.estimators import AssetScoresEstimator
from sintethic_data_generation.telemetry_data_generation import TelemetryDataGeneratorWrapper


def days_between_dates(start_date_str, end_date_str):
    # Convert the start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Calculate the end date by adding 'number_of_months' to the start date
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Calculate the difference in days
    delta = end_date - start_date

    return delta.days


def generate_loop():

    asset_data_generator = AssetDataGenerator()
    telemetry_data_generator = TelemetryDataGeneratorWrapper()
    data = []

    done = False
    while not done:

        asset_data = asset_data_generator.generate_new_asset()

        telemetry_data = telemetry_data_generator.generate_telemetry_data(asset_data)

        today = pd.Timestamp.today().strftime('%Y-%m-%d')

        number_of_units = min(days_between_dates(asset_data["start_date"], today), len(telemetry_data))

        # TODO forecasting the telemetry

        telemetry_data = {
            "lower_bound_curve" : telemetry_data,
            "upper_bound_curve" : telemetry_data,
            "mean_curve" : telemetry_data
        }

        asset_scores = AssetScoresEstimator.get_scores(asset_data, telemetry_data, number_of_units)

        asset_data.pop("category_data")
        asset_data.pop("city_data")
        data.append(asset_data)
        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        choice = input("Would you like to continue generating? y/n")
        if choice == 'n':
            done = True

