from time_series_generation.neural_prophet_style.tsg_neural_prophet import TimeSeriesDirectorNP, TimeSeriesGeneratorNP, ParametersGenerationConfigsNP
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Function to calculate number of days between start date and start date + number of months
def days_between_month(start_date_str, number_of_months):
    # Convert the start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Calculate the end date by adding 'number_of_months' to the start date
    end_date = start_date + relativedelta(months=+number_of_months)

    # Calculate the difference in days
    delta = end_date - start_date

    return delta.days



class TelemetryDataGeneratorWrapper:

    def __init__(self):

        self.time_series_generator = TimeSeriesGeneratorNP()
        self.config = ParametersGenerationConfigsNP()
        self.tsd = TimeSeriesDirectorNP(self.time_series_generator, self.config)


    def generate_telemetry_data(self, asset_data):
        ''' Generate sintethic telemetry data using the generator, assume hours as unit of the telemetry'''
        self.time_series_generator.reset()

        # Build the baseline
        num_units = days_between_month(asset_data["start_date"], asset_data["contract_data"]["contract_months"])
        baseline_value = asset_data["category_data"]["useful_life_hours"] / num_units
        max_value = 24
        sum_value = asset_data["category_data"]["useful_life_hours"]

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