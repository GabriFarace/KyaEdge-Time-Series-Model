import json

import numpy as np
import pandas as pd
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet

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

def plot_differences_telemetry(true_telemetry, forecasted_telemetry, today, start_date):
    # Get data
    df_true = create_prophet_dataframe(true_telemetry,start_date)
    df_forecast = create_prophet_dataframe(forecasted_telemetry["mean_curve"],start_date)

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_true['ds'].values, df_true['y'], c='#0072B2')
    ax.plot(df_forecast['ds'].values, df_forecast['y'], c='red')
    ax.fill_between(df_forecast['ds'].values, forecasted_telemetry["lower_bound_curve"],
                    forecasted_telemetry['upper_bound_curve'], color='#0072B2',
                    alpha=0.2)
    ax.axvline(x=pd.to_datetime(today), c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('y')
    ax.set_xlabel('ds')
    plt.show()

def create_prophet_dataframe(time_series, start_date=None):
  """
  Converts a time series to a DataFrame compatible with Prophet.

  Parameters:
  - time_series (list or pd.Series): A time series of numeric values to forecast.
  - start_date (str, optional): The starting date for the time series in 'YYYY-MM-DD' format.
    If not provided, it defaults to tomorrow's date.

  Returns:
  - pd.DataFrame: A DataFrame with columns 'ds' (datestamp) and 'y' (values),
    formatted for Prophet.
  """
  # Default start_date to tomorrow if not provided
  if start_date is None:
      start_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

  # Ensure time_series is a pandas Series for easier handling
  if not isinstance(time_series, pd.Series):
      time_series = pd.Series(time_series)

  # Generate date range starting from the start_date
  date_range = pd.date_range(start=start_date, periods=len(time_series), freq='D')

  # Create the DataFrame for Prophet
  prophet_df = pd.DataFrame({
      'ds': date_range,
      'y': time_series.values
  })

  return prophet_df

def get_forecasted_telemetry(telemetry_data, future_periods, sum_maximum, today, start_date=None):
    df = create_prophet_dataframe(telemetry_data, start_date)
    m = Prophet(n_changepoints=200)
    m.add_seasonality(name='monthly', period=30, fourier_order=5)
    m.add_seasonality(name='bimonthly', period=60, fourier_order=7)
    m.fit(df)
    future = m.make_future_dataframe(periods=future_periods)
    forecast = m.predict(future)
    m.plot(forecast)


    # Filter rows where 'ds' > today
    filtered_df = forecast[forecast['ds'] > today]

    # Extract the columns as lists
    yhat_list = np.clip(filtered_df['yhat'], 0, 24).tolist()
    yhat_lower_list = np.clip(filtered_df['yhat_lower'], 0, 24).tolist()
    yhat_upper_list = np.clip(filtered_df['yhat_upper'], 0, 24).tolist()



    def build_sum(telemetry, sum_value):
        '''' Constraint the sum of the time series to be sum_value by truncating to 0 the series from the time step where the sum is reached'''

        telemetry_copy = np.array(telemetry)
        # Step 1: Compute the cumulative sum
        cumulative_sum = np.cumsum(telemetry_copy)

        # Step 2: Find the index where cumulative sum exceeds the limit
        truncation_index = np.argmax(cumulative_sum > sum_value) if np.any(cumulative_sum > sum_value) else -1

        # Step 3: Truncate the series
        if truncation_index != -1:  # Only truncate if the limit is surpassed
            telemetry_copy[truncation_index:] = 0

        return telemetry_copy.tolist()

    return  {
        "lower_bound_curve": build_sum(telemetry_data + yhat_lower_list, sum_maximum),
        "upper_bound_curve": build_sum(telemetry_data + yhat_upper_list, sum_maximum),
        "mean_curve": build_sum(telemetry_data + yhat_list, sum_maximum),
    }

def generate_loop():

    asset_data_generator = AssetDataGenerator()
    telemetry_data_generator = TelemetryDataGeneratorWrapper()
    data = []

    done = False
    while not done:

        asset_data = asset_data_generator.generate_new_asset()

        telemetry_data = telemetry_data_generator.generate_telemetry_data(asset_data)

        # Plot the generated time series
        fig, ax = plt.subplots()
        ax.plot(telemetry_data, label="Generated Time Series")
        ax.set_title("Time Series")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.legend()

        plt.show()

        today = pd.Timestamp.today().strftime('%Y-%m-%d')

        if len(telemetry_data) < days_between_dates(asset_data["start_date"], today) :
            print("NO FORECASTING \n\n")
            number_of_units = len(telemetry_data)
            telemetry_input = {
                "lower_bound_curve": telemetry_data,
                "upper_bound_curve": telemetry_data,
                "mean_curve": telemetry_data
            }
        else:
            print("FORECASTING \n\n")
            number_of_units = days_between_dates(asset_data["start_date"], today)
            future_periods = len(telemetry_data) - number_of_units
            telemetry_input = get_forecasted_telemetry(telemetry_data[:number_of_units], future_periods, asset_data["category_data"]["useful_life_hours"], today, asset_data["start_date"])
            plot_differences_telemetry(telemetry_data, telemetry_input, today, asset_data["start_date"])

        asset_scores = AssetScoresEstimator.get_scores(asset_data, telemetry_input, number_of_units)
        asset_data["scores"] = asset_scores
        asset_data.pop("category_data")
        asset_data.pop("city_data")
        data.append(asset_data)
        with open('data.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        choice = input("Would you like to continue generating? y/n")
        if choice == 'n':
            done = True


if __name__ == '__main__':
    generate_loop()
