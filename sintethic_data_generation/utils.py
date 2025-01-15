import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

def create_monthly_average_df(curve, start_date):
    """
    Aggregates a Prophet-compatible DataFrame to monthly averages.

    Parameters:
    - prophet_df (pd.DataFrame): A DataFrame with columns 'ds' (dates) and 'y' (values).

    Returns:
    - pd.DataFrame: A DataFrame with monthly 'ds' and average 'y'.
    """
    prophet_df = create_prophet_dataframe(curve, start_date)
    # Ensure 'ds' is a datetime type
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Extract year and month, then group by them to calculate the average 'y'
    monthly_df = (
        prophet_df
        .groupby(prophet_df['ds'].dt.to_period('M'))
        .agg({'y': 'mean'})
        .reset_index()
    )

    # Convert period to string for the 'ds' column (e.g., "January 2022")
    monthly_df['ds'] = monthly_df['ds'].dt.strftime('%B %Y')

    return monthly_df

def compact_into_months(curve, start_date):
    return create_monthly_average_df(curve, start_date)["y"].tolist()

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





def days_between_dates(start_date_str, end_date_str):
    ''' Return the number of days between two dates.'''

    # Convert the start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Calculate the end date by adding 'number_of_months' to the start date
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    # Calculate the difference in days
    delta = end_date - start_date

    return delta.days + 1


def days_between_month(start_date_str, number_of_months):
    '''Function to calculate number of days between start date and start date + number of months'''
    # Convert the start date string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

    # Calculate the end date by adding 'number_of_months' to the start date
    end_date = start_date + relativedelta(months=+number_of_months)

    # Calculate the difference in days
    delta = end_date - start_date

    return delta.days


def months_between_inclusive(start_date):
    """
    Calculates the number of months passed between a start_date and today, inclusive of the start_date and the current month.

    Parameters:
    - start_date (str): The starting date in 'YYYY-MM-DD' format.

    Returns:
    - int: The number of months passed, inclusive of the start_date and the current month.
    """
    # Parse the start_date into a datetime object
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.today()

    # Calculate the difference in months
    difference = relativedelta(today, start_date)
    months_passed = difference.years * 12 + difference.months

    # Include the start_date and current month
    return months_passed + 1
