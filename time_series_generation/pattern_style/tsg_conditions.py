from typing import Union

import pandas as pd
import numpy as np
from enum import Enum

from matplotlib import pyplot as plt


class Weekday(Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"

def is_valid_date(date_str):
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d')
        return True
    except ValueError:
        return False

class TimeSeriesGeneratorConditions:
    def __init__(self):
        self.ts = None
        self.min_value = None
        self.max_value = None
        self.history = []

    def build_baseline(self, num_units: int, baseline_value: float, min_value: float, max_value: float, noise_ratio_std=None, start_date=None):
        '''Build the time series by setting all the values to baseline_value'''

        if self.ts is not None:
            raise ValueError('Time series already built')

        # Build baseline
        if baseline_value < 0:
            raise ValueError('baseline value must be greater than zero')
        if num_units <= 0:
            raise ValueError('The series length must be greater than zero')
        if min_value > max_value:
            raise ValueError('Max Value must be greater than or equal to Min value.')

        self.min_value = min_value
        self.max_value = max_value

        time_series = np.full(num_units, baseline_value)

        # Add optionally noise
        if noise_ratio_std is not None:
            noise = np.random.normal(0, noise_ratio_std * baseline_value, time_series.size)
            time_series = time_series + noise

        # Default start_date to tomorrow if not provided
        if start_date is None:
            start_date = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # Validate start_date
            if not is_valid_date(start_date):
                raise ValueError(f"Invalid start_date provided: {start_date}. Expected format: 'YYYY-MM-DD'.")

        # Ensure time_series is a pandas Series for easier handling
        if not isinstance(time_series, pd.Series):
            time_series = pd.Series(time_series)

        # Generate date range starting from the start_date
        date_range = pd.date_range(start=start_date, periods=len(time_series), freq='D')

        # Create the DataFrame for Prophet
        self.ts = pd.DataFrame({
            'ds': date_range,
            'y': time_series.values
        })

        # Add a column for the weekday name
        self.ts['weekday'] = self.ts['ds'].dt.day_name()

        self.ts["y"] = np.clip(self.ts["y"], self.min_value, self.max_value)

        print(self.ts.head())

        return self

    def apply_func_condition(self, condition: Union[Weekday, int], func, start_date=None, end_date=None):
        """
        Apply a function to all elements in the time series that fall on a specific condition (weekday or day of the month)
        and are within the optional date range [from_date, to_date].
        """

        self.history.append(self.ts.copy(deep=True))

        if self.ts is None:
            raise ValueError(f"No time series has been built yet.")


        # Convert from_date and to_date to pandas.Timestamp if provided
        if start_date:
            # Validate start_date
            if not is_valid_date(start_date):
                raise ValueError(f"Invalid start_date provided: {start_date}. Expected format: 'YYYY-MM-DD'.")
            start_date = pd.to_datetime(start_date)
        if end_date:
            # Validate end_date
            if not is_valid_date(end_date):
                raise ValueError(f"Invalid end_date provided: {end_date}. Expected format: 'YYYY-MM-DD'.")
            end_date = pd.to_datetime(end_date)


        # Create the mask
        if isinstance(condition, Weekday):
            # Filter the time series by weekday and date range
            mask = (self.ts['weekday'] == condition.value)
        elif isinstance(condition, int):
            # Filter the time series by day number and date range
            if condition < 1 or condition > 31:
                raise ValueError(f"Invalid day number provided: {condition}. Must be between 1 and 31.")
            mask = (self.ts['ds'].dt.day == condition)
        else:
            raise ValueError(f"Invalid condition provided: {condition}. Expected Weekday or int.")

        if start_date:

            mask &= (self.ts['ds'] >= start_date)
        if end_date:
            mask &= (self.ts['ds'] <= end_date)

        # Apply the function to the filtered rows
        self.ts.loc[mask, 'y'] = self.ts.loc[mask, 'y'].apply(func)


        self.ts["y"] = np.clip(self.ts["y"], self.min_value, self.max_value)


        return self

    def reset(self):
        self.ts = None
        self.history = []

    def undo(self):
        if len(self.history) > 0:
            self.ts = self.history.pop()






