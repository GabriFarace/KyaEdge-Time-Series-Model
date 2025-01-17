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
        self.sum_value = None


    def build_baseline(self, num_units: int, baseline_value: float, min_value: float, max_value: float, sum_value=None, noise_ratio_std=None, start_date=None):
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
        if sum_value < 0 or sum_value < min_value:
            raise  ValueError("Sum Value cannot be less than 0 or the minimum value")

        self.min_value = min_value
        self.max_value = max_value
        self.sum_value = sum_value

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

        # Consider the interval constraint
        self.ts["y"] = np.clip(self.ts["y"], self.min_value, self.max_value)

        # Consider the sum constraint
        if self.sum_value is not None:
            self._build_sum(self.sum_value)

        return self

    def apply_func_condition(self, condition: Union[Weekday, int, tuple[str, str]], func, start_date=None, end_date=None):
        """
        Apply a function to all elements in the time series that fall on a specific condition (weekday or day of the month)
        and are within the optional date range [from_date, to_date].
        """


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



        # Create the mask # SEASONALITY
        if isinstance(condition, Weekday):
            # Filter the time series by weekday and date range   WEEKLY
            mask = (self.ts['weekday'] == condition.value)


        elif isinstance(condition, int):
            # Filter the time series by month day number and date range MONTHLY
            if condition < 1 or condition > 31:
                raise ValueError(f"Invalid day number provided: {condition}. Must be between 1 and 31.")
            mask = (self.ts['ds'].dt.day == condition)


        elif isinstance(condition, tuple):
            # Filter the time series by considering the months and the range of dates YEARLY
            start, end = condition

            if not is_valid_date(start):
                raise ValueError(f"Invalid start date provided: {start}. Expected format: 'YYYY-MM-DD'.")
            if not is_valid_date(end):
                raise ValueError(f"Invalid end date provided: {end}. Expected format: 'YYYY-MM-DD'.")
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)

            if start.month > end.month:
                raise ValueError(f"The start date month must be less than the end date month.")
            if start.month == end.month and start.day > end.day:
                raise ValueError(f"The start date day must be less than the end date day, when they are the same month.")

            mask = (self.ts['ds'].dt.month >= start.month) & (self.ts['ds'].dt.month <= end.month)
            mask &= ((self.ts['ds'].dt.day >= start.day) | (self.ts['ds'].dt.month != start.month))
            mask &= ((self.ts['ds'].dt.day <= end.day) | (self.ts['ds'].dt.month != end.month))

        else:
            raise ValueError(f"Invalid condition provided: {condition}. Expected Weekday or int.")



        if start_date:

            mask &= (self.ts['ds'] >= start_date)
        if end_date:
            mask &= (self.ts['ds'] <= end_date)

        # Apply the function to the filtered rows
        self.ts.loc[mask, 'y'] = self.ts.loc[mask, 'y'].apply(func)


        # Consider the interval constraint
        self.ts["y"] = np.clip(self.ts["y"], self.min_value, self.max_value)

        # Consider the sum constraint
        if self.sum_value is not None:
            self._build_sum(self.sum_value)

        return self

    def _build_sum(self, sum_value):
        '''' Constraint the sum of the time series to be sum_value by truncating to 0 the series from the time step where the sum is reached'''

        if self.ts is None:
            raise ValueError('Time series has not been built')

        # Step 1: Compute the cumulative sum
        cumulative_sum = np.cumsum(self.ts["y"])

        # Step 2: Find the index where cumulative sum exceeds the limit
        truncation_index = np.argmax(cumulative_sum > sum_value) if np.any(cumulative_sum > sum_value) else -1

        # Step 3: Truncate the series
        if truncation_index != -1:  # Only truncate if the limit is surpassed
            self.ts.loc[truncation_index: , "y"] = 0

        return self

    def reset(self):
        self.ts = None
        self.sum_value = None
        self.max_value = None
        self.min_value  = None




if __name__ == '__main__':
    tsg = TimeSeriesGeneratorConditions()
    tsg.build_baseline(num_units=730, baseline_value=100, min_value=0, max_value=1000, sum_value=60000)
    tsg.apply_func_condition(("2024-02-11","2024-04-15"), lambda x : 0)

    # Plot the generated time series
    fig, ax = plt.subplots()
    ax.plot(tsg.ts["ds"], tsg.ts["y"], label="Generated Time Series")
    ax.set_title("Time Series")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Value")
    ax.legend()

    plt.show()

