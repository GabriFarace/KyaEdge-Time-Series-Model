from typing import Union

import pandas as pd
import numpy as np
from enum import Enum
import json

from syntethic_data_generation.asset_data_generation import AssetDataGenerator
from syntethic_data_generation.utils import days_between_month


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



class TimeSeriesConditionsDirector:
    def __init__(self):
        self.time_series_generator = TimeSeriesGeneratorConditions()

        with open(f"../../syntethic_data_generation/categories.json", "r") as f:
            categories = json.load(f)

        with open(f"../../syntethic_data_generation/cities_data.json", "r") as f:
            cities_data = json.load(f)

        self.asset_data_generator = AssetDataGenerator(cities_data=cities_data, categories=categories)

        with open(f"config_tsg_conditions.json", "r") as f:
            self.config = json.load(f)

        self.time_series_data = None


    def _build_baseline(self):
        ''' Build the baseline starting from the data obtained by the asset data generator'''
        asset_data = self.asset_data_generator.generate_new_asset()

        # Get baseline data
        num_days_life = 365 * asset_data["category_data"]["useful_life_years"]
        baseline_value = int(asset_data["category_data"]["useful_life_hours"] / num_days_life)
        max_value = self.config["max_value_series"]
        min_value = self.config["min_value_series"]
        sum_value = asset_data["category_data"]["useful_life_hours"]
        print(
            f"ASSET CATEGORY : {asset_data["category_data"]["name"]}, BASELINE VALUE: {baseline_value}, SUM VALUE: {sum_value}")

        # Generate noise and contract data
        start_date = self.config["asset_start_date"]
        years = int(np.random.choice(np.arange(self.config["contract_years_range"][0], self.config["contract_years_range"][1] + 1)))
        contract_months = years * 12
        num_units = days_between_month(start_date, contract_months)
        noise_ratio_std = np.random.uniform(0, self.config["max_ratio_noise_std"])
        print(f"START DATE: {start_date}, NUMBER OF DAYS: {num_units}, NOISE RATIO STD: {noise_ratio_std}")

        self.time_series_data["category"] = asset_data["category_data"]["name"]
        self.time_series_data["baseline_value"] = baseline_value
        self.time_series_data["sum_value"] = sum_value
        self.time_series_data["max_value"] = max_value
        self.time_series_data["min_value"] = min_value
        self.time_series_data["num_units"] = num_units

        # Build the baseline
        self.time_series_generator.build_baseline(num_units=num_units, baseline_value=baseline_value, min_value=min_value,
                                                max_value=max_value, sum_value=sum_value,
                                                noise_ratio_std=noise_ratio_std, start_date=start_date)

    def _build_changepoints(self):

        num_shifts = np.random.choice(np.arange(1, self.config["max_number_changepoints"] + 1))
        change_points = np.random.choice(np.arange(self.config["changepoints_start_end_interval"], self.time_series_data["num_units"] - self.config["changepoints_start_end_interval"]), num_shifts, replace=False)
        change_points = np.sort(change_points)
        self.time_series_data["changepoints"] = change_points.tolist()
        change_points = np.concatenate(([0], change_points, [self.time_series_data["num_units"]]))
        intervals = [(int(change_points[i]), int(change_points[i + 1] - 1)) for i in range(change_points.size - 1)]

        dates_intervals = []
        for interval in intervals:
            dates_intervals.append([str(self.time_series_generator.ts["ds"][interval[0]])[:10], str(self.time_series_generator.ts["ds"][interval[1]])[:10] ])

        self.time_series_data["dates_intervals"] = dates_intervals

    def _build_weekly(self):
        ''' build the weekly seasonality on the time series'''

        weekdays_array = [Weekday.MONDAY, Weekday.TUESDAY, Weekday.WEDNESDAY, Weekday.THURSDAY, Weekday.FRIDAY, Weekday.SATURDAY, Weekday.SUNDAY]

        # Obtain the general conditions randomly (general means that they apply to the entire time series)
        general_conditions_array = []
        if np.random.choice([True, False], p=[self.config["probability_general_seasonality"], 1 - self.config["probability_general_seasonality"]]):
            num_conditions = np.random.choice(np.arange(1, self.config["max_conditions_week"] + 1))
            conditions = np.random.choice(weekdays_array, num_conditions, replace=False).tolist()
            add_values = np.random.uniform(-self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"], self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"], num_conditions).tolist()

            general_conditions_array.append({
                "conditions" : conditions,
                "start_date": None,
                "end_date": None,
                "add_values" : add_values
            })




        # Obtain the specific conditions of a date interval
        weekly_conditions_array = []
        for interval in self.time_series_data["dates_intervals"]:
            num_conditions = np.random.choice(np.arange(1, self.config["max_conditions_week"] + 1))
            conditions = np.random.choice(weekdays_array, num_conditions, replace=False).tolist()
            add_values = np.random.uniform(-self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"], self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"], num_conditions).tolist()


            # Extract the weekly seasonalities for each interval
            weekly_seasonality = []
            for weekday in weekdays_array:
                value = 0
                for i,condition in enumerate(conditions):
                    if condition == weekday:
                        value += add_values[i]

                for i,condition in enumerate(general_conditions_array[0]["conditions"]):
                    if condition == weekday:
                        value += general_conditions_array[0]["add_values"][i]
                weekly_seasonality.append(value)

            weekly_conditions_array.append({
                "conditions" : conditions,
                "start_date": interval[0],
                "end_date": interval[1],
                "add_values" : add_values,
                "seasonality" : weekly_seasonality
            })




        # Apply Seasonality
        total_conditions_array = weekly_conditions_array + general_conditions_array
        for weekly_conditions in total_conditions_array:
            selected_conditions = weekly_conditions["conditions"]
            start_date = weekly_conditions["start_date"]
            end_date = weekly_conditions["end_date"]
            add_values_c = weekly_conditions["add_values"]

            for i,selected_condition in enumerate(selected_conditions):
                self.time_series_generator.apply_func_condition(condition=selected_condition,
                                                              func=lambda x: x + add_values_c[i], start_date=start_date,
                                                              end_date=end_date)

        # Save data
        for cond in total_conditions_array:
            cond["conditions"] = list(map( lambda x : x.value, cond["conditions"]))

        self.time_series_data["weekly_seasonality"] = total_conditions_array

    def _build_monthly(self):
        ''' build the monthly seasonality on the time series'''

        monthdays_array = range(1, 32)

        # Obtain the general conditions randomly (general means that they apply to the entire time series)
        general_conditions_array = []
        if np.random.choice([True, False], p=[self.config["probability_general_seasonality"],
                                              1 - self.config["probability_general_seasonality"]]):
            num_conditions = np.random.choice(np.arange(1, self.config["max_conditions_month"] + 1))
            conditions = np.random.choice(monthdays_array, num_conditions, replace=False).tolist()
            add_values = np.random.uniform(
                -self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"],
                self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"],
                num_conditions).tolist()

            general_conditions_array.append({
                "conditions": conditions,
                "start_date": None,
                "end_date": None,
                "add_values": add_values
            })

        # Obtain the specific conditions of a date interval
        monthly_conditions_array = []
        for interval in self.time_series_data["dates_intervals"]:
            num_conditions = np.random.choice(np.arange(1, self.config["max_conditions_month"] + 1))
            conditions = np.random.choice(monthdays_array, num_conditions, replace=False).tolist()
            add_values = np.random.uniform(
                -self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"],
                self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"],
                num_conditions).tolist()

            # Extract the weekly seasonalities for each interval
            monthly_seasonality = []
            for monthday in monthdays_array:
                value = 0
                for i, condition in enumerate(conditions):
                    if condition == monthday:
                        value += add_values[i]

                for i, condition in enumerate(general_conditions_array[0]["conditions"]):
                    if condition == monthday:
                        value += general_conditions_array[0]["add_values"][i]
                monthly_seasonality.append(value)

            monthly_conditions_array.append({
                "conditions": conditions,
                "start_date": interval[0],
                "end_date": interval[1],
                "add_values": add_values,
                "seasonality": monthly_seasonality
            })

        # Apply Seasonality
        total_conditions_array = monthly_conditions_array + general_conditions_array
        for monthly_conditions in total_conditions_array:
            selected_conditions = monthly_conditions["conditions"]
            start_date = monthly_conditions["start_date"]
            end_date = monthly_conditions["end_date"]
            add_values_c = monthly_conditions["add_values"]

            for i, selected_condition in enumerate(selected_conditions):
                self.time_series_generator.apply_func_condition(condition=selected_condition,
                                                                func=lambda x: x + add_values_c[i],
                                                                start_date=start_date,
                                                                end_date=end_date)
        # Save data
        self.time_series_data["monthly_seasonality"] = total_conditions_array

    @staticmethod
    def day_of_year_to_date(day_of_year):
        # Days in each month for a non-leap year
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        month = 1
        for days in days_in_month:
            if day_of_year <= days:
                # Return the date in the "2025-MM-DD" format
                return f"2025-{month:02d}-{day_of_year:02d}"
            day_of_year -= days
            month += 1

    @staticmethod
    def build_month_day_map():
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        month_day_map = {}
        current_day = 1

        for month, days in enumerate(days_in_month, start=1):
            # Generate a range of days for the current month
            month_day_map[month] = list(range(current_day, current_day + days))
            current_day += days

        return month_day_map


    def _build_yearly(self):
        ''' Build the yearly seasonality, upon the entire series'''

        yeardays_array = range(1, 366)
        yearmonths_array = range(1, 13)
        month_day_map = self.build_month_day_map()


        # Obtain the general conditions (always present for the yearly seasonality)
        yearly_conditions_array = []
        num_conditions = np.random.choice(np.arange(1, self.config["max_conditions_year"] + 1))
        conditions = np.random.choice(yearmonths_array, num_conditions, replace=False).tolist()


        add_values = np.random.uniform(
            -self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"],
            self.config["max_ratio_add_value_std"] * self.time_series_data["baseline_value"],
            num_conditions).tolist()

        # Extract the weekly seasonalities for each interval
        yearly_seasonality = []
        for yearday in range(1,366):
            value = 0
            for i, condition in enumerate(conditions):
                if yearday in month_day_map[condition]:
                    value += add_values[i]

            yearly_seasonality.append(value)

        final_conditions = []
        for condition in conditions:
            final_conditions.append((self.day_of_year_to_date(month_day_map[condition][0]), self.day_of_year_to_date(month_day_map[condition][-1])))

        yearly_conditions_array.append({
            "conditions": final_conditions,
            "start_date": None,
            "end_date": None,
            "add_values": add_values,
            "seasonality": yearly_seasonality
        })

        # Apply Seasonality
        for yearly_conditions in yearly_conditions_array:
            selected_conditions = yearly_conditions["conditions"]
            start_date = yearly_conditions["start_date"]
            end_date = yearly_conditions["end_date"]
            add_values_c = yearly_conditions["add_values"]

            for i, selected_condition in enumerate(selected_conditions):
                self.time_series_generator.apply_func_condition(condition=selected_condition,
                                                                func=lambda x: x + add_values_c[i],
                                                                start_date=start_date,
                                                                end_date=end_date)
        # Save data
        self.time_series_data["yearly_seasonality"] = yearly_conditions_array



    def make_ts_conditions(self):
        '''
        Use the builder to make the time series starting from the asset data generator and generating randomly
        conditions using the configurations
        '''

        # Reset the data
        self.time_series_data = {}
        self.time_series_generator.reset()

        # Build the baseline
        self._build_baseline()

        # Create the random changepoints
        self._build_changepoints()

        # Create the weekly seasonality
        self._build_weekly()

        # Create the monthly seasonality
        self._build_monthly()

        # Create the yearly seasonality
        self._build_yearly()

        self.time_series_data["time_series"] = self.time_series_generator.ts["y"].tolist()

        return self.time_series_data.copy()



