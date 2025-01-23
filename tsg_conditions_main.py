import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_series_generation.tsg_conditions import Weekday, TimeSeriesGeneratorConditions


def is_valid_date(date_str):
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d')
        return True
    except ValueError:
        return False


def set_input(input_value, default_value):
    try:
        if type(default_value) == int:
            input_value = int(input_value)
        elif type(default_value) == float:
            input_value = float(input_value)
        elif type(default_value) == str:
            input_value = str(input_value)
        return input_value
    except ValueError:
        return default_value

def main_generation():
    # Retrieve cached instances
    tsg = TimeSeriesGeneratorConditions()

    default_baseline_value = 8.
    default_num_units = 1095
    default_min_value = 0.
    default_max_value = 24.
    default_noise_ratio_std = 0.

    baseline = input(f"Do you want to set the baseline value? Default : {default_baseline_value}")
    num_un = input(f"Do you want to set the number of units? Default : {default_num_units}")
    min_v = input(f"Do you want to set the minimum value? Default : {default_min_value}")
    max_v = input(f"Do you want to set the maximum value? Default : {default_max_value}")
    noise_ratio = input(f"Do you want to set the noise ratio standard deviation? Default : {default_noise_ratio_std}")

    baseline_value = set_input(baseline, default_baseline_value)
    num_units = set_input(num_un, default_num_units)
    min_value = set_input(min_v, default_min_value)
    max_value = set_input(max_v, default_max_value)
    noise_ratio_std = set_input(noise_ratio, default_noise_ratio_std)

    tsg.build_baseline(num_units=num_units, baseline_value=baseline_value, max_value=max_value, min_value=min_value, noise_ratio_std=noise_ratio_std)
    # Plot the generated time series
    fig, ax = plt.subplots()
    ax.plot(tsg.ts["ds"], tsg.ts["y"], label="Generated Time Series")
    ax.set_title("Time Series")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Value")
    ax.legend()

    plt.show()

    done = False
    while not done:

            selected_op = None
            selected_condition = None
            start_date = None
            end_date = None

            condition_options = [w_day for w_day in Weekday] + np.arange(1,32).tolist()
            selected = False
            while not selected:
                selected_c = input(f"Select one condition index, Weekday or Monthday : {condition_options} : ")
                if int(selected_c) in range(len(condition_options)):
                    selected_condition = condition_options[int(selected_c)]
                    selected = True

            operation_options = ["Set", "Add", "Multiply"]
            selected = False
            while not selected:
                selected_o = input(f"Select one operation index : {operation_options} : ")
                if int(selected_o) in range(len(operation_options)):
                    selected_op = operation_options[int(selected_o)]
                    selected = True


            value = input(f"Select the value : ")
            value = set_input(value, 0.)

            # Allow the user to select a start date
            selected = False
            while not selected:
                start_date = input(f"Pick the start date, Expected format: 'YYYY-MM-DD' : ")
                if is_valid_date(start_date):
                    selected = True

            # Allow the user to select an end date
            selected = False
            while not selected:
                end_date = input(f"Pick the end date, Expected format: 'YYYY-MM-DD' : ")
                if is_valid_date(end_date):
                    selected = True



            if selected_op == "Add":
                func = lambda x: x + value
            elif selected_op == "Multiply":
                func = lambda x: x * value
            else:
                func = lambda x: value

            tsg.apply_func_condition(condition=selected_condition, func=func, start_date=start_date, end_date=end_date)



            # Plot the generated time series
            fig, ax = plt.subplots()
            ax.plot(tsg.ts["ds"], tsg.ts["y"], label="Generated Time Series")
            ax.set_title("Time Series")
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Value")
            ax.legend()

            plt.show()

            keep_going = input("Do you want to keep going? yes-y or no-n : ")
            if keep_going == "n":
                done = True

if __name__ == '__main__':
    main_generation()

    # 2025-01-01