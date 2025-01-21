import numpy as np
from matplotlib import pyplot as plt

from syntethic_data_generation.asset_data_generation import AssetDataGenerator
from syntethic_data_generation.utils import days_between_month
from time_series_generation.pattern_style.tsg_conditions import TimeSeriesGeneratorConditions, \
    TimeSeriesConditionsDirector
import json
import pandas as pd

def generate_ts_loop(num_generation, name_output):
    ''' Main loop that generates sinthetic asset data with daily and monthly granularity'''

    time_series_director = TimeSeriesConditionsDirector()
    data = []


    for i in range(num_generation):
        time_series_data = time_series_director.make_ts_conditions()

        # Plot the generated time series
        fig, ax = plt.subplots()
        ax.plot(time_series_data["time_series"], label="Generated Time Series")
        ax.set_title("Time Series")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.legend()

        plt.show()

        data.append(time_series_data)

    with open(f'{name_output}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == '__main__':
    generate_ts_loop(num_generation=1, name_output="time_series_generated")