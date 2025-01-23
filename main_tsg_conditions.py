from matplotlib import pyplot as plt
from time_series_generation.tsg_conditions import TimeSeriesConditionsDirector
import json


def generate_ts_loop(num_generation, name_output, plotting):
    ''' Main loop that generates sinthetic asset data with daily and monthly granularity'''

    with open(f"json_files/cities_data.json", "r") as f:
        cities_data = json.load(f)
    with open(f"json_files/categories.json", "r") as f:
        categories = json.load(f)
    with open(f"json_files/config_tsg_conditions.json", "r") as f:
        config = json.load(f)

    time_series_director = TimeSeriesConditionsDirector(categories=categories, config=config, cities_data=cities_data)
    data = []


    for i in range(num_generation):
        time_series_data = time_series_director.make_ts_conditions()

        if plotting:
            # Plot the generated time series
            fig, ax = plt.subplots()
            ax.plot(time_series_data["time_series"], label="Generated Time Series")
            ax.set_title("Time Series")
            ax.set_xlabel("Time (Days)")
            ax.set_ylabel("Value")
            ax.legend()

            plt.show()

        data.append(time_series_data)

    with open(f'json_files/{name_output}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == '__main__':
    generate_ts_loop(num_generation=1, name_output="time_series_generated", plotting=False)