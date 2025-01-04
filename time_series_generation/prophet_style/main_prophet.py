import json
import logging
from tsg_prophet import TimeSeriesDirectorProphet, TimeSeriesGeneratorProphet, ParametersGenerationConfigs
from time_series_generation.prophet_style.tsg_prophet import TimeSeriesFlags
import matplotlib.pyplot as plt

def generate_ts_main():

    with open("tsg_config_prophet.json", "r") as config_file:
        config_data = json.load(config_file)
    config = ParametersGenerationConfigs()
    config.reset(config_data)
    tsd = TimeSeriesDirectorProphet(TimeSeriesGeneratorProphet(), config)

    finish = False
    while not finish:
        time_series = tsd.make_ts_conditional(TimeSeriesFlags(True, True, True, False, False, True, True))

        plt.figure(figsize=(12, 6))
        plt.plot(time_series, label="Generated Time Series Prophet style")
        plt.xlabel("Time (Days)")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        choice = input("Would you like to generate another time series? (y/n): ").lower()
        if choice == 'n':
            finish = True


if __name__ == "__main__":
    generate_ts_main()