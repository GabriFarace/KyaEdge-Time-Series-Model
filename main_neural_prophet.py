import json
import matplotlib.pyplot as plt

from time_series_generation.tsg_neural_prophet import ParametersGenerationConfigsNP, TimeSeriesDirectorNP, \
    TimeSeriesGeneratorNP, TimeSeriesFlagsNP


def generate_ts_main():

    with open("json_files/tsg_config_neural_prophet.json", "r") as config_file:
        config_data = json.load(config_file)
    config = ParametersGenerationConfigsNP(reset_configuration=config_data)
    tsd = TimeSeriesDirectorNP(TimeSeriesGeneratorNP(), config)

    finish = False
    while not finish:
        time_series = tsd.make_ts_conditional(TimeSeriesFlagsNP(True, True, True, False, False, True, True, False))

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