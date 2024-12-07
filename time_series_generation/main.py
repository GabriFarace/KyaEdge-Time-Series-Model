import logging
from time_series_generators import TimeSeriesDirector, TimeSeriesGenerator, TimeSeriesFlags
import json
import matplotlib.pyplot as plt

def generate_ts_main():
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format for log messages
        datefmt='%Y-%m-%d %H:%M:%S'  # Format for timestamps
    )

    # Create a logger instance
    logger = logging.getLogger("MyLogger")

    # Get config file
    with open("tsg_config.json", "r") as config_file:
        config = json.load(config_file)

    tsg = TimeSeriesGenerator()
    tsd = TimeSeriesDirector(tsg, logger, config)

    finish = False
    while not finish:
        logger.info("Generating time series")
        tsd.make_ts_conditional(TimeSeriesFlags(True, True, True, True, True, False))
        time_series = tsg.generate()

        plt.figure(figsize=(12, 6))
        plt.plot(time_series, label="Generated Time Series")
        plt.xlabel("Time (Days)")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

        choice = input("Would you like to generate another time series? (y/n): ").lower()
        if choice == 'n':
            finish = True


if __name__ == "__main__":
    generate_ts_main()