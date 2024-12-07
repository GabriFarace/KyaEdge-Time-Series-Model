import logging
from time_series_generators import TimeSeriesDirector, TimeSeriesGenerator
import json

def generate_time_series():
    # Configure the logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format for log messages
        datefmt='%Y-%m-%d %H:%M:%S'  # Format for timestamps
    )

    # Create a logger instance
    logger = logging.getLogger("MyLogger")

    # Get config file
    with open("tsg_config.json", "w") as config_file:
        config = json.load(config_file)

    tsg = TimeSeriesGenerator()
    tsd = TimeSeriesDirector(tsg, logger, config)

    finish = False
    while not finish:
        logger.info("Generating time series")
        tsd.make_ts_all()
        time_series = tsg.generate()


if __name__ == "__main__":
    generate_time_series()