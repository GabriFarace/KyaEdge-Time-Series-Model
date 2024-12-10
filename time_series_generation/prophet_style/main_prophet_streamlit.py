import logging
from tsg_prophet import TimeSeriesDirectorProphet, TimeSeriesGeneratorProphet
from time_series_generation.mine.time_series_generators import TimeSeriesFlags
import json
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np


def generate_ts_main():
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format for log messages
        datefmt='%Y-%m-%d %H:%M:%S'  # Format for timestamps
    )

    # Create a logger instance
    logger = logging.getLogger("ProphetLogger")

    # Get config file
    with open("tsg_config_prophet.json", "r") as config_file:
        config = json.load(config_file)

    tsg = TimeSeriesGeneratorProphet()
    tsd = TimeSeriesDirectorProphet(tsg, logger, config)




    # Streamlit app
    st.title("Time Series Generator")

    # Sidebar for parameters
    with st.form("Time Series Generation Parameters"):
        baseline_min = st.number_input("The minimum baseline value", value=10)
        a0 = st.number_input("a0 (Constant term)", value=0.5)
        n_terms = st.number_input("Number of terms (n)", min_value=1, value=10)
        ts_length = st.number_input("Time series length", min_value=100, value=1000)
        include_sine = st.checkbox("Include Sine Terms", value=True)
        include_cosine = st.checkbox("Include Cosine Terms", value=True)
        # Group for cosine-related inputs

        with st.expander("Baseline Component Settings"):
            max_years = st.number_input("Maximum Number of Years", value=20)
            baseline_min = st.number_input("Minimum baseline value", value=10)
            baseline_max = st.number_input("Maximum baseline value", value=500)

        with st.expander("Trend Component Settings"):
            max_shift_year = st.number_input("Maximum number of shift in the trend for each year", value=3)
            value_change_ratio = st.number_input("Maximum change ratio of the value wrt to baseline in a trend interval", value=3)

        if "frequencies" not in st.session_state:
            st.session_state["frequencies"] = [
                {"value" :  7, "params_number": 3, "coeff_ratio_std" :  0.5, "prob" :  0.7},
                {"value" :  30, "params_number": 5, "coeff_ratio_std" :  0.6, "prob" :  0.5},
                {"value" :  60, "params_number": 7, "coeff_ratio_std" :  0.7, "prob" :  0.5},
                {"value" :  365, "params_number": 10, "coeff_ratio_std" :  0.8, "prob" :  0.5}
            ]
        with st.expander(f"Seasonality Component Settings"):
            for idx, freq in enumerate(st.session_state["frequencies"]):
                col1, col2, col3, col4 = st.columns(4)
                freq["value"] = col1.number_input(f"Value (Hz)", value=freq["value"], key=f"value_{idx}")
                freq["params_number"] = col2.number_input(
                    f"Number of component of the Fourier series", value=freq["params_number"], min_value=1, key=f"params_number_{idx}"
                )
                freq["coeff_ratio_std"] = col3.number_input(
                    f"Coeff Ratio Std", value=freq["coeff_ratio_std"], min_value=0.0, max_value=1.0,
                    key=f"coeff_ratio_std_{idx}"
                )
                freq["prob"] = col4.number_input(
                    f"Probability", value=freq["prob"], min_value=0.0, max_value=1.0, key=f"prob_{idx}"
                )
                st.button("Remove", on_click=st.session_state["frequencies"].pop(idx), args=(idx,), key=f"remove_{idx}")

            # Add new frequency
            st.button("Add Frequency", on_click=add_frequency)

            an = st.text_input("Cosine coefficients (comma-separated)", value="0.5,0.4,0.3,0.2,0.1")

        # Submit button
        submitted = st.form_submit_button("Generate")

    if submitted:
        # Parse coefficients
        logger.info("Generating time series")

        tsd.make_ts_conditional(TimeSeriesFlags(True, False, True, True, True, False))
        time_series = tsg.generate()

        # Plot the time series
        fig, ax = plt.subplots()
        ax.plot(time_series, label="Generated Time Series Prophet style")
        ax.set_title("Time Series")
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.legend()

        # Display plot
        st.pyplot(fig)


if __name__ == "__main__":
    generate_ts_main()