import logging
from tsg_prophet import TimeSeriesDirectorProphet, TimeSeriesGeneratorProphet, TimeSeriesFlags
import json
import matplotlib.pyplot as plt
import streamlit as st


    # ODO add the possibiity of adding a frequency
    #st.button("Add Frequency", on_click=st.session_state["frequencies"].append({  "value": new_value, "params_number": new_params_number, "coeff_ratio_std": new_coeff_ratio_std, "prob": new_prob}), key=f"add_{idx})
    # ODO add the possibility of specifying the holidays



# Step 1: Fixed Instances (using caching)
@st.cache_resource
def initialize_time_series_objects():
    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Format for log messages
        datefmt='%Y-%m-%d %H:%M:%S'  # Format for timestamps
    )
    logger = logging.getLogger("ProphetLogger")

    # Load configuration file
    with open("tsg_config_prophet.json", "r") as config_file:
        config = json.load(config_file)

    # Initialize Time Series Generator and Director
    tsg = TimeSeriesGeneratorProphet()
    tsd = TimeSeriesDirectorProphet(tsg, logger, config)

    return tsg, tsd, logger

# Retrieve cached instances
tsg, tsd, logger = initialize_time_series_objects()

# Step 2: Streamlit UI
st.title("Time Series Generator")

# Sidebar for parameters
with st.form("Time Series Generation Parameters"):
    # Initialize session state for flags
    if "components" not in st.session_state:
        st.session_state["components"] = {
            "trend": True,
            "seasonality": False,
            "noise": False,
            "holidays": False,
            "max" : False
        }

    if "frequencies" not in st.session_state:
        st.session_state["frequencies"] = [
            {"value": 7, "params_number": 3, "coeff_ratio_std": 0.5, "prob": 0.7},
            {"value": 30, "params_number": 5, "coeff_ratio_std": 0.6, "prob": 0.5},
            {"value": 60, "params_number": 7, "coeff_ratio_std": 0.7, "prob": 0.5},
            {"value": 365, "params_number": 10, "coeff_ratio_std": 0.8, "prob": 0.5},
        ]

    # Master expander for enabling components
    with st.expander("Enable Components"):
        st.write("Enable the components you want to configure:")
        st.session_state["components"]["trend"] = st.checkbox(
            "Enable Trend", value=st.session_state["components"]["trend"]
        )
        st.session_state["components"]["seasonality"] = st.checkbox(
            "Enable Seasonality", value=st.session_state["components"]["seasonality"]
        )
        st.session_state["components"]["noise"] = st.checkbox(
            "Enable Noise", value=st.session_state["components"]["noise"]
        )
        st.session_state["components"]["holidays"] = st.checkbox(
            "Enable Holidays", value=st.session_state["components"]["holidays"]
        )
        st.session_state["components"]["max"] = st.checkbox(
            "Enable Max", value=st.session_state["components"]["max"]
        )

    with st.expander("Baseline Component Settings"):
        max_years = st.number_input("Maximum Number of Years", value=20)
        baseline_min = st.number_input("Minimum baseline value", value=10)
        baseline_max = st.number_input("Maximum baseline value", value=500)


    with st.expander("Trend Component Settings"):
        max_shift_year = st.number_input(
            "Maximum number of shifts in the trend for each year", value=3
        )
        value_change_ratio = st.number_input(
            "Maximum change ratio of the value wrt to baseline in a trend interval",
            value=3,
        )
    f = []
    with st.expander("Seasonality Component Settings"):
        for idx, freq in enumerate(st.session_state["frequencies"]):
            col1, col2, col3, col4 = st.columns(4)
            freq["value"] = col1.number_input(
                f"Frequency value", value=freq["value"], key=f"value_{idx}"
            )
            freq["params_number"] = col2.number_input(
                f"Number of components of the Fourier series",
                value=freq["params_number"],
                min_value=1,
                key=f"params_number_{idx}",
            )
            freq["coeff_ratio_std"] = col3.number_input(
                f"Standard Deviation of the coefficients ratio wrt to baseline",
                value=freq["coeff_ratio_std"],
                min_value=0.0,
                max_value=1.0,
                key=f"coeff_ratio_std_{idx}",
            )
            freq["prob"] = col4.number_input(
                f"Probability of being inserted",
                value=freq["prob"],
                min_value=0.0,
                max_value=1.0,
                key=f"prob_{idx}",
            )
            f.append({"value": freq["value"], "params_number": freq["params_number"], "coeff_ratio_std": freq["coeff_ratio_std"], "prob": freq["prob"]})


    with st.expander("Noise Component Settings"):
        noise_std_max = st.number_input(
            "Maximum ratio value of the standard deviation for the noise", value=0.1
        )


    with st.expander("Holidays Component Settings"):
        holiday_std_max = st.number_input(
            "Maximum ratio value of the standard deviation for the holidays",
            value=1,
        )
        max_holidays = st.number_input("Maximum number of holidays per year", value=5)
        max_window = st.number_input(
            "Maximum value of windows per holiday", value=3
        )


    # Submit button
    submitted = st.form_submit_button("Generate")

# Step 3: Generate Time Series Upon Submission
if submitted:


    logger.info("Generating time series with user parameters...")
    new_config = {
              "baseline" : {
                "n_years_max" : max_years,
                "baseline_min" : baseline_min,
                "baseline_max" : baseline_max
              },
              "trend" : {
                "max_shift_year" : max_shift_year,
                "value_change_ratio" : value_change_ratio
              },
              "seasonal" : {
                "frequencies" : f
              },
              "noise" : {
                "std_max" : noise_std_max
              },
              "holidays" : {
                "max_number_of_holidays_year" : max_holidays,
                "holidays_max_window" : max_window,
                "std_max" : holiday_std_max
              }
            }

    tsd.set_config(new_config)
    tsd.make_ts_conditional(
        TimeSeriesFlags(
            st.session_state["components"]["trend"],
            st.session_state["components"]["seasonality"],
            st.session_state["components"]["noise"],
            st.session_state["components"]["holidays"],
            False,
            st.session_state["components"]["max"],
        )
    )
    time_series = tsg.generate()

    # Step 4: Plot the generated time series
    fig, ax = plt.subplots()
    ax.plot(time_series, label="Generated Time Series Prophet Style")
    ax.set_title("Time Series")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Value")
    ax.legend()

    # Display plot
    st.pyplot(fig)
    #cd time_series_generation/prophet_style
    #streamlit run main_prophet_streamlit.py