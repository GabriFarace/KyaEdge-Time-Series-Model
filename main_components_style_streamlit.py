import matplotlib.pyplot as plt
import streamlit as st
import json
from time_series_generation.tsg_components_style import TimeSeriesGeneratorComponents, ComponentsFlags


# Step 1: Fixed Instances (using caching)
@st.cache_resource
def initialize_time_series_objects():

    with open("json_files/config_generator_components_style.json", "r") as config_file:
        config = json.load(config_file)

    # Initialize Time Series Generator and Director
    tsg = TimeSeriesGeneratorComponents(config)

    return tsg, config

# Retrieve cached instances
tsg, config = initialize_time_series_objects()

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
            "inactivity" : False,
            "autoregression" : False,
            "sum_constraint" : False,
            "interval_constraint" : False,
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
        st.session_state["components"]["autoregression"] = st.checkbox(
            "Enable autoregression", value=st.session_state["components"]["autoregression"]
        )
        st.session_state["components"]["inactivity"] = st.checkbox(
            "Enable Inactivity", value=st.session_state["components"]["inactivity"]
        )
        st.session_state["components"]["interval_constraint"] = st.checkbox(
            "Enable Interval Constraint", value=st.session_state["components"]["interval_constraint"]
        )
        st.session_state["components"]["sum_constraint"] = st.checkbox(
            "Enable Sum Constraint", value=st.session_state["components"]["sum_constraint"]
        )

    with st.expander("Baseline Component Settings"):
        max_units = st.number_input("Maximum Number of units", value=7300)
        min_units = st.number_input("Maximum Number of units", value=365)
        baseline_min = st.number_input("Minimum baseline value", value=10)
        baseline_max = st.number_input("Maximum baseline value", value=500)
        unit_is_energy = st.checkbox(
            "Unit is Energy", value=True
        )

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

    with st.expander("Autoregressive Component Settings"):
        max_number_of_lags = st.number_input(
            "Maximum number of lags in the autoregressive component", value=100
        )
        max_coefficient = st.number_input(
            "Maximum value of the coefficients for the autoregressive component",
            value=0.1,
        )

    with st.expander("Inactivity Component Settings"):
        inactivity_max_prob = st.number_input(
            "Maximum probability value of inactivity (0)", value=0.01
        )


    # Submit button
    submitted = st.form_submit_button("Generate")

# Step 3: Generate Time Series Upon Submission
if submitted:
    print("\n\n")
    # Update Configurations
    tsg.config["baseline"]["n_units_max"] = max_units
    tsg.config["baseline"]["n_units_min"] = min_units
    tsg.config["baseline"]["baseline_min"] = baseline_min
    tsg.config["baseline"]["baseline_max"] = baseline_max
    tsg.config["baseline"]["unit_is_energy"] = unit_is_energy
    tsg.config["trend"]["max_shift_year"] = max_shift_year
    tsg.config["trend"]["value_change_ratio"] = value_change_ratio
    tsg.config["noise"]["std_max"] = noise_std_max
    tsg.config["autoregression"]["max_number_of_lags"] = max_number_of_lags
    tsg.config["autoregression"]["max_coefficient"] = max_coefficient
    tsg.config["seasonal"]["frequencies"] = f
    tsg.config["inactivity"]["max_prob"] = inactivity_max_prob

    # Make the time series with the director
    time_series = tsg.generate(
        ComponentsFlags(
            trend=st.session_state["components"]["trend"],
            seasonal=st.session_state["components"]["seasonality"],
            noise=st.session_state["components"]["noise"],
            inactivity=st.session_state["components"]["inactivity"],
            autoregression=st.session_state["components"]["autoregression"],
            interval_constraint=st.session_state["components"]["interval_constraint"],
            sum_constraint=st.session_state["components"]["sum_constraint"]
        )
    )

    # Step 4: Plot the generated time series
    fig, ax = plt.subplots()
    ax.plot(time_series, label="Generated Time Series Prophet Style")
    ax.set_title("Time Series")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Value")
    ax.legend()

    # Display plot
    st.pyplot(fig)




    #streamlit run main_components_style_streamlit.py
    #jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --no-browser
