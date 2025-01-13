from tsg_conditions import *
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np



# Step 1: Fixed Instances (using caching)
@st.cache_resource
def initialize_time_series_objects():
    # Initialize Time Series Generator and Director
    tsg = TimeSeriesGeneratorConditions()

    return tsg

# Retrieve cached instances
tsg = initialize_time_series_objects()

# Step 2: Streamlit UI
st.title("Time Series Generator")

# Sidebar for parameters
with st.form("Time Series Generation Parameters"):
    # Initialize session state for flags

    baseline_value = st.number_input("Baseline value", value=8)
    num_units = st.number_input("Number of units", value=1095)
    min_value = st.number_input("Minimum value", value=0)
    max_value = st.number_input("Maximum value", value=24)
    noise_ratio_std = st.number_input("Noise ratio std", value=0)

    with st.expander("Operation settings"):
        condition_options = [w_day for w_day in Weekday] + np.arange(1,32).tolist()
        selected_condition = st.selectbox("Select one condition: Weekday or Monthday", condition_options)

        operation_options = ["Set", "Add", "Multiply"]
        selected_op = st.selectbox("Select one operation:", operation_options)

        value = st.number_input(
            "Value of the operation",
            value=0
        )

        # Allow the user to select a start date
        start_date = st.date_input("Pick the start date:")

        # Allow the user to select an end date
        end_date = st.date_input("Pick the end date:")



    # Submit button
    submitted = st.form_submit_button("Apply")

    # Undo button
    undo = st.form_submit_button("Undo")

    # Reset button
    reset = st.form_submit_button("Reset")

# Step 3: Generate Time Series Upon Submission
if submitted:
    if tsg.ts is None:
        tsg.build_baseline(num_units=num_units, baseline_value=baseline_value, max_value=max_value, min_value=min_value, noise_ratio_std=noise_ratio_std)
    else:
        if selected_op == "Add":
            func = lambda x: x + value
        elif selected_op == "Multiply":
            func = lambda x: x * value
        else:
            func = lambda x: value

        # Convert the selected date to a string in the format 'YYYY-MM-DD'
        if start_date:
            start_date = start_date.strftime("%Y-%m-%d")
        # Convert the selected date to a string in the format 'YYYY-MM-DD'
        if end_date:
            end_date = end_date.strftime("%Y-%m-%d")
        tsg.apply_func_condition(condition=selected_condition, func=func, start_date=start_date, end_date=end_date)


    time_series = tsg.ts

    # Plot the generated time series
    fig, ax = plt.subplots()
    ax.plot(time_series["ds"], time_series["y"], label="Generated Time Series")
    ax.set_title("Time Series")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Value")
    ax.legend()

    # Display plot
    st.pyplot(fig)

if undo:
    tsg.undo()

if reset:
    tsg.reset()



    #cd time_series_generation/pattern_style
    #streamlit run tsg_conditions_main.py