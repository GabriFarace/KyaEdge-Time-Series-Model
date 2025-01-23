from matplotlib import pyplot as plt
import pandas as pd

from syntethic_data_generation.pd_date_utils import create_prophet_dataframe, create_monthly_average_df


def plot_differences_telemetry(true_telemetry, forecasted_telemetry, today, start_date):
    ''' Plot the difference between the true telemetry and the forecasted one'''
    # Get data
    df_true = create_prophet_dataframe(true_telemetry,start_date)
    df_forecast = create_prophet_dataframe(forecasted_telemetry["mean_curve"],start_date)

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_true['ds'].values, df_true['y'], c='red')
    ax.plot(df_forecast['ds'].values, df_forecast['y'], c='#0072B2')
    ax.fill_between(df_forecast['ds'].values, forecasted_telemetry["lower_bound_curve"],
                    forecasted_telemetry['upper_bound_curve'], color='#0072B2',
                    alpha=0.2)
    ax.axvline(x=pd.to_datetime(today), c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('y')
    ax.set_xlabel('ds')
    plt.show()

def plot_differences_telemetry_months(true_telemetry, forecasted_telemetry, start_date):
    ''' Plot the difference between the true telemetry and the forecasted one'''
    # Get data
    df_true = create_monthly_average_df(true_telemetry,start_date)
    df_forecast = create_monthly_average_df(forecasted_telemetry,start_date)

    # Convert 'ds' back to datetime for sorting purposes
    df_true['ds_datetime'] = pd.to_datetime(df_true['ds'], format='%B %Y')

    # Sort by date
    df_true = df_true.sort_values(by='ds_datetime')

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_true['ds_datetime'], df_true['y'], c='red', label='True (Generated) telemetry')
    ax.plot(df_true['ds_datetime'], df_forecast['y'], c='#0072B2', label='Forecasted telemetry')


    today = pd.Timestamp.today()
    current_month = pd.Timestamp(today.year, today.month, 1)
    ax.axvline(x=current_month, c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('hours')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

def plot_leasing_risk(remarketing_value_curve, residual_debt, gap_curve, start_date):
    ''' Plot the leasing risk curves'''

    # Get data
    df_market = create_monthly_average_df(remarketing_value_curve["mean_curve"],start_date)
    df_residual_debt = create_monthly_average_df(residual_debt["curve"],start_date)
    df_gap = create_monthly_average_df(gap_curve["mean_curve"],start_date)

    # Convert 'ds' back to datetime for sorting purposes
    df_market['ds_datetime'] = pd.to_datetime(df_market['ds'], format='%B %Y')

    # Sort by date
    df_market = df_market.sort_values(by='ds_datetime')

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_market['ds_datetime'], df_market['y'], c='green', label='Remarketing value curve')
    ax.plot(df_market['ds_datetime'], df_residual_debt['y'], c='blue', label='Residual debt curve')
    ax.plot(df_market['ds_datetime'], df_gap['y'], c='red', label='GAP curve')

    today = pd.Timestamp.today()
    current_month = pd.Timestamp(today.year, today.month, 1)
    ax.axvline(x=current_month, c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('$')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

def plot_quality_rating(quality_rating_curve, operational_use_curve, start_date):
    ''' Plot the quality rating curves'''

    # Get data
    df_quality = create_monthly_average_df(quality_rating_curve["mean_curve"],start_date)
    df_ou = create_monthly_average_df(operational_use_curve["mean_curve"],start_date)

    # Convert 'ds' back to datetime for sorting purposes
    df_quality['ds_datetime'] = pd.to_datetime(df_quality['ds'], format='%B %Y')

    # Sort by date
    df_quality = df_quality.sort_values(by='ds_datetime')

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_quality['ds_datetime'], df_quality['y'], c='red', label='Quality Rating curve')

    today = pd.Timestamp.today()
    current_month = pd.Timestamp(today.year, today.month, 1)
    ax.axvline(x=current_month, c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('quality')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()


    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_quality['ds_datetime'], df_ou['y'], c='red', label='Operational use curve')

    today = pd.Timestamp.today()
    current_month = pd.Timestamp(today.year, today.month, 1)
    ax.axvline(x=current_month, c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('operational use in %')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

def plot_esg_rating(footprint_curve, energy_consumed, start_date):
    ''' Plot the esg rating curves'''

    # Get data
    df_footprint = create_monthly_average_df(footprint_curve["mean_curve"],start_date)
    df_energy = create_monthly_average_df(energy_consumed["mean_curve"],start_date)

    # Convert 'ds' back to datetime for sorting purposes
    df_footprint['ds_datetime'] = pd.to_datetime(df_footprint['ds'], format='%B %Y')

    # Sort by date
    df_footprint = df_footprint.sort_values(by='ds_datetime')

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_footprint['ds_datetime'], df_footprint['y'], c='blue', label='Footprint curve mean Kg CO2')
    ax.plot(df_footprint['ds_datetime'], df_energy['y'], c='orange', label='Energy consumed curve mean kWh')

    today = pd.Timestamp.today()
    current_month = pd.Timestamp(today.year, today.month, 1)
    ax.axvline(x=current_month, c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('y')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

def plot_lower_upper(curves, start_date, name):
    ''' Plot the lower, mean and upper bound of the curves given in input'''
    # Get data
    df = create_monthly_average_df(curves["mean_curve"],start_date)
    df_lower = create_monthly_average_df(curves["lower_bound_curve"],start_date)
    df_upper = create_monthly_average_df(curves["upper_bound_curve"],start_date)

    # Convert 'ds' back to datetime for sorting purposes
    df['ds_datetime'] = pd.to_datetime(df['ds'], format='%B %Y')

    # Sort by date
    df = df.sort_values(by='ds_datetime')

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df['ds_datetime'], df['y'], c='#0072B2', label=name)
    ax.fill_between(df['ds_datetime'], df_lower["y"],
                    df_upper['y'], color='#0072B2',
                    alpha=0.2)

    today = pd.Timestamp.today()
    current_month = pd.Timestamp(today.year, today.month, 1)
    ax.axvline(x=current_month, c='gray', lw=4, alpha=0.5)
    ax.set_ylabel('y')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

