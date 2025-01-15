from matplotlib import pyplot as plt
from main_generation import compact_into_months, create_monthly_average_df
import pandas as pd
import json

def plot_leasing_risk(remarketing_value_curve, residual_debt, gap_curve, start_date):

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
    ax.set_ylabel('y')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

def plot_quality_rating(quality_rating_curve, operational_use_curve, start_date):

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
    ax.set_ylabel('y')
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
    ax.set_ylabel('y')
    ax.set_xlabel('month')
    # Format x-axis to show months
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%B %Y'))
    plt.gcf().autofmt_xdate()

    # Add legend
    plt.legend()
    plt.show()

def plot_esg_rating(footprint_curve, energy_consumed, start_date):

    # Get data
    df_footprint = create_monthly_average_df(footprint_curve["mean_curve"],start_date)
    df_energy = create_monthly_average_df(energy_consumed["mean_curve"],start_date)

    # Convert 'ds' back to datetime for sorting purposes
    df_footprint['ds_datetime'] = pd.to_datetime(df_footprint['ds'], format='%B %Y')

    # Sort by date
    df_footprint = df_footprint.sort_values(by='ds_datetime')

    fig = plt.figure(facecolor='w', figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df_footprint['ds_datetime'], df_footprint['y'], c='blue', label='Footprint curve')
    ax.plot(df_footprint['ds_datetime'], df_energy['y'], c='orange', label='Energy consumed curve')

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


def plot_main(number_of_asset):
    with open("data.json", "r") as f:
        data = json.load(f)

    counter = 0

    for asset_data in data:
        scores = asset_data["scores"]

        plot_leasing_risk(scores["leasing_risk"]["remarketing_value_curve"], scores["leasing_risk"]["residual_debt"], scores["leasing_risk"]["gap_curve"], asset_data["start_date"])
        plot_lower_upper(scores["leasing_risk"]["remarketing_value_curve"], asset_data["start_date"], "Market Value")

        plot_quality_rating(scores["asset_quality_rating"]["quality_rating_curve"], scores["asset_quality_rating"]["operational_use_curve"], asset_data["start_date"])
        plot_lower_upper(scores["asset_quality_rating"]["quality_rating_curve"], asset_data["start_date"], "Quality Value")
        plot_lower_upper(scores["asset_quality_rating"]["operational_use_curve"], asset_data["start_date"],
                         "Operational Use")

        plot_esg_rating(scores["esg_rating"]["footprint_curve"], scores["esg_rating"]["energy_consumed"], asset_data["start_date"])
        plot_lower_upper(scores["esg_rating"]["footprint_curve"], asset_data["start_date"], "Footprint Value")
        plot_lower_upper(scores["esg_rating"]["energy_consumed"], asset_data["start_date"],
                         "Energy Consumed")

        counter +=1
        if counter == number_of_asset:
            break

if __name__ == '__main__':
    plot_main(5)