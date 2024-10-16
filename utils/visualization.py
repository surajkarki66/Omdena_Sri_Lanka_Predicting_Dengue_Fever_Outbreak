# src/visualization.py
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd


def plot_historical_data(data: pd.DataFrame, district_name: str, column_name: str):
    """
    Plot historical dengue cases.

    Args:
        data (pd.DataFrame): Historical data.
        district_name (str): Name of the district.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        data,
        x='Week_End_Date',
        y=column_name,
        title=f'Historical {column_name} in {district_name}',
        labels={column_name: column_name, 'Week_End_Date': 'Week End Date'}
    )
    fig.update_layout(hovermode='x unified')
    return fig


def plot_forecast(forecast_df: pd.DataFrame, district_name: str):
    """
    Plot forecasted dengue cases.

    Args:
        forecast_df (pd.DataFrame): Forecasted data.
        district_name (str): Name of the district.

    Returns:
        Plotly Figure.
    """
    fig = px.line(
        forecast_df,
        x='Week_End_Date',
        y='predicted_cases',
        title=f'Forecasted Dengue Cases for {district_name}',
        labels={'predicted_cases': 'Predicted Cases',
                'week_end_date': 'Week End Date'}
    )
    fig.update_layout(hovermode='x unified')
    return fig


def plot_comparison(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, district_name: str):
    """
    Plot comparison between historical and forecasted dengue cases.

    Args:
        historical_df (pd.DataFrame): Historical data.
        forecast_df (pd.DataFrame): Forecasted data.
        district_name (str): Name of the district.

    Returns:
        Plotly Figure.
    """
    combined_df = pd.concat([
        historical_df[['Week_End_Date', 'Number_of_Cases']].rename(
            columns={'Number_of_Cases': 'Actual Cases'}),
        forecast_df.rename(columns={'predicted_cases': 'Forecasted Cases'})
    ], ignore_index=True)

    fig = px.line(
        combined_df,
        x='Week_End_Date',
        y=['Actual Cases', 'Forecasted Cases'],
        title=f'Historical vs Forecasted Dengue Cases in {district_name}',
        labels={'Number_of_Cases': 'Number of Cases',
                'Week_End_Date': 'Week End Date', 'variable': 'Legend'}
    )
    fig.update_layout(hovermode='x unified')
    return fig


def plot_yearly_cases_all_districts(yearly_data):
    # Create a line plot for yearly cases for all districts
    fig = px.line(yearly_data, x='Year', y='Number_of_Cases', color='District',
                  title='Yearly Dengue Cases by District for Each Year',
                  labels={
                      'Year': 'Year', 'Number_of_Cases': 'Number of Cases', 'District': 'District'},
                  markers=True)
    return fig


def plot_weekly_cases(weekly_data):
    # Create a new DataFrame to hold the week and case data
    weekly_data_list = []

    # Loop through each year to prepare data for Plotly
    for year in weekly_data['Year'].unique():
        year_data = weekly_data[weekly_data['Year'] == year]
        # Generate Week numbers from 1 to len(year_data)
        year_data['Week'] = np.arange(1, len(year_data) + 1)

        # Append year data to the list for each year
        weekly_data_list.append(year_data)

    # Concatenate all year data into a single DataFrame
    combined_data = pd.concat(weekly_data_list)

    # Create a line plot for weekly cases
    fig = px.line(combined_data, x='Week', y='Number_of_Cases', color='Year',
                  title='Weekly Dengue Cases for Selected District for Each Year',
                  labels={'Week': 'Week', 'Number_of_Cases': 'Number of Cases'},
                  markers=True)

    fig.update_traces(mode='lines+markers')  # Show both lines and markers
    fig.update_layout(xaxis_title='Week',
                      yaxis_title='Number of Cases', xaxis_tickangle=-45)

    return fig
