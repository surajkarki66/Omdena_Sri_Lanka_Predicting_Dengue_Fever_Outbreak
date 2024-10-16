import streamlit as st
import pandas as pd

from typing import Any, Dict
from utils.model_handler import forecast_cases
from darts import TimeSeries

from utils.logger import logger
from utils.shap_utils import get_explainer, get_shap_explainability, plot_feature_importance, plot_feature_values, st_shap
from utils.utils import aggregate_weekly_cases, aggregate_yearly_cases_all_districts
from utils.visualization import plot_comparison, plot_forecast, plot_historical_data, plot_weekly_cases, plot_yearly_cases_all_districts


# Define functions for each tab's content
def display_forecasted_data(data: Dict):
    st.header("🔮 Forecasted Data")
    st.write(
        "This tab displays the forecasted data by forecaster for the selected district.")

    selected_district = data.get('selected_district')
    requires_weather = data.get('requires_weather')
    weather_data = data.get('weather_data')
    n_weeks = data.get('n_weeks')
    model = data.get('model')
    forecast_dates = data.get('forecast_dates')
    filtered_data = data.get('filtered_data')
    forecast_df = None
    if requires_weather:
        if weather_data is not None and not weather_data.empty:
            # Proceed with forecasting
            with st.spinner("Generating forecast..."):
                try:
                    # Convert weather_data to Darts TimeSeries
                    weather_timeseries = TimeSeries.from_dataframe(
                        weather_data,
                        time_col='Week_End_Date',
                        value_cols=[
                            "Avg Max Temp (°C)",
                            "Avg Min Temp (°C)",
                            "Avg Apparent Max Temp (°C)",
                            "Avg Apparent Min Temp (°C)",
                            "Total Precipitation (mm)",
                            "Avg Wind Speed (km/h)"
                        ],
                    )
                    forecast_df = forecast_cases(
                        model, n_weeks, forecast_dates, weather_data=weather_timeseries)
                    logger.info(f"Generated forecast for {n_weeks} weeks.")
                except TypeError:
                    # If forecast_cases doesn't accept weather_data, fallback
                    forecast_df = forecast_cases(
                        model, selected_district, n_weeks, forecast_dates)
                    logger.warning(
                        "forecast_cases does not accept weather_data. Proceeding without it.")
                except Exception as e:
                    logger.error(f"Error during forecasting: {e}")
                    st.error(f"Error during forecasting: {e}")
                    forecast_df = pd.DataFrame()

            if not forecast_df.empty:
                # Plot forecast
                fig_forecast = plot_forecast(forecast_df, selected_district)
                st.plotly_chart(fig_forecast, use_container_width=True)

                with st.expander("📄 View Forecast Data"):
                    st.dataframe(forecast_df)

                # Download Forecast Data
                with st.expander("📄 Download Forecast Data"):
                    csv = forecast_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Forecast as CSV",
                        data=csv,
                        file_name=f'forecast_{selected_district}.csv',
                        mime='text/csv'
                    )

                # Comparison plot
                st.subheader(f"🔄 Historical vs Forecasted")
                fig_comparison = plot_comparison(
                    filtered_data, forecast_df, selected_district)
                st.plotly_chart(fig_comparison, use_container_width=True)

                # Optionally, display weather data used for forecasting
                with st.expander("🌡️ Weather Data Used for Forecasting"):
                    st.dataframe(weather_data)
        else:
            # Prompt user to upload weather data
            st.warning(
                "Please upload a valid weather data CSV file to generate forecasts for this district.")
    else:
        # District does not require weather data; proceed with forecasting
        with st.spinner("Generating forecast..."):
            try:
                forecast_df = forecast_cases(
                    model, n_weeks, forecast_dates)
                logger.info(f"Generated forecast for {n_weeks} weeks.")
            except Exception as e:
                logger.error(f"Error during forecasting: {e}")
                st.error(f"Error during forecasting: {e}")
                forecast_df = pd.DataFrame()

        if not forecast_df.empty:
            # Plot forecast
            fig_forecast = plot_forecast(forecast_df, selected_district)
            st.plotly_chart(fig_forecast, use_container_width=True)

            with st.expander("📄 View Forecast Data"):
                st.dataframe(forecast_df)

            # Download Forecast Data
            with st.expander("📄 Download Forecast Data"):
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv,
                    file_name=f'forecast_{selected_district}.csv',
                    mime='text/csv'
                )

            fig_comparison = plot_comparison(
                filtered_data, forecast_df, selected_district)
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Optionally, display weather data used for forecasting (if any)
            if weather_data is not None and not weather_data.empty:
                with st.expander("🌡️ Weather Data Used for Forecasting"):
                    st.dataframe(weather_data)

    return forecast_df


def display_shap_explanation(data: Dict[str, Any], model: object):
    # Extract data from the input dictionary
    filtered_data = data.get('filtered_data')
    forecast_df = data.get('forecast_df')
    weather_data = data.get('weather_data')
    requires_weather = data.get('requires_weather')
    n_weeks = data.get('n_weeks')

    # Define common variables
    target_col = 'Number_of_Cases'
    covariate_cols = [
        'Avg Max Temp (°C)',
        'Avg Min Temp (°C)',
        "Avg Apparent Max Temp (°C)",
        "Avg Apparent Min Temp (°C)",
        "Total Precipitation (mm)",
        "Avg Wind Speed (km/h)"
    ]

    st.header("🔍 SHAP Explanation")
    st.write(
        "This tab provides SHAP explanations for the selected district's forecast.")

    value_cols = [target_col] + (covariate_cols if requires_weather else [])

    # Create the TimeSeries object
    series = TimeSeries.from_dataframe(
        filtered_data,
        time_col='Week_End_Date',
        value_cols=value_cols
    )
    background_data = series[target_col]

    if requires_weather:
        future_covariates = series[covariate_cols]
    else:
        future_covariates = None  # No covariates required

    # Initialize the explainer
    explainer = get_explainer(model, background_data, future_covariates)

    # Show loading spinner while processing SHAP explanation results
    with st.spinner("Calculating SHAP values..."):
        # Prepare forecasted DataFrame
        forecasted_df = forecast_df.rename(
            columns={'predicted_cases': target_col})
        filtered_df = filtered_data[['Week_End_Date', target_col]][:-12]
        filtered_last_12 = filtered_df.tail(12)
        final_df = pd.concat([filtered_last_12, forecasted_df]).sort_values(
            by='Week_End_Date').reset_index(drop=True)

        if requires_weather:
            # Prepare covariates DataFrame
            future_covariates_last_12 = filtered_data[[
                'Week_End_Date'] + covariate_cols][:-12].tail(12)
            future_covariates_given = weather_data[[
                'Week_End_Date'] + covariate_cols]
            final_covariates_df = pd.concat([future_covariates_last_12, future_covariates_given]).sort_values(
                by='Week_End_Date').reset_index(drop=True)

            # Create TimeSeries objects with covariates
            forecasted_series = TimeSeries.from_dataframe(
                final_df, time_col='Week_End_Date', value_cols=[target_col]
            )
            covariates_series = TimeSeries.from_dataframe(
                final_covariates_df, time_col='Week_End_Date', value_cols=covariate_cols
            )

            # Get SHAP explainability results
            results = get_shap_explainability(
                explainer, forecasted_series, covariates_series, horizons=n_weeks
            )
            shap_values = results.get_explanation(horizon=n_weeks)

            # Generate and display plots
            fig_shap = plot_feature_importance(shap_values)
            st.plotly_chart(fig_shap, use_container_width=True)

            feature_values = results.get_feature_values(horizon=n_weeks)
            fig_feat_values = plot_feature_values(feature_values)
            st.plotly_chart(fig_feat_values, use_container_width=True)

            # Generate and display force plot with covariates
            force_plot = explainer.force_plot_from_ts(
                foreground_series=forecasted_series,
                foreground_future_covariates=covariates_series,
                horizon=n_weeks
            )
        else:
            # Create TimeSeries object without covariates
            forecasted_series = TimeSeries.from_dataframe(
                final_df, time_col='Week_End_Date', value_cols=[target_col]
            )

            # Get SHAP explainability results without covariates
            results = get_shap_explainability(
                explainer, forecasted_series, horizons=n_weeks
            )
            shap_values = results.get_explanation(horizon=n_weeks)

            # Generate and display plots
            fig_shap = plot_feature_importance(shap_values)
            st.plotly_chart(fig_shap, use_container_width=True)

            feature_values = results.get_feature_values(horizon=n_weeks)
            fig_feat_values = plot_feature_values(feature_values)
            st.plotly_chart(fig_feat_values, use_container_width=True)

            # Generate and display force plot without covariates
            force_plot = explainer.force_plot_from_ts(
                foreground_series=forecasted_series,
                horizon=n_weeks
            )

    # Display the force plot after processing
    st.write("**Force Plot (Influence of Each Feature on the Forecast)**")
    st_shap(force_plot, 768)


def display_data_visualization(data: Dict):
    st.header("📊 Data Visualization")
    st.write("This tab contains visualizations of the original data from the year 2007 to 2024.")
    filtered_data = data.get('filtered_data')
    selected_district = data.get('selected_district')
    selected_variable = data.get('selected_variable')
    original_data = data.get('original_data')

    # Plot historical data
    fig_historical = plot_historical_data(
        filtered_data, selected_district, selected_variable)
    st.plotly_chart(fig_historical, use_container_width=True)

    with st.expander("📄 View Raw Data"):
        st.dataframe(filtered_data)

    # Aggregate yearly cases for all districts
    yearly_cases_all = aggregate_yearly_cases_all_districts(
        original_data, selected_district)

    # Aggregate weekly cases for the selected district
    weekly_cases = aggregate_weekly_cases(original_data, selected_district)

    # Plot yearly cases for all districts
    if not yearly_cases_all.empty or not weekly_cases.empty:
        fig_yearly_all = plot_yearly_cases_all_districts(yearly_cases_all)
        st.plotly_chart(fig_yearly_all, use_container_width=True)

        fig = plot_weekly_cases(weekly_cases)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write("No data available for any district.")
