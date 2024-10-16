# src/model_handler.py
from typing import List, Union

import joblib
import os
import datetime
import pandas as pd

from darts import TimeSeries

from config.constants import MODEL_LOADERS, TransformerModel, NBEATSModel

def load_model(model_file: str) -> object:
    """
    Load a forecasting model from a specified file.

    Args:
        model_file (str): Path to the model file.

    Returns:
        Loaded model instance.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    model_class = MODEL_LOADERS.get(model_file)
    if model_class:
        return model_class.load(model_file)
    else:
        # Fallback to joblib for non-Darts models
        return joblib.load(model_file)


def forecast_cases(
    model: object,
    n_weeks: int,
    forecast_dates: Union[datetime.date, List[datetime.date], pd.Series],
    weather_data: TimeSeries = None
) -> pd.DataFrame:
    """
    Generate dengue case forecasts for the next n_weeks.
    
    Args:
        model: Trained model.
        data (pd.DataFrame): Historical data for the selected district.
        n_weeks (int): Number of weeks to forecast.
        
    Returns:
        pd.DataFrame: DataFrame with forecasted dates and predicted cases.
    """
    #Determine if the model requires future co-variates
    if isinstance(model, (TransformerModel, NBEATSModel)):
        model.to_cpu()
        
    if weather_data:
        forecast_values = model.predict(n_weeks, future_covariates=weather_data)
    else:
        forecast_values = model.predict(n_weeks)


    # Extract the raw NumPy array and flatten it
    forecast_values_array = forecast_values.values()

    # Round the forecasted values to integers
    forecast_values_rounded = [round(value[0]) for value in forecast_values_array]

    # Create the forecast DataFrame
    forecast_df = pd.DataFrame({
        'Week_End_Date': forecast_dates,
        'predicted_cases': forecast_values_rounded
    })
    
    return forecast_df