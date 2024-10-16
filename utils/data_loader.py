# src/data_loader.py
import pandas as pd
import os

def load_data(data_file: str) -> pd.DataFrame:
    """
    Load historical dengue cases data from a CSV file.
    
    Args:
        data_file (str): Path to the CSV data file.
        
    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file, parse_dates=['Week_Start_Date', 'Week_End_Date'])
    
    # Validate required columns
    										
    required_columns = {'District', 'Number_of_Cases', 'Week_Start_Date', 'Month', 'Year', 'Week', 'Week_End_Date', 'Avg Max Temp (°C)', 'Avg Apparent Max Temp (°C)', 'Avg Apparent Min Temp (°C)', 'Total Precipitation (mm)', 'Total Rain (mm)', 'Avg Wind Speed (km/h)', 'Max Wind Gusts (km/h)', 'Weather Code', 'Avg Daylight Duration (hours)', 'Avg Sunrise Time', 'Avg Sunset Time'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns in data: {missing}")
    
    return df
