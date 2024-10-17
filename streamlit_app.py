# app.py
import datetime
import streamlit as st
import pandas as pd
import yaml
import os

from utils.utils import extract_pdf
from utils.data_loader import load_data
from utils.model_handler import load_model
from utils.logger import logger
from config.constants import DISTRICT_WITH_WEATHER_FIELD, DISTRICT_WITHOUT_SHAP_EXPLANATION
from components.tabs import display_data_visualization, display_forecasted_data, display_help, display_shap_explanation

# ------------------------
# Configuration and Setup
# ------------------------

# Set Streamlit page configuration
st.set_page_config(
    page_title="Sri Lanka Dengue Fever Outbreak Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("Sri Lanka Dengue Fever Outbreak Prediction")

# Load configuration


@st.cache_resource(show_spinner=True)
def load_config(config_path: str = "config/districts.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        st.error(f"Configuration file not found: {config_path}")
        return {}

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config()

if not config:
    st.stop()

districts = [district['name'] for district in config.get('districts', [])]

st.sidebar.image('assets/logo.png', use_column_width=True)

# ------------------------
# Sidebar for District, Variable, Forecast Weeks, and Weather Data Selection
# ------------------------

st.sidebar.header("📍 Select District and Variable")

# Select District
selected_district = st.sidebar.selectbox(
    "Choose a district", options=sorted(districts))

# Select Variable to Plot
plotable_columns = [
    'Number_of_Cases',
    'Avg Max Temp (°C)',
    'Avg Min Temp (°C)',
    'Avg Apparent Max Temp (°C)',
    'Avg Apparent Min Temp (°C)',
    'Total Precipitation (mm)',
    'Total Rain (mm)',
    'Avg Wind Speed (km/h)',
    'Max Wind Gusts (km/h)',
    'Avg Daylight Duration (hours)',
    'Avg Sunrise Time',
    'Avg Sunset Time'
]

selected_variable = st.sidebar.selectbox(
    "Choose a variable to plot", options=plotable_columns)

# ------------------------
# Forecast Weeks Selection
# ------------------------

# Sidebar for forecast parameters
st.sidebar.subheader("🔮 Forecast Parameters")

# Create a dropdown menu for selecting months
month_options = {
    "3 Months": 13 if selected_district in DISTRICT_WITH_WEATHER_FIELD else 12,
    "4 Months": 16,
    "5 Months": 20,
    "6 Months": 24
}

# Use selectbox to select the forecast duration
selected_month = st.sidebar.selectbox(
    "Select Forecast Duration",
    options=list(month_options.keys())
)

# Get the corresponding number of weeks from the selected month
n_weeks = month_options[selected_month]

st.sidebar.write(f"Number of Months to Forecast: {selected_month}")

# Our training data was up to this point
last_date = datetime.datetime(2024, 4, 30, 0, 0)
forecast_dates = pd.date_range(last_date, periods=n_weeks, freq='W-MON')

# ------------------------
# Conditional Weather Data Input Fields
# ------------------------
weather_data = None

# Initialize a placeholder for uploaded weather data
uploaded_weather_data = None

if selected_district in DISTRICT_WITH_WEATHER_FIELD:
    st.sidebar.subheader("🌦️ Upload Weather Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a CSV file containing weather data (Note: Checkout the Help tab to understand what and how to upload an input weather data.)",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            uploaded_weather_data = pd.read_csv(uploaded_file)

            # Define required columns
            required_columns = [
                'Week_Start_Date',
                'Week_End_Date',
                'Avg Max Temp (°C)',
                'Avg Min Temp (°C)',
                'Avg Apparent Max Temp (°C)',
                'Avg Apparent Min Temp (°C)',
                'Total Precipitation (mm)',
                'Avg Wind Speed (km/h)',
                'Avg Daylight Duration (hours)',
                'Avg Sunrise Time',
                'Avg Sunset Time'
            ]

            # Check if all required columns are present
            missing_columns = [
                col for col in required_columns if col not in uploaded_weather_data.columns]

            if missing_columns:
                st.error(
                    f"The uploaded CSV is missing the following required columns: {', '.join(missing_columns)}")
                uploaded_weather_data = None
            else:
                # Convert 'Week_End_Date' and 'Week_Start_Date' to datetime
                uploaded_weather_data['Week_End_Date'] = pd.to_datetime(
                    uploaded_weather_data['Week_End_Date'], errors='coerce')
                uploaded_weather_data['Week_Start_Date'] = pd.to_datetime(
                    uploaded_weather_data['Week_Start_Date'], errors='coerce')

                # Check for any NaT (Not a Time) values resulting from failed conversions
                if uploaded_weather_data['Week_End_Date'].isnull().any() or uploaded_weather_data['Week_Start_Date'].isnull().any():
                    st.error(
                        "Some dates in 'Week_End_Date' or 'Week_Start_Date' could not be parsed. Please ensure they are in the correct format (YYYY-MM-DD).")
                    uploaded_weather_data = None
                else:
                    # Define the cutoff date
                    cutoff_date = pd.to_datetime('2024-04-29')

                    # Get the actual minimum Week_Start_Date
                    actual_min_date = uploaded_weather_data['Week_Start_Date'].min(
                    )

                    # Check if the minimum Week_Start_Date is exactly the cutoff date
                    if actual_min_date != cutoff_date:
                        st.warning(
                            f"The minimum Week_Start_Date in the uploaded data is {actual_min_date.strftime('%Y-%m-%d')}, which is not equal to 2024-04-29.")
                        uploaded_weather_data = None
                    else:
                        # Sort the data by 'Week_Start_Date' ascending
                        uploaded_weather_data = uploaded_weather_data.sort_values(
                            'Week_Start_Date')

                        # Calculate the number of available weeks
                        available_weeks = len(uploaded_weather_data)

                        # Validate if available weeks meet the forecast requirement
                        if available_weeks >= n_weeks:
                            # Fetch the required number of weeks
                            weather_data_temp = uploaded_weather_data.head(
                                n_weeks).copy()
                            # Select and reorder necessary columns
                            weather_data = weather_data_temp[[
                                "Week_End_Date",
                                "Avg Max Temp (°C)",
                                "Avg Min Temp (°C)",
                                "Avg Apparent Max Temp (°C)",
                                "Avg Apparent Min Temp (°C)",
                                "Total Precipitation (mm)",
                                "Avg Wind Speed (km/h)"
                            ]]
                            st.success(
                                f"Weather data uploaded and validated successfully! Using {n_weeks} weeks of data.")
                        else:
                            st.warning(
                                f"The uploaded weather data contains only {available_weeks} weeks, but {n_weeks} weeks are required for forecasting.")
                            uploaded_weather_data = None
                            weather_data = None

        except Exception as e:
            logger.error(f"Error processing the uploaded file: {e}")
            st.error(
                f"An error occurred while processing the uploaded file: {e}")

    else:
        # If no file is uploaded, do not attempt to fetch automatically
        pass

# ------------------------
# Fetch Selected District Configuration
# ------------------------

district_config = next(
    (d for d in config['districts'] if d['name'] == selected_district),
    None
)

if not district_config:
    logger.error(f"No configuration found for {selected_district}")
    st.error(f"No configuration found for {selected_district}")
    st.stop()

model_file = district_config['model_file']
data_file = 'data/Copy of Sri_lanka_dengue_cases_weather_weekly_2007_2024_.csv'

logger.info(f"Selected District: {selected_district}")
logger.info(f"Model File: {model_file}")
logger.info(f"Data File: {data_file}")

# ------------------------
# Data and Model Loading
# ------------------------


@st.cache_data(show_spinner=True)
def get_historical_data(data_file: str) -> pd.DataFrame:
    """
    Load historical data with caching.

    Args:
        data_file (str): Path to the data file.

    Returns:
        pd.DataFrame: Historical data.
    """
    try:
        df = load_data(data_file)
        logger.info(f"Loaded data from {data_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=True)
def get_model(model_file: str):
    """
    Load model with caching.

    Args:
        model_file (str): Path to the model file.

    Returns:
        Loaded model.
    """
    try:
        model = load_model(model_file)
        logger.info(f"Loaded model from {model_file}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None


# Load data and model
with st.spinner("Loading data..."):
    data = get_historical_data(data_file)

with st.spinner("Loading model..."):
    model = get_model(model_file)


# ------------------------
# Upload Multiple PDFs
# ------------------------

st.sidebar.subheader("📄 Upload dengue cases PDF files")
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload dengue cases PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Dummy processing function for PDF files


def process_pdfs(pdf_files):
    """
    Dummy function to process PDF files.

    Args:
        pdf_files (list): List of PDF file objects.

    Returns:
        list: List of processed file names.
    """
    try:
        pdf_data = extract_pdf(pdf_files)

        # Convert the list of extracted data into a DataFrame
        df_extracted_data = pd.DataFrame(pdf_data)
        df_extracted_data['Number_of_Cases'] = df_extracted_data['Number_of_Cases'].replace(
            'Nil', 0)
        df_sorted = df_extracted_data.sort_values(
            by=['District', 'Week', 'Week_Start_Date']).reset_index(drop=True)

        # Reset the index (optional)
        df_sorted.reset_index(drop=True, inplace=True)
        return df_sorted
    except Exception as e:
        logger.error(f"Failed to process: {e}")


if uploaded_pdfs:
    processed_df = process_pdfs(uploaded_pdfs)
    st.success(f"Processed {len(uploaded_pdfs)} PDF files.")

    processed_df['Week_Start_Date'] = pd.to_datetime(
        processed_df['Week_Start_Date'])

    # Get min and max week start dates
    min_week_start_date = processed_df['Week_Start_Date'].min().date()
    max_week_start_date = processed_df['Week_Start_Date'].max().date()

    # Format the dates for the filename
    filename_date_range = f"{min_week_start_date} to {max_week_start_date}".replace(
        "-", "_")  # Replace dashes for filename
    # Create the CSV filename
    csv_filename = f"weekly_dengue_cases_{filename_date_range}.csv"

    # Convert DataFrame to CSV
    csv = processed_df.to_csv(index=False).encode('utf-8')

    # Create a download button for the CSV file
    st.download_button(
        label="Download Processed Data as CSV",
        data=csv,
        file_name=csv_filename,  # Use the dynamic filename
        mime='text/csv',
    )


if data.empty or model is None:
    st.warning("Unable to load data or model. Please check configurations.")
    st.stop()

# Filter data for the selected district
filtered_data = data[data['District'] == selected_district].copy()

if filtered_data.empty:
    st.warning(f"No historical data available for {selected_district}.")
    st.stop()

# Ensure 'Week_End_Date' is datetime
filtered_data['Week_End_Date'] = pd.to_datetime(filtered_data['Week_End_Date'])

# ------------------------
# Application Tabs
# ------------------------

# Check if the selected district requires weather data
requires_weather = selected_district in DISTRICT_WITH_WEATHER_FIELD

# Define all possible tabs
tabs = st.tabs(
    ["🔮 Forecasted Data", "🔍 SHAP Explanation", "📊 Data Visualization", "❔ Help"])


# Data Visualization Tab
with tabs[2]:
    display_data_visualization({
        'filtered_data': filtered_data,
        'selected_district': selected_district,
        'selected_variable': selected_variable,
        'original_data': data
    })

# Help Tab
with tabs[3]:
    display_help()

# Forecasted Data Tab
with tabs[0]:
    forecast_df = display_forecasted_data({
        'selected_district': selected_district,
        'requires_weather': requires_weather,
        'weather_data': weather_data,
        'n_weeks': n_weeks,
        'model': model,
        'forecast_dates': forecast_dates,
        'filtered_data': filtered_data
    })


# SHAP Explanation Tab
with tabs[1]:
    if selected_district not in DISTRICT_WITHOUT_SHAP_EXPLANATION:
        if forecast_df is None:
            st.write(
                "🔄 Waiting for forecast data (First forecast dengue cases by going into the Forecasted Data Tab.)")
            st.spinner("Loading SHAP explanation...")
        else:
            display_shap_explanation({
                'filtered_data': filtered_data,
                'forecast_df': forecast_df,
                'weather_data': weather_data,
                'requires_weather': requires_weather,
                'n_weeks': n_weeks
            }, model)
    else:
        st.markdown("### 🔍 SHAP Explanation not available for this district.")
