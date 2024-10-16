import pdfplumber
import re
import pandas as pd
from dateutil import parser
import os

pattern = re.compile(r'Week (\d{2})\s*\(?\s*(\d{1,2}(?:st|nd|rd|th)?)\s*(\w+)?\s*(?:–|-)\s*(\d{1,2}(?:st|nd|rd|th)?)\s*(\w*)\s+(\d{4})\)?')

# List to store the extracted data
extracted_data = []

# Function to process a single PDF file
def process_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[0]
        # Extract text from the page
        text = page.extract_text()
        table_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "explicit_vertical_lines": [],
                "explicit_horizontal_lines": [],
                "snap_tolerance": 5,
                "join_tolerance": 3,
                "edge_min_length": 10,
                "min_words_vertical": 2,
                "min_words_horizontal": 2,
                "intersection_tolerance": 5,
                "text_x_tolerance": 2,
                "text_y_tolerance": 2,
            }
        table = page.extract_table(table_settings=table_settings)
 
        
        # Find all matches in the text
        matches = pattern.findall(text)
        for match in matches:
            if match[2] == '':
                week_number, start_day, _, end_day, month, year = match
                start_date_str = f"{start_day} {month}"
                end_date_str = f"{end_day} {month}"
                
            else:
                week_number, start_day, start_month, end_day, end_month, year = match
                start_date_str = f"{start_day} {start_month}"
                end_date_str = f"{end_day} {end_month}"
            
            # Parse dates to convert to desired format
            start_date = parser.parse(start_date_str).strftime('%Y-%m-%d')
            end_date = parser.parse(end_date_str).strftime('%Y-%m-%d')

            # Process the table data
            for row in table[4:]:  # Assuming header and initial rows need to be skipped
                if row and isinstance(row[0], str) and row[0] != '' and row[0] != 'Total':
                    # Extract district and number of cases
                    district = row[0]
                    number_of_cases = row[4] if len(row) > 4 else None
                    
                    # Append the extracted data
                    extracted_data.append({
                        "Year": year,
                        "Week": week_number,
                        "Week_Start_Date": start_date,
                        "Week_End_Date": end_date,
                        "District": district,
                        "Number_of_Cases": number_of_cases
                    })
                    
            

def extract_pdf(pdf_files):
    for file in pdf_files:
        filename = file.name
        
        match = re.search(r'Week (\d+)', filename)
        if match:
            week_number = match.group(1)  # Extract the week number
            print(f'Week Number: {week_number}')
        else:
            print("Week number not found.")

        process_pdf(file)
    return extracted_data


def aggregate_yearly_cases_all_districts(data, district):
    # Filter the data for the selected district
    print(data)
    district_data = data[data['District'] == district]

    # Convert the 'date' column to datetime format if it's not already
    district_data['Week_End_Date'] = pd.to_datetime(district_data['Week_End_Date'])

    # Extract the year from the date and group by year and district to get the count of cases
    yearly_data = data.groupby([data['Week_End_Date'].dt.year, 'District'])['Number_of_Cases'].sum().reset_index()
    yearly_data.columns = ['Year', 'District', 'Number_of_Cases']  # Rename columns for clarity


    return yearly_data

def aggregate_weekly_cases(data, selected_district):
    # Convert the 'date' column to datetime format if it's not already
    data['Week_End_Date'] = pd.to_datetime(data['Week_End_Date'])
    
    # Filter data for the selected district
    district_data = data[data['District'] == selected_district]

    # Set date as index
    district_data.set_index('Week_End_Date', inplace=True)

    # Resample to weekly frequency and sum the cases
    weekly_data = district_data.resample('W')['Number_of_Cases'].sum().reset_index()

    # Extract year for plotting
    weekly_data['Year'] = weekly_data['Week_End_Date'].dt.year
    
    return weekly_data
