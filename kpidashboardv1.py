import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(page_title="KPI Dashboard", layout="wide")

# Title for the app
st.title("KPI Dashboard")

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
start_date = datetime(2021, 3, 1)
end_date = datetime(2022, 12, 1)
dates = pd.date_range(start=start_date, end=end_date, freq='MS')

# Create base data with seasonal patterns
def generate_kpi_data():
    # Number of records
    n = len(dates)
    
    # Generate sales with seasonal pattern, trend, and random variation
    base_sales = np.linspace(45000, 85000, n)  # Upward trend
    seasonal = 15000 * np.sin(np.linspace(0, 4 * np.pi, n))  # Seasonal pattern
    noise = np.random.normal(0, 5000, n)  # Random variation
    sales = base_sales + seasonal + noise
    
    # Generate other metrics
    regions = ['West', 'Central', 'South']
    data = []
    
    for region in regions:
        # Regional variation factors
        if region == 'West':
            sales_factor = 1.2
            profit_factor = 1.1
        elif region == 'Central':
            sales_factor = 1.0
            profit_factor = 1.0
        else:  # South
            sales_factor = 0.8
            profit_factor = 0.9
            
        for i, date in enumerate(dates):
            # Calculate regional sales
            regional_sales = sales[i] * sales_factor * (1 + np.random.normal(0, 0.05))
            
            # Calculate other metrics
            profit_ratio = (8.5 + i * 0.1) * profit_factor * (1 + np.random.normal(0, 0.02))
            planned_sales = regional_sales * (1 + np.random.normal(0.1, 0.02))
            discount = 15 + np.random.normal(0, 1)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'region': region,
                'sales': round(regional_sales, 2),
                'planned_sales': round(planned_sales, 2),
                'profit_ratio': round(profit_ratio, 2),
                'discounts': round(discount, 2)
            })
    
    return pd.DataFrame(data)

# Generate the data
df = generate_kpi_data()

# Display data and statistics in Streamlit
st.subheader("Generated KPI Data")
st.dataframe(df)

st.subheader("Summary Statistics")
st.write(df.describe())

# Save to CSV and provide a download button
st.download_button(
    label="Download CSV",
    data=df.to_csv(index=False),
    file_name="kpi_dashboard_data.csv",
    mime="text/csv"
)
