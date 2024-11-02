import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

# Set Streamlit page configuration
st.set_page_config(page_title="Enhanced KPI Dashboard", layout="wide")

# Title for the app
st.title("Enhanced KPI Dashboard")

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
start_date = datetime(2021, 3, 1)
end_date = datetime(2022, 12, 1)
dates = pd.date_range(start=start_date, end=end_date, freq='MS')

# Create base data with seasonal patterns
def generate_kpi_data():
    n = len(dates)
    base_sales = np.linspace(45000, 85000, n)  # Upward trend
    seasonal = 15000 * np.sin(np.linspace(0, 4 * np.pi, n))  # Seasonal pattern
    noise = np.random.normal(0, 5000, n)  # Random variation
    sales = base_sales + seasonal + noise
    
    regions = ['West', 'Central', 'South']
    data = []
    
    for region in regions:
        if region == 'West':
            sales_factor = 1.2
            profit_factor = 1.1
        elif region == 'Central':
            sales_factor = 1.0
            profit_factor = 1.0
        else:
            sales_factor = 0.8
            profit_factor = 0.9
            
        for i, date in enumerate(dates):
            regional_sales = sales[i] * sales_factor * (1 + np.random.normal(0, 0.05))
            profit_ratio = (8.5 + i * 0.1) * profit_factor * (1 + np.random.normal(0, 0.02))
            planned_sales = regional_sales * (1 + np.random.normal(0.1, 0.02))
            discount = 15 + np.random.normal(0, 1)
            
            data.append({
                'date': date,
                'region': region,
                'sales': round(regional_sales, 2),
                'planned_sales': round(planned_sales, 2),
                'profit_ratio': round(profit_ratio, 2),
                'discounts': round(discount, 2)
            })
    
    return pd.DataFrame(data)

# Generate the data
df = generate_kpi_data()

# Sidebar filters
st.sidebar.header("Filter Options")
selected_region = st.sidebar.multiselect("Select Region", options=df["region"].unique(), default=df["region"].unique())
selected_dates = st.sidebar.date_input("Select Date Range", [start_date, end_date])

# Filter data based on user selections
filtered_df = df[(df["region"].isin(selected_region)) & 
                 (df["date"] >= pd.to_datetime(selected_dates[0])) & 
                 (df["date"] <= pd.to_datetime(selected_dates[1]))]

# Display filtered data and statistics
st.subheader("Filtered KPI Data")
st.dataframe(filtered_df)

st.subheader("Summary Statistics")
st.write(filtered_df.describe())

# KPI Visualizations
st.subheader("KPI Trends by Region")

# Sales over time
fig_sales = px.line(filtered_df, x="date", y="sales", color="region",
                    title="Sales Over Time by Region",
                    labels={"sales": "Sales ($)", "date": "Date"})
st.plotly_chart(fig_sales, use_container_width=True)

# Profit ratio over time
fig_profit = px.line(filtered_df, x="date", y="profit_ratio", color="region",
                     title="Profit Ratio Over Time by Region",
                     labels={"profit_ratio": "Profit Ratio (%)", "date": "Date"})
st.plotly_chart(fig_profit, use_container_width=True)

# Download button
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_kpi_dashboard_data.csv",
    mime="text/csv"
)
