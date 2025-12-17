import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(page_title="UK Agile Power Price Tracker", layout="wide")

# Constants
API_BASE_URL = "https://prices.fly.dev/api"
REGIONS = {
    "National Average": "A",
    "Eastern England": "E",
    "East Midlands": "F",
    "London": "C",
    "Merseyside & N. Wales": "D",
    "West Midlands": "G",
    "North Eastern England": "B",
    "North Western England": "H",
    "Southern England": "J",
    "South Eastern England": "K",
    "Southern Wales": "L",
    "South Western England": "M",
    "Yorkshire": "N",
    "Southern Scotland": "P",
    "Northern Scotland": "N"
}

st.title("⚡ UK Power Prices (Agile Forecast)")
st.markdown("Visualizing Octopus Agile price forecasts and historical data.")

# Sidebar for controls
st.sidebar.header("Settings")
selected_region_name = st.sidebar.selectbox("Select DNO Region", list(REGIONS.keys()), index=3) # Default London
region_code = REGIONS[selected_region_name]
days_to_fetch = st.sidebar.slider("Days of data", 1, 14, 3)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_power_data(region, days):
    try:
        url = f"{API_BASE_URL}/{region}?days={days}&forecast_count=1"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        # The API returns a list of forecasts; we take the most recent one
        prices_list = data[0]['prices']
        df = pd.DataFrame(prices_list)
        
        # Rename columns for clarity and convert time
        df['date_time'] = pd.to_datetime(df['date_time'])
        df = df.rename(columns={
            'agile_pred': 'Predicted Price (p/kWh)',
            'agile_low': 'Low Estimate',
            'agile_high': 'High Estimate'
        })
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch Data
with st.spinner('Fetching latest price data...'):
    df = fetch_power_data(region_code, days_to_fetch)

if df is not None:
    # Key Metrics
    now = datetime.now()
    # Find the row closest to "now"
    current_row = df.iloc[(df['date_time'] - now).abs().argsort()[:1]]
    current_price = current_row['Predicted Price (p/kWh)'].values[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Est. Price", f"{current_price:.2f} p/kWh")
    col2.metric("Max Price (Period)", f"{df['Predicted Price (p/kWh)'].max():.2f} p/kWh")
    col3.metric("Min Price (Period)", f"{df['Predicted Price (p/kWh)'].min():.2f} p/kWh")

    # Main Graph
    st.subheader(f"Price Forecast for {selected_region_name}")
    
    # Create interactive Plotly chart
    fig = px.line(df, x='date_time', y='Predicted Price (p/kWh)', 
                  title="Electricity Price Forecast (inc. VAT)",
                  labels={'date_time': 'Time', 'Predicted Price (p/kWh)': 'Price (p/kWh)'},
                  template="plotly_dark")
    
    # Add a horizontal line at 0 for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Shade the area between low and high estimates if available
    if 'Low Estimate' in df.columns and 'High Estimate' in df.columns:
        fig.add_scatter(x=df['date_time'], y=df['High Estimate'], name='High Est', line=dict(width=0), showlegend=False)
        fig.add_scatter(x=df['date_time'], y=df['Low Estimate'], name='Low Est', fill='tonexty', 
                        fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0), showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # Data Table
    with st.expander("View Raw Data Table"):
        st.dataframe(df.sort_values('date_time', ascending=False), use_container_width=True)
else:
    st.info("Unable to load data. Please check the API status or your internet connection.")

st.caption("Data source: AgilePredict (https://prices.fly.dev/)")
