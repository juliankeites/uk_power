import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Battery Optimization", layout="wide")

@st.cache_data
def get_solar_forecast(lat=51.5, lon=-0.1, forecast_date=date.today()):
    """Realistic UK solar pattern (kW)"""
    times = pd.date_range(forecast_date, periods=24, freq='H', tz='Europe/London')
    # December UK solar: peaks ~10am-2pm, max 3kW for 4kW system
    solar_pattern = np.maximum(0, 3 * np.sin(np.pi * (np.arange(24) - 8) / 8))
    return pd.Series(solar_pattern, index=times)

@st.cache_data  
def get_baseload():
    """Typical UK household baseload (kW)"""
    times = pd.date_range(date.today(), periods=24, freq='H', tz='Europe/London')
    pattern = [0.8,0.7,0.6,0.6,0.7,0.9,1.2,1.5,1.8,2.0,2.2,2.1,2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1,1.0,0.9]
    return pd.Series(pattern, index=times)

def run_app():
    st.title("🔋 Powerwall Optimization")
    
    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=51.5, step=0.1)
        lon = st.number_input("Longitude", value=-0.1, step=0.1)
    with col2:
        battery_kwh = st.number_input("Battery (kWh)", value=13.5, step=1.0)
        date_input = st.date_input("Date", value=date.today())
    
    # FIXED OPTIMIZATION - NO INDEX ERRORS
    solar = get_solar_forecast(lat, lon, date_input)
    base = get_baseload()
    
    # CRITICAL: Align indices first
    common_idx = solar.index.intersection(base.index)
    solar = solar.loc[common_idx]
    base = base.loc[common_idx]
    
    excess = solar - base  # Same index/length guaranteed
    charge_hours = excess[excess > 0]  # Direct indexing - NO pd.Series() wrapper
    
    # Simple battery logic
    battery_flow = pd.Series(0.0, index=solar.index)
    for i in range(len(solar)):
        if excess.iloc[i] > 0:
            battery_flow.iloc[i] = min(5.0, excess.iloc[i])  # 5kW charge limit
    
    net_grid = base + battery_flow - solar
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Solar Generation", f"{solar.sum():.1f} kWh")
    col2.metric("Excess Solar", f"{charge_hours.sum():.1f} kWh") 
    col3.metric("Charge Hours", len(charge_hours))
    
    # Plot
    fig, ax = st.pyplot(plt.subplots(2,1,figsize=(12,8)))
    ax[0].plot(solar.index, solar.values, 'gold', linewidth=3, label='Solar')
    ax[0].plot(base.index, base.values, 'lightblue', linewidth=3, label='Baseload')
    ax[0].plot(battery_flow.index, battery_flow.values, 'orange', linewidth=4, label='Battery')
    ax[0].legend(); ax[0].grid(True, alpha=0.3)
    
    ax[1].bar(net_grid.index, net_grid.values, color='red', alpha=0.7, label='Grid Import')
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Table
    st.subheader("Best Charge Times")
    charge_df = pd.DataFrame({
        'Time': charge_hours.index.strftime('%H:%M'),
        'Excess (kW)': charge_hours.round(1)
    })
    st.dataframe(charge_df)

if __name__ == "__main__":
    run_app()
