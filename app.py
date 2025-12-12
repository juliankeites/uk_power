import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pvlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Battery Optimization", layout="wide")

def get_solar_forecast(latitude, longitude, date):
    """Get 24h solar forecast using pvlib"""
    try:
        # Simplified pvlib forecast - replace with your forecast.solar API
        times = pd.date_range(date, periods=24, freq='H', tz='Europe/London')
        solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        # Dummy forecast - replace with real API data
        clearsky = pvlib.clearsky.ineichen(times, latitude, longitude)
        solar_kw = clearsky['ghi'] / 1000 * 10  # Scale to kW system
        return pd.Series(solar_kw.values, index=times)
    except:
        # Fallback hourly pattern
        times = pd.date_range(date, periods=24, freq='H', tz='Europe/London')
        solar_pattern = np.maximum(0, np.sin(np.linspace(0, 2*np.pi, 24)) * 8)
        return pd.Series(solar_pattern, index=times)

def get_typical_baseload():
    """Typical UK household baseload pattern (kW)"""
    times = pd.date_range('2025-12-12', periods=24, freq='H', tz='Europe/London')
    pattern = [0.8, 0.7, 0.6, 0.6, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 
               2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 
               1.2, 1.1, 1.0, 0.9]
    return pd.Series(pattern, index=times)

@st.cache_data
def optimize_battery(solar_forecast, baseload, battery_capacity_kwh=13.5, 
                    charge_rate_kw=5.0, efficiency=0.95):
    """Optimize Powerwall charge/discharge schedule"""
    # Align indices
    common_idx = solar_forecast.index.intersection(baseload.index)
    solar = solar_forecast.loc[common_idx].fillna(0)
    base = baseload.loc[common_idx].fillna(1.0)
    
    # Calculate excess solar
    excess_solar = solar - base
    excess_mask = excess_solar > 0
    
    # Best charge hours (FIXED INDEXING)
    best_charge_hours = excess_solar[excess_mask]
    
    # Simple optimization: charge during peak excess
    charge_schedule = pd.Series(0.0, index=solar.index)
    soc = 0.0  # State of charge
    
    for i, t in enumerate(solar.index):
        excess = excess_solar.iloc[i]
        
        # Charge phase: prioritize excess solar
        if excess > 0 and soc < 1.0:
            charge_amount = min(charge_rate_kw * efficiency, excess, 
                              (1.0 - soc) * battery_capacity_kwh)
            charge_schedule.iloc[i] = charge_amount / efficiency
            soc += charge_amount / battery_capacity_kwh
        
        # Discharge phase: cover baseload when solar insufficient
        elif solar.iloc[i] < base.iloc[i] and soc > 0.1:
            discharge_needed = base.iloc[i] - solar.iloc[i]
            discharge_amount = min(discharge_needed / efficiency, 
                                 soc * battery_capacity_kwh, charge_rate_kw)
            charge_schedule.iloc[i] = -discharge_amount
            soc -= discharge_amount / battery_capacity_kwh
    
    # Net grid import
    net_grid = base + charge_schedule - solar
    savings = -net_grid[net_grid < 0].sum()  # Energy saved from grid
    
    return {
        'solar': solar,
        'baseload': base,
        'excess_solar': excess_solar,
        'charge_schedule': charge_schedule,
        'net_grid': net_grid,
        'best_charge_hours': best_charge_hours,
        'total_savings_kwh': savings,
        'battery_capacity': battery_capacity_kwh
    }

def run_app():
    st.title("🔋 Tesla Powerwall Optimization")
    
    # Sidebar inputs
    st.sidebar.header("System Settings")
    lat = st.sidebar.number_input("Latitude", value=51.5, step=0.1)
    lon = st.sidebar.number_input("Longitude", value=-0.1, step=0.1)
    battery_kwh = st.sidebar.number_input("Battery Capacity (kWh)", 
                                         value=13.5, step=1.0)
    charge_rate = st.sidebar.number_input("Charge Rate (kW)", value=5.0, step=0.5)
    
    date = st.sidebar.date_input("Forecast Date", value=pd.Timestamp.now().date())
    
    # Main optimization
    with st.spinner("Optimizing battery schedule..."):
        solar_forecast = get_solar_forecast(lat, lon, date)
        baseload = get_typical_baseload()
        results = optimize_battery(solar_forecast, baseload, battery_kwh, charge_rate)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Solar (kWh)", f"{results['solar'].sum():.1f}")
    with col2:
        st.metric("Excess Solar (kWh)", f"{results['excess_solar'][results['excess_solar']>0].sum():.1f}")
    with col3:
        st.metric("Battery Savings (kWh)", f"{results['total_savings_kwh']:.1f}")
    with col4:
        st.metric("Best Charge Hours", len(results['best_charge_hours']))
    
    # Plotting
    fig = make_subplots(rows=3, cols=1, 
                       subplot_titles=('Power Flows', 'Excess Solar & Charge Schedule', 'Net Grid Import'),
                       vertical_spacing=0.08,
                       row_heights=[0.4, 0.4, 0.2])
    
    # Plot 1: Power flows
    fig.add_trace(go.Scatter(x=results['solar'].index, y=results['solar'].values,
                            name='Solar', line=dict(color='gold', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['baseload'].index, y=results['baseload'].values,
                            name='Baseload', line=dict(color='lightblue', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['charge_schedule'].index, y=results['charge_schedule'].values,
                            name='Battery', line=dict(color='orange', width=4)), row=1, col=1)
    
    # Plot 2: Excess & optimal charging
    fig.add_trace(go.Scatter(x=results['excess_solar'].index, y=results['excess_solar'].values,
                            name='Excess Solar', line=dict(color='green')), row=2, col=1)
    charge_pos = results['charge_schedule'] > 0
    fig.add_trace(go.Scatter(x=results['charge_schedule'][charge_pos].index, 
                            y=results['charge_schedule'][charge_pos].values,
                            name='Charge Periods', line=dict(color='darkgreen', width=6)), row=2, col=1)
    
    # Plot 3: Net grid
    fig.add_trace(go.Bar(x=results['net_grid'].index, y=results['net_grid'].values,
                        name='Grid Import', marker_color='red'), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True, title_text="24h Powerwall Optimization")
    st.plotly_chart(fig, use_container_width=True)
    
    # Best charge hours table
    st.subheader("Optimal Charging Schedule")
    charge_table = pd.DataFrame({
        'Hour': results['best_charge_hours'].index.strftime('%H:%M'),
        'Excess Solar (kW)': results['best_charge_hours'].round(2),
        'Charge Rate (kW)': np.minimum(results['best_charge_hours'], charge_rate).round(2)
    })
    st.dataframe(charge_table, use_container_width=True)
    
    # Debug info (remove in production)
    with st.expander("Debug Info"):
        st.write(f"Solar shape: {results['solar'].shape}")
        st.write(f"Baseload shape: {results['baseload'].shape}")
        st.write(f"Excess mask sum: {(results['excess_solar'] > 0).sum()}")

if __name__ == "__main__":
    run_app()
