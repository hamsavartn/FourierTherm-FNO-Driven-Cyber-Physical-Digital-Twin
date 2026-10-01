import os
import sys
import streamlit as st
import numpy as np
import pandas as pd

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dtr.twin.rc_model import TwinParams, DigitalTwin

st.set_page_config(page_title="Cable Thermal Digital Twin", layout="wide")

st.title("⚡ 11kV Underground Cable Thermal Digital Twin")
st.markdown("This dashboard visually simulates the thermal state of the 3-core cable using the parameters extracted from the datasheet.")

# Sidebar controls
st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation Duration (Days)", 1, 7, 3)
dt = 300  # 5 minute steps
n_steps = int((days * 24 * 3600) / dt)

st.sidebar.header("Phase Loads (kW)")
pa = st.sidebar.slider("Phase A Power (kW)", 0.0, 2000.0, 800.0, 50.0)
pb = st.sidebar.slider("Phase B Power (kW)", 0.0, 2000.0, 850.0, 50.0)
pc = st.sidebar.slider("Phase C Power (kW)", 0.0, 2000.0, 750.0, 50.0)

# Load twin
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'twin.yaml')
params = TwinParams.from_yaml(config_path)
twin = DigitalTwin(params)

# Initial steady state assuming 50% load
P_init = [pa * 0.5, pb * 0.5, pc * 0.5]
x0 = twin.steady_state(P_init, T_inf=20.0)

# Run simulation
st.write("Running RK4 integration...")
P_series = np.array([[pa, pb, pc]] * n_steps)

# Add some daily cyclic variation to T_inf
t_sec = np.arange(n_steps) * dt
T_inf_series = 20.0 + 2.0 * np.sin(2 * np.pi * t_sec / 86400.0)

# Add cyclic variation to load (peak in evening, low at night)
load_multiplier = 0.5 + 0.5 * np.sin(2 * np.pi * t_sec / 86400.0 - np.pi/2)
P_series = P_series * load_multiplier[:, None]

results = twin.rollout(x0, P_series, T_inf_series, dt)

# Create DataFrame for plotting
time_axis = pd.date_range("2026-01-01", periods=n_steps, freq="5min")
df_temps = pd.DataFrame({
    'Core A (°C)': results['states'][:, 0],
    'Core B (°C)': results['states'][:, 1],
    'Core C (°C)': results['states'][:, 2],
    'Sheath (°C)': results['states'][:, 4],
    'Near Soil (°C)': results['states'][:, 5],
}, index=time_axis)

st.subheader("Thermal Response Over Time")
st.line_chart(df_temps)

st.subheader("Maximum Temperatures Reached")
col1, col2, col3 = st.columns(3)
col1.metric("Max Core A", f"{results['states'][:, 0].max():.1f} °C")
col2.metric("Max Core B", f"{results['states'][:, 1].max():.1f} °C")
col3.metric("Max Core C", f"{results['states'][:, 2].max():.1f} °C")

if results['states'][:, :3].max() > params.T_max:
    st.error(f"⚠️ WARNING: Cable exceeds XLPE continuous limit of {params.T_max}°C!")
else:
    st.success(f"✅ Cable is operating within safe thermal limits ({params.T_max}°C).")

st.markdown("### Soil Moisture State")
df_moisture = pd.DataFrame({'Dry Fraction (s)': results['s']}, index=time_axis)
st.line_chart(df_moisture)
