import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path

# --- 1. THEME & CONFIG ---
st.set_page_config(page_title="Pro Crypto Visualizer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for FinTech styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. BOMB-PROOF DATA ENGINE (Stage 4 Fix) ---
@st.cache_data
def load_and_clean_data():
    # AUTOMATIC PATH FIX: Finds the file in the current folder
    current_dir = Path(__file__).parent
    file_path = current_dir / "test_crypto_full_dataset.csv"
    
    try:
        df = pd.read_csv(file_path)
        
        # Cleaning & Formatting (Requirement for 'Distinguished')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        # Handle Missing Values (Filling with average so charts don't break)
        df['Close'] = df['Close'].fillna(df['Close'].mean())
        df['Volume'] = df['Volume'].fillna(df['Volume'].mean())
        
        # Rename/Standardize for clarity
        df = df.rename(columns={'Close': 'Price'})
        
        # Advanced Technical Metrics
        df['Volatility'] = df['Price'].rolling(window=7).std().fillna(0)
        df['SMA_20'] = df['Price'].rolling(window=20).mean().fillna(df['Price'])
        
        return df
    except FileNotFoundError:
        st.error(f"❌ File Not Found! Ensure 'test_crypto_full_dataset.csv' is in: {current_dir}")
        return None

# --- 3. MATH SIMULATION ENGINE (Stage 6) ---
def generate_simulation(pattern, amp, freq, drift, noise_level, steps=100):
    t = np.linspace(0, 10, steps)
    if pattern == "Cyclical (Fourier)":
        signal = amp * (np.sin(freq * t) + 0.5 * np.cos(2 * freq * t))
    else:
        signal = np.cumsum(np.random.normal(0, noise_level, steps))
    
    slope = np.linspace(0, drift * 100, steps)
    sim_price = 20000 + signal + slope
    return t, sim_price

# --- 4. SIDEBAR ---
st.sidebar.title("🛠️ Visualizer Settings")
mode = st.sidebar.selectbox("Dashboard Mode", ["Live Market Analysis", "Mathematical Modeling"])

# --- 5. EXECUTION ---
if mode == "Live Market Analysis":
    df = load_and_clean_data()
    
    if df is not None:
        st.title("📊 Real-Time Market Volatility")
        
        # Stage 4 Requirement: Data Preview
        with st.expander("🔍 View Raw Data Exploration (df.head)"):
            st.write(df.head())

        # Header Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}")
        m2.metric("24h High", f"${df['High'].max():,.2f}")
        m3.metric("Avg Sentiment", f"{df['Sentiment_Score'].mean():.2f}")
        m4.metric("Market Volatility", f"{df['Volatility'].iloc[-1]:.2f}")

        # STAGE 5 VISUALS: Interactive Price & Volatility Zone
        st.subheader("Interactive Price & Volatility Map")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['High'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Low'], line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(59, 130, 246, 0.1)', name='Volatility Zone'))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Price'], line=dict(color='#3b82f6', width=3), name='Close Price'))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SMA_20'], line=dict(color='#f59e0b', dash='dot'), name='Trendline'))
        
        fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Volume Chart
        st.subheader("Trading Volume Analysis")
        vol_fig = px.bar(df, x='Timestamp', y='Volume', color='Volume', color_continuous_scale='Blues')
        st.plotly_chart(vol_fig, use_container_width=True)

else:
    # MATHEMATICAL MODELING MODE
    st.title("🧪 Mathematical Swing Simulation")
    st.sidebar.subheader("Math Parameters")
    pattern = st.sidebar.radio("Swing Pattern", ["Cyclical (Fourier)", "Stochastic (Noise)"])
    amp = st.sidebar.slider("Amplitude", 50, 5000, 1000)
    freq = st.sidebar.slider("Frequency", 0.1, 10.0, 2.0)
    drift = st.sidebar.slider("Drift (Integral)", -50, 50, 10)
    noise = st.sidebar.slider("Noise", 0, 500, 100)

    t, sim_price = generate_simulation(pattern, amp, freq, drift, noise)
    
    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(x=t, y=sim_price, line=dict(color='#10b981', width=4), name='Simulated Result'))
    sim_fig.update_layout(title=f"Model: {pattern}", template="plotly_dark")
    st.plotly_chart(sim_fig, use_container_width=True)
