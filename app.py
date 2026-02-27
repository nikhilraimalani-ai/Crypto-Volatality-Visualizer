import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# --- 1. THEME & CONFIG (For "Distinguished" Visuals) ---
st.set_page_config(page_title="Crypto Volatility Visualizer Pro", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #3b82f6; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FAIL-SAFE DATA ENGINE (Stage 4) ---
@st.cache_data
def load_and_clean_data():
    # List of possible file locations (Local and Cloud)
    possible_paths = [
        'test_crypto_full_dataset.csv', 
        '/mount/src/crypto-volatality-visualizer/test_crypto_full_dataset.csv',
        '../test_crypto_full_dataset.csv'
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
            
    if file_path is None:
        return None

    try:
        df = pd.read_csv(file_path)
        
        # Data Cleaning (Stage 4 Requirement)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        # Handling Missing Values (Handles the NaNs in your specific CSV)
        df['Close'] = df['Close'].fillna(method='ffill')
        df['Volume'] = df['Volume'].fillna(df['Volume'].mean())
        
        # Standardizing column for the app
        df['Price'] = df['Close']
        
        # Advanced Technical Metrics for extra marks
        df['Volatility'] = df['Price'].rolling(window=7).std().fillna(0)
        df['SMA_20'] = df['Price'].rolling(window=20).mean().fillna(df['Price'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 3. MATH SIMULATION ENGINE (Stage 6) ---
def generate_simulation(pattern, amp, freq, drift, noise_level, steps=100):
    t = np.linspace(0, 10, steps)
    
    # Sine/Cosine for Waves
    if pattern == "Cyclical Waves":
        signal = amp * (np.sin(freq * t) + 0.5 * np.cos(2 * freq * t))
    # Random Jumps (Stochastic)
    else:
        signal = np.cumsum(np.random.normal(0, noise_level, steps))
    
    # Integral-like Drift (Slope)
    slope = np.linspace(0, drift * 100, steps)
    sim_price = 25000 + signal + slope
    return t, sim_price

# --- 4. APP STRUCTURE ---
st.title("📈 Crypto Volatility Visualizer")
st.markdown("---")

# Sidebar Controls (Stage 6)
st.sidebar.title("🎮 Dashboard Controls")
app_mode = st.sidebar.selectbox("Select View", ["Live Market Data", "Mathematical Simulation"])

if app_mode == "Live Market Data":
    df = load_and_clean_data()
    
    if df is not None:
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}")
        m2.metric("Market Volatility", f"{df['Volatility'].iloc[-1]:.2f}")
        m3.metric("Social Sentiment", f"{df['Sentiment_Score'].mean():.2f}")
        m4.metric("Trading Volume", f"{df['Volume'].iloc[-1]:.0f}")

        # Stage 4 Requirement: Head Preview
        with st.expander("📂 Stage 4: Data Exploration (df.head)"):
            st.dataframe(df.head(10))

        # Stage 5: Visualizations (Interactive Plotly)
        st.subheader("Price Volatility & Trend Analysis")
        fig = go.Figure()
        # High-Low Range Shading
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['High'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Low'], line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(59, 130, 246, 0.1)', name='Volatility Range'))
        # Main Price
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Price'], line=dict(color='#3b82f6', width=3), name='BTC Price'))
        # 20-Day SMA
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SMA_20'], line=dict(color='#f59e0b', dash='dot'), name='20D Trend'))
        
        fig.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Volume Chart
        st.subheader("Volume Analysis")
        vol_fig = px.bar(df, x='Timestamp', y='Volume', color='Volume', color_continuous_scale='GnBu')
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.error("🚨 Dataset not found. Please ensure 'test_crypto_full_dataset.csv' is uploaded to your GitHub repo.")

else:
    # MATHEMATICAL SIMULATION
    st.title("🧪 Mathematical Price Modeling")
    st.sidebar.markdown("---")
    patt = st.sidebar.radio("Simulation Pattern", ["Cyclical Waves", "Random Jumps"])
    amp = st.sidebar.slider("Amplitude (A)", 100, 5000, 1500)
    freq = st.sidebar.slider("Frequency (ω)", 0.5, 10.0, 2.0)
    drift = st.sidebar.slider("Drift (Integral Slope)", -50, 50, 5)
    noise = st.sidebar.slider("Random Noise Level", 0, 500, 200)

    t, sim_price = generate_simulation(patt, amp, freq, drift, noise)
    
    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(x=t, y=sim_price, line=dict(color='#10b981', width=4), name='Simulated Price'))
    sim_fig.update_layout(title=f"Math Model: {patt}", template="plotly_dark", xaxis_title="Time (t)", yaxis_title="Price ($)")
    st.plotly_chart(sim_fig, use_container_width=True)
    
    st.info("This simulation uses Sine/Cosine for wave-like oscillations and a linear drift to represent long-term market trends.")
