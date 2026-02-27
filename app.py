import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# --- 1. THEME & CONFIG ---
st.set_page_config(page_title="Pro Crypto Visualizer", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a "FinTech" look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ADVANCED DATA ENGINE (Stage 4) ---
@st.cache_data
def process_data(file_path):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Advanced Metrics: Volatility (Rolling Std Dev)
    df['Volatility'] = df['Close'].rolling(window=7).std()
    # Simple Moving Average for trend analysis
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    return df.dropna()

# --- 3. MATH SIMULATION ENGINE (Stage 6) ---
def generate_simulation(pattern, amp, freq, drift, noise_level, steps=100):
    t = np.linspace(0, 10, steps)
    
    if pattern == "Cyclical (Fourier)":
        # Multi-wave interference: Sin(x) + Cos(2x)
        signal = amp * (np.sin(freq * t) + 0.5 * np.cos(2 * freq * t))
    else:
        # Stochastic / Geometric Brownian Motion logic
        signal = np.cumsum(np.random.normal(0, noise_level, steps))
    
    # Integration of drift (Stage 6: Long-term slope)
    slope = np.linspace(0, drift * 100, steps)
    simulated_price = 20000 + signal + slope
    
    return t, simulated_price

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Visualizer Settings")
st.sidebar.subheader("Data Source")
data_mode = st.sidebar.selectbox("Dashboard Mode", ["Live Market Analysis", "Mathematical Modeling"])

# --- 5. DASHBOARD MODES ---
if data_mode == "Live Market Analysis":
    try:
        df = process_data('test_crypto_full_dataset.csv')
        
        # Header Metrics
        st.title("📊 Real-Time Market Volatility")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"${df['Close'].iloc[-1]:,.2f}")
        m2.metric("24h High", f"${df['High'].max():,.2f}")
        m3.metric("Avg Sentiment", f"{df['Sentiment_Score'].mean():.2f}")
        m4.metric("Market Volatility", f"{df['Volatility'].iloc[-1]:.2f}")

        # Main Visualization (High vs Low / Volume)
        st.subheader("Interactive Price & Volatility Map")
        fig = go.Figure()
        
        # Shaded Volatility Zone (High-Low Fill)
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['High'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Low'], line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(59, 130, 246, 0.1)', name='Volatility Zone'))
        
        # Main Price Line
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Close'], line=dict(color='#3b82f6', width=3), name='Close Price'))
        
        # Add SMA Trendline
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SMA_20'], line=dict(color='#f59e0b', dash='dot'), name='20-Day Trend'))
        
        fig.update_layout(template="plotly_dark", hovermode="x unified", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Volume & Sentiment Analysis
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Volume Analysis")
            vol_fig = px.bar(df, x='Timestamp', y='Volume', color='Volume', color_continuous_scale='Blues')
            st.plotly_chart(vol_fig, use_container_width=True)
        with c2:
            st.subheader("Sentiment vs Price Change")
            # Showing how social sentiment affects the market (Advanced Feature)
            df['Price_Change'] = df['Close'].pct_change()
            sent_fig = px.scatter(df, x='Sentiment_Score', y='Price_Change', size='Volume', color='Exchange', trendline="ols")
            st.plotly_chart(sent_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Connect your dataset to begin. {e}")

else:
    # MATHEMATICAL MODELING MODE
    st.title("🧪 Mathematical Swing Simulation")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Math Parameters")
    pattern = st.sidebar.radio("Swing Pattern", ["Cyclical (Fourier)", "Stochastic (Noise)"])
    amp = st.sidebar.slider("Amplitude (Swing Size)", 50, 5000, 1000)
    freq = st.sidebar.slider("Frequency (Speed)", 0.1, 10.0, 2.0)
    drift = st.sidebar.slider("Drift (Slope)", -50, 50, 10)
    noise = st.sidebar.slider("Random Noise Level", 0, 500, 100)

    t, sim_price = generate_simulation(pattern, amp, freq, drift, noise)
    
    # Simulation Visual
    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(x=t, y=sim_price, line=dict(color='#10b981', width=4), name='Simulated Result'))
    sim_fig.update_layout(title=f"Mathematical Model: {pattern}", template="plotly_dark", xaxis_title="Time Function (t)", yaxis_title="Calculated Price")
    st.plotly_chart(sim_fig, use_container_width=True)
    
    st.markdown(f"""
    > **Mathematical Explanation:** This model uses a **{pattern}** function. 
    > It calculates price as: $P(t) = P_0 + f(t) + \int drift \cdot dt + \epsilon$
    > where $f(t)$ is the swing pattern and $\epsilon$ is the random noise.
    """)
