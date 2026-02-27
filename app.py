import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# --- 1. THEME & CONFIG ---
st.set_page_config(page_title="Crypto Volatility Visualizer Pro", layout="wide")

# Custom FinTech Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #3b82f6; }
    .stExpander { background-color: #1f2937; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. FAIL-SAFE DATA ENGINE (Stage 4) ---
@st.cache_data
def load_and_clean_data():
    # Path logic to handle local and Streamlit Cloud
    possible_paths = [
        'test_crypto_full_dataset.csv', 
        '/mount/src/crypto-volatality-visualizer/test_crypto_full_dataset.csv'
    ]
    
    file_path = next((p for p in possible_paths if os.path.exists(p)), None)
            
    if file_path is None:
        return None

    try:
        df = pd.read_csv(file_path)
        
        # 1. Cleaning: Convert Timestamps
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
        
        # 2. Cleaning: Handle Missing Values (Handles the NaNs in your file)
        # Using ffill (forward fill) ensures the chart stays continuous
        df['Close'] = df['Close'].fillna(method='ffill')
        df['Volume'] = df['Volume'].fillna(df['Volume'].mean())
        df['Price'] = df['Close'] # Renaming for the app logic
        
        # 3. Advanced Metrics (Stage 4 Distinguished requirements)
        df['Volatility'] = df['Price'].rolling(window=7).std().fillna(0)
        df['SMA_20'] = df['Price'].rolling(window=20).mean().fillna(df['Price'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 3. MATH SIMULATION ENGINE (Stage 6) ---
def generate_simulation(pattern, amp, freq, drift, noise_level, steps=100):
    t = np.linspace(0, 10, steps)
    if pattern == "Cyclical Waves":
        signal = amp * (np.sin(freq * t) + 0.5 * np.cos(2 * freq * t))
    else:
        signal = np.cumsum(np.random.normal(0, noise_level, steps))
    
    slope = np.linspace(0, drift * 100, steps)
    sim_price = 25000 + signal + slope
    return t, sim_price

# --- 4. APP UI ---
st.title("📈 Crypto Volatility Visualizer")
st.sidebar.title("🎮 Controls")
app_mode = st.sidebar.selectbox("Select View", ["Real Market Analysis", "Mathematical Simulation"])

if app_mode == "Real Market Analysis":
    df = load_and_clean_data()
    
    if df is not None:
        # Metrics Header
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Data Points", len(df)) # Shows all 182 entries!
        c2.metric("Latest Price", f"${df['Price'].iloc[-1]:,.2f}")
        c3.metric("Avg Sentiment", f"{df['Sentiment_Score'].mean():.2f}")
        c4.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")

        # STAGE 4: DATA EXPLORATION (No longer limited to 10!)
        with st.expander("📂 Stage 4: Full Dataset Exploration"):
            st.write(f"Showing all {len(df)} rows from the CSV:")
            st.dataframe(df) # Shows the entire table

        # STAGE 5: VISUALIZATIONS
        st.subheader("Price Volatility Map (Full Dataset)")
        fig = go.Figure()
        # High-Low Range Shading
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['High'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Low'], line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(59, 130, 246, 0.1)', name='Daily Range'))
        # Main Price Line
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Price'], line=dict(color='#3b82f6', width=3), name='BTC Price'))
        
        fig.update_layout(template="plotly_dark", hovermode="x unified", height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Volume Analysis
        st.subheader("Trading Activity")
        vol_fig = px.bar(df, x='Timestamp', y='Volume', color='Volume', color_continuous_scale='GnBu')
        vol_fig.update_layout(template="plotly_dark")
        st.plotly_chart(vol_fig, use_container_width=True)

    else:
        st.error("🚨 Dataset not found. Check if 'test_crypto_full_dataset.csv' is in your GitHub repo!")

else:
    # MATHEMATICAL SIMULATION
    st.title("🧪 Mathematical Modeling")
    st.sidebar.markdown("---")
    patt = st.sidebar.radio("Pattern", ["Cyclical Waves", "Random Jumps"])
    amp = st.sidebar.slider("Amplitude", 100, 5000, 1500)
    freq = st.sidebar.slider("Frequency", 0.5, 10.0, 2.0)
    drift = st.sidebar.slider("Drift", -50, 50, 5)
    noise = st.sidebar.slider("Noise", 0, 500, 200)

    t, sim_price = generate_simulation(patt, amp, freq, drift, noise)
    
    sim_fig = go.Figure()
    sim_fig.add_trace(go.Scatter(x=t, y=sim_price, line=dict(color='#10b981', width=4), name='Simulated Price'))
    sim_fig.update_layout(title=f"Math Model: {patt}", template="plotly_dark")
    st.plotly_chart(sim_fig, use_container_width=True)

