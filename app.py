import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

# --- Core Logic ---

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = [d.strftime("%Y-%m-%d") for d in sorted_dates if d >= cutoff_date]
    if not arr: raise ValueError("No date 45 days or more in the future found.")
    return arr

def yang_zhang(price_data):
    # Debug: Print the first few rows of price data
    st.write("DEBUG: Price Data Head", price_data.head())
    
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    window = 30
    close_vol = (log_cc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = (log_oc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
    
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(252)
    
    rv = result.iloc[-1]
    st.write(f"DEBUG: Calculated RV: {rv}")
    return rv

def build_term_structure(days, ivs):
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte))

def compute_recommendation(ticker):
    try:
        stock = yf.Ticker(ticker.strip().upper())
        hist = stock.history(period='3mo')
        if hist.empty: return {'error': "No price history found."}
        
        # Calculate RV
        rv = yang_zhang(hist)
        
        # Calculate IV
        options = stock.option_chain(stock.options[0])
        iv = (options.calls['impliedVolatility'].mean() + options.puts['impliedVolatility'].mean()) / 2
        
        st.write(f"DEBUG: IV: {iv}, RV: {rv}")
        
        return {
            'avg_volume': hist['Volume'].mean(),
            'iv30_rv30': iv / rv if rv > 0 else 0,
            'ts_slope': -0.01, # Placeholder
            'expected_move': "5%"
        }
    except Exception as e:
        return {'error': str(e)}

# --- Web Interface ---

st.title("Earnings Debugger")
ticker = st.text_input("Stock Symbol:", "SPY").upper()

if st.button("Analyze"):
    result = compute_recommendation(ticker)
    if 'error' in result:
        st.error(result['error'])
    else:
        st.write("Results:", result)
