import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d

# --- Core Logic ---

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = (log_cc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol = (log_oc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))
    
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    return result.iloc[-1] if return_last_only else result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days, ivs = days[sort_idx], ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    return lambda dte: float(spline(dte))

def get_current_price(ticker):
    return ticker.history(period='1d')['Close'].iloc[0]

def compute_recommendation(ticker):
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        if not stock.options: raise ValueError("No options found.")
        exp_dates = filter_dates(list(stock.options))
        underlying_price = get_current_price(stock)
        
        atm_iv, straddle = {}, None
        for i, exp_date in enumerate(exp_dates):
            chain = stock.option_chain(exp_date)
            if chain.calls.empty or chain.puts.empty: continue
            
            call_idx = (chain.calls['strike'] - underlying_price).abs().idxmin()
            put_idx = (chain.puts['strike'] - underlying_price).abs().idxmin()
            
            atm_iv[exp_date] = (chain.calls.loc[call_idx, 'impliedVolatility'] + chain.puts.loc[put_idx, 'impliedVolatility']) / 2.0
            
            if i == 0:
                straddle = (chain.calls.loc[call_idx, 'bid'] + chain.calls.loc[call_idx, 'ask'] + 
                           chain.puts.loc[put_idx, 'bid'] + chain.puts.loc[put_idx, 'ask']) / 2.0

        today = datetime.today().date()
        dtes = [(datetime.strptime(d, "%Y-%m-%d").date() - today).days for d in atm_iv.keys()]
        term_spline = build_term_structure(dtes, list(atm_iv.values()))
        
        price_history = stock.history(period='3mo')
        ts_slope = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        avg_vol = price_history['Volume'].rolling(30).mean().iloc[-1]
        
        return {
            'avg_volume': avg_vol,
            'avg_volume_pass': avg_vol >= 1500000,
            'iv30_rv30': iv30_rv30,
            'iv30_rv30_pass': iv30_rv30 >= 1.25,
            'ts_slope': ts_slope,
            'ts_slope_pass': ts_slope <= -0.00406,
            'expected_move': f"{round(straddle / underlying_price * 100, 2)}%" if straddle else "N/A"
        }
    except Exception as e:
        return {'error': str(e)}

# --- Web Interface ---

st.title("Earnings Position Checker")
ticker = st.text_input("Enter Stock Symbol:", "AAPL").upper()

if st.button("Submit"):
    with st.spinner('Analyzing...'):
        result = compute_recommendation(ticker)
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader(f"Analysis for {ticker}")
            
            # Use columns for a clean dashboard look
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Avg Volume", f"{int(result['avg_volume']):,}", 
                        "PASS" if result['avg_volume_pass'] else "FAIL")
            col2.metric("IV30/RV30", f"{result['iv30_rv30']:.2f}", 
                        "PASS" if result['iv30_rv30_pass'] else "FAIL")
            col3.metric("TS Slope", f"{result['ts_slope']:.4f}", 
                        "PASS" if result['ts_slope_pass'] else "FAIL")
            
            st.write(f"### Expected Move: {result['expected_move']}")