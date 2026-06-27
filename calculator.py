"""
DISCLAIMER: 
This software is provided solely for educational and research purposes. 
"""
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import FreeSimpleGUI as sg
import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import threading

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
    log_oc_sq = log_oc**2
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    k = 0.34 / (1.34 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)
    return result.iloc[-1] if return_last_only else result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    sort_idx = days.argsort()
    days, ivs = days[sort_idx], ivs[sort_idx]
    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")
    def term_spline(dte):
        if dte < days[0]: return ivs[0]
        elif dte > days[-1]: return ivs[-1]
        else: return float(spline(dte))
    return term_spline

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
        ts_slope = (term_spline(45) - term_spline(dtes[0])) / (45-dtes[0])
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        
        return {
            'avg_volume': price_history['Volume'].rolling(30).mean().iloc[-1] >= 1500000,
            'iv30_rv30': iv30_rv30 >= 1.25,
            'ts_slope_0_45': ts_slope <= -0.00406,
            'expected_move': f"{round(straddle / underlying_price * 100, 2)}%" if straddle else "N/A"
        }
    except Exception as e:
        return {'error': str(e)}

def main_gui():
    window = sg.Window("Earnings Position Checker", [
        [sg.Text("Enter Stock Symbol:"), sg.Input(key="stock", size=(20, 1), focus=True)],
        [sg.Button("Submit", bind_return_key=True), sg.Button("Exit")]
    ])
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"): break
        if event == "Submit":
            result = compute_recommendation(values["stock"])
            if 'error' in result:
                sg.popup_error(f"Error: {result['error']}")
            else:
                msg = (
                    f"Analysis for {values['stock']}:\n\n"
                    f"Avg Volume Pass: {result['avg_volume']}\n"
                    f"IV30/RV30 Pass: {result['iv30_rv30']}\n"
                    f"Trend Slope Pass: {result['ts_slope_0_45']}\n"
                    f"Expected Move: {result['expected_move']}"
                )
                sg.popup_scrolled(msg, title="Results")

    window.close()

if __name__ == "__main__":
    main_gui()