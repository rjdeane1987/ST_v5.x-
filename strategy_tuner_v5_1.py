import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from datetime import datetime

# strategy_tuner_v5_1.py

st.set_page_config(layout="wide")
st.title("📈 Strategy Tuner v5.1 – Zero Excuses Edition")

# --- Sidebar Inputs ---
ticker = st.sidebar.text_input("Ticker", "SPY")
start_date = st.sidebar.date_input("Start Date", datetime(2015, 1, 1))
end_date   = st.sidebar.date_input("End Date", datetime.today())
window     = st.sidebar.slider("HMM/Rolling Window (bars)", 100, 504, 252)
vol_target = st.sidebar.slider("Volatility Target (%)",      1, 50, 15)
drawdown_limit = st.sidebar.slider("Max Drawdown (%)",      1, 50, 15)
slippage_pct   = st.sidebar.slider("Slippage (%)",        0.0, 1.0, 0.1)
commission_pct = st.sidebar.slider("Commission (%)",      0.0, 1.0, 0.05)

# --- Data Loading & Returns ---
@st.cache_data
def load_data(tkr, start, end):
    df = yf.download(tkr, start=start, end=end, auto_adjust=True)
    if df.empty or "Adj Close" not in df:
        return pd.DataFrame()
    df = df[["Adj Close"]].rename(columns={"Adj Close": "Price"})
    df["LogRet"] = np.log(df["Price"] / df["Price"].shift(1))
    return df.dropna()

df = load_data(ticker, start_date, end_date)
if df.empty or len(df) < window+2:
    st.warning("Insufficient data to run strategy.")
    st.stop()

# --- Helper Functions ---
def hurst(ts):
    lags = np.arange(2, 20)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    m = np.polyfit(np.log(lags), np.log(tau), 1)
    return m[0] * 2.0

def spectral_slope(ts):
    freqs, psd = plt.mlab.psd(ts, NFFT=min(256, len(ts)))
    # skip zero freq
    idx = freqs > 0
    m = np.polyfit(np.log(freqs[idx]), np.log(psd[idx]), 1)
    return m[0]

# --- Rolling Indicators ---
df["Hurst"]   = df["LogRet"].rolling(window).apply(hurst, raw=False)
df["Spectral"]= df["LogRet"].rolling(window).apply(spectral_slope, raw=False)

# --- Forward-walking HMM Regime Detection ---
regimes = []
for i in range(window, len(df)):
    X = df["LogRet"].iloc[i-window:i].values.reshape(-1,1)
    try:
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
        model.fit(X)
        regimes.append(model.predict(df["LogRet"].iloc[i].reshape(-1,1))[0])
    except:
        regimes.append(0)  # fallback to neutral
df = df.iloc[window:].copy()
df["Regime"] = regimes

# --- Simulation Loop ---
equity = []
cap = 1.0
peak = 1.0
for i in range(len(df)):
    row = df.iloc[i]
    vol = row["LogRet"].rolling(21).std() if i>=21 else df["LogRet"].iloc[:i+1].std()
    reg = row["Regime"]
    # signal: only go long in the regime with highest mean return
    signal = 1.0 if reg==df.groupby("Regime")["LogRet"].mean().idxmax() else 0.0
    # size = volatility-targeted
    size = min(1.0, (vol_target/100)/(vol*np.sqrt(252))) if vol>0 else 0.0
    pos = signal * size
    # PnL
    ret = row["LogRet"] * pos
    cap *= np.exp(ret)
    # friction
    cap *= (1 - (slippage_pct+commission_pct)/100)
    # drawdown kill
    peak = max(peak, cap)
    if (peak - cap)/peak > drawdown_limit/100:
        st.warning("🔴 Drawdown exceeded — halting")
        break
    equity.append(cap)

df["Equity"] = equity + [equity[-1]]*(len(df)-len(equity))

# --- Output ---
st.subheader("📈 Equity Curve vs. Buy & Hold")
fig, ax = plt.subplots(figsize=(10,4))
df["Equity"].plot(ax=ax, label="Strategy")
(df["Price"]/df["Price"].iloc[0]).plot(ax=ax, alpha=0.4, label="Buy & Hold")
ax.legend()
st.pyplot(fig)

st.subheader("📊 Performance Metrics")
total_ret = df["Equity"].iloc[-1] - 1
sharpe    = df["Equity"].pct_change().mean()/df["Equity"].pct_change().std()*np.sqrt(252)
max_dd    = ((peak - df["Equity"]) / peak).max()
col1,col2,col3 = st.columns(3)
col1.metric("Total Return", f"{total_ret:.2%}")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_dd:.2%}")

# --- Done ---
