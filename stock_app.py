"""
Stock Technical Indicator App
Shows: Price (K-line), MA5/10/20, RSI5/10, Williams %R (威廉), DMI (+DI/-DI/ADX), BIAS

Run:
    pip install streamlit yfinance pandas plotly
    streamlit run stock_app.py
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------- indicators
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing (used by RSI / DMI)."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    diff = close.diff()
    gain = wilder(diff.clip(lower=0), n)
    loss = wilder(-diff.clip(upper=0), n)
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def williams_r(high, low, close, n: int) -> pd.Series:
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return (hh - close) / (hh - ll) * -100  # range 0 ~ -100


def dmi(high, low, close, n: int = 14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = wilder(tr, n)
    plus_di = 100 * wilder(plus_dm, n) / atr
    minus_di = 100 * wilder(minus_dm, n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = wilder(dx, n)
    return plus_di, minus_di, adx


def bias(close: pd.Series, n: int) -> pd.Series:
    ma = sma(close, n)
    return (close - ma) / ma * 100


# ---------------------------------------------------------------- data
@st.cache_data(ttl=600)
def load(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns multi-index
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Stock Indicators", layout="wide")
st.title("📈 Stock Technical Indicators")

PRESETS = {
    "2330.TW 台積電": "2330.TW",
    "2317.TW 鴻海": "2317.TW",
    "2454.TW 聯發科": "2454.TW",
    "0050.TW 元大台灣50": "0050.TW",
    "2603.TW 長榮": "2603.TW",
    "AAPL Apple": "AAPL",
    "NVDA Nvidia": "NVDA",
}

with st.sidebar:
    st.header("Stock")
    choice = st.selectbox("Pick a stock", list(PRESETS.keys()))
    custom = st.text_input("…or type a ticker (上市 .TW, 上櫃 .TWO)", "")
    ticker = custom.strip().upper() or PRESETS[choice]
    period = st.selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=1)

    st.header("Parameters")
    wr_n = st.number_input("Williams %R period", 2, 100, 14)
    dmi_n = st.number_input("DMI period", 2, 100, 14)
    bias_ns = st.multiselect("BIAS periods", [5, 6, 10, 12, 20, 24, 60], default=[5, 10, 20])

df = load(ticker, period)
if df.empty:
    st.error(f"No data for **{ticker}**. Check the ticker (e.g. 2330.TW, 6488.TWO, AAPL).")
    st.stop()

c, h, l = df["Close"], df["High"], df["Low"]
for n in (5, 10, 20):
    df[f"MA{n}"] = sma(c, n)
df["RSI5"], df["RSI10"] = rsi(c, 5), rsi(c, 10)
df["WR"] = williams_r(h, l, c, wr_n)
df["+DI"], df["-DI"], df["ADX"] = dmi(h, l, c, dmi_n)
for n in bias_ns:
    df[f"BIAS{n}"] = bias(c, n)

# ---- latest values
last, prev = df.iloc[-1], df.iloc[-2]
chg = last["Close"] - prev["Close"]
cols = st.columns(6)
cols[0].metric(ticker, f"{last['Close']:.2f}", f"{chg:+.2f} ({chg / prev['Close'] * 100:+.2f}%)")
cols[1].metric("MA5 / 10 / 20", f"{last['MA5']:.1f} / {last['MA10']:.1f} / {last['MA20']:.1f}")
cols[2].metric("RSI5 / RSI10", f"{last['RSI5']:.1f} / {last['RSI10']:.1f}")
cols[3].metric(f"威廉 %R({wr_n})", f"{last['WR']:.1f}")
cols[4].metric("+DI / -DI / ADX", f"{last['+DI']:.1f} / {last['-DI']:.1f} / {last['ADX']:.1f}")
if bias_ns:
    cols[5].metric("BIAS", " / ".join(f"{last[f'BIAS{n}']:.2f}" for n in bias_ns))

# ---- charts
UP, DOWN = "#e53935", "#2e7d32"  # Taiwan convention: red up, green down
fig = make_subplots(
    rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.025,
    row_heights=[0.36, 0.1, 0.135, 0.135, 0.135, 0.135],
    subplot_titles=("Price & MA", "Volume", "RSI", f"Williams %R ({wr_n})", f"DMI ({dmi_n})", "BIAS"),
)
fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=h, low=l, close=c, name="Price",
                             increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
for n, col in zip((5, 10, 20), ("#ff9800", "#2196f3", "#9c27b0")):
    fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{n}"], name=f"MA{n}", line=dict(width=1.3, color=col)), 1, 1)

vol_colors = [UP if cl >= op else DOWN for op, cl in zip(df["Open"], c)]
fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=vol_colors, name="Volume", showlegend=False), 2, 1)

fig.add_trace(go.Scatter(x=df.index, y=df["RSI5"], name="RSI5", line=dict(color="#ff9800")), 3, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["RSI10"], name="RSI10", line=dict(color="#2196f3")), 3, 1)
for y in (80, 20):
    fig.add_hline(y=y, line_dash="dot", line_color="gray", row=3, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["WR"], name="%R", line=dict(color="#795548")), 4, 1)
for y in (-20, -80):
    fig.add_hline(y=y, line_dash="dot", line_color="gray", row=4, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df["+DI"], name="+DI", line=dict(color=UP)), 5, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["-DI"], name="-DI", line=dict(color=DOWN)), 5, 1)
fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(color="#607d8b", dash="dash")), 5, 1)

for n, col in zip(bias_ns, ("#ff9800", "#2196f3", "#9c27b0", "#009688", "#e91e63", "#3f51b5", "#8bc34a")):
    fig.add_trace(go.Scatter(x=df.index, y=df[f"BIAS{n}"], name=f"BIAS{n}", line=dict(color=col)), 6, 1)
fig.add_hline(y=0, line_color="gray", row=6, col=1)

fig.update_layout(height=1300, xaxis_rangeslider_visible=False, hovermode="x unified",
                  margin=dict(l=40, r=20, t=40, b=20), legend=dict(orientation="h", y=1.02))
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])  # hide weekends
st.plotly_chart(fig, use_container_width=True)

with st.expander("Raw data"):
    st.dataframe(df.iloc[::-1].round(2), use_container_width=True)
