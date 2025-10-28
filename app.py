import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import requests
from datetime import datetime
from dotenv import load_dotenv

# =====================================
# CONFIG & PATHS
# =====================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

load_dotenv()
LOCAL_FILE = os.getenv("LOCAL_FINANCIALS_FILE", str(DATA_DIR / "FMP_all_tickers_financials.xlsx"))
FMP_API_KEY = os.getenv("FMP_API_KEY", "")

st.set_page_config(page_title="DCF Valuation", layout="wide")

# =====================================
# CUSTOM CSS
# =====================================
st.markdown(
    """
    <style>
        .main {background-color:#fafafa;font-family:'Segoe UI',sans-serif;}
        h2 {margin-top:40px;background-color:#f0f0f0;padding:10px 15px;border-radius:6px;font-size:20px;}
        table {border-collapse:collapse;width:100%;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.1);}
        th {background-color:#222;color:white!important;text-align:center!important;padding:6px;}
        td {padding:6px;text-align:center!important;}
        tr:hover td {background-color:#eaf6ff;}
        .sticky-header {position:fixed;top:0;left:0;right:0;height:35px;background-color:#ffffffcc;
            backdrop-filter:blur(6px);display:flex;justify-content:center;align-items:center;gap:15px;
            border-bottom:1px solid #ddd;z-index:100;}
        .sticky-header a {color:#007bff;text-decoration:none;font-size:14px;font-weight:600;}
        .sticky-header a:hover {text-decoration:underline;}
        body {padding-top:40px;}
        .upside-positive {color:green;font-size:28px;font-weight:700;}
        .upside-negative {color:#d32f2f;font-size:28px;font-weight:700;}
        .upside-neutral {color:#555;font-size:28px;font-weight:700;}
    </style>
    <div class="sticky-header">
        <a href="#historical">Historical</a>
        <a href="#forecast">Forecast</a>
        <a href="#dcf">DCF</a>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================
# HELPERS
# =====================================
@st.cache_data
def load_local_financials(ticker: str):
    if not os.path.exists(LOCAL_FILE):
        st.error(f"File not found: {LOCAL_FILE}")
        return None
    df = pd.read_excel(LOCAL_FILE)
    df = df[df["Ticker"].str.upper() == ticker.upper()].copy()
    for col in ["TTM", "2024", "2023", "2022", "2021", "2020"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df if not df.empty else None

def discount_values(values, r):
    r /= 100
    return [v / ((1 + r) ** (i + 1)) for i, v in enumerate(values)]

def gordon_terminal(last_cf, wacc, g):
    r, g = wacc / 100, g / 100
    return float("inf") if r <= g else last_cf * (1 + g) / (r - g)

@st.cache_data(ttl=3600)
def fmp_get_profile(ticker: str, apikey: str):
    if not apikey:
        return {}
    url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={apikey}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            row = data[0]
            return {
                "marketCap": row.get("mktCap"),
                "totalDebt": row.get("totalDebt"),
                "totalCash": row.get("cash"),
                "sharesOutstanding": row.get("sharesOutstanding"),
                "bookValue": row.get("bookValue"),
                "price": row.get("price"),
            }
    except Exception:
        return {}
    return {}

def marketcap_from_fmp(info: dict):
    mc = info.get("marketCap")
    return mc / 1e6 if mc else None

def get_balance_from_fmp(info: dict):
    try:
        debt = (info.get("totalDebt") or 0) / 1e6
        cash = (info.get("totalCash") or 0) / 1e6
        shares = info.get("sharesOutstanding") or 0
        return debt, cash, shares
    except Exception:
        return 0.0, 0.0, 0

def get_tbv_from_fmp(info: dict):
    bv = info.get("bookValue")
    so = info.get("sharesOutstanding")
    if bv and so:
        try:
            tbv_usd = float(bv) * float(so) / 1e6
            return tbv_usd, "FMP"
        except Exception:
            return None, None
    return None, None

# =====================================
# MAIN
# =====================================
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
st.title(f"💰 DCF Valuation – {ticker}")

df = load_local_financials(ticker)
if df is None:
    st.error("No data found.")
    st.stop()

def metric(df, m):
    try:
        return float(df.loc[df["Metric"].str.lower() == m.lower(), "TTM"].iloc[0])
    except Exception:
        return None

rev, ni, fcf = metric(df, "Revenue"), metric(df, "Net income"), metric(df, "FCF")

# =====================================
# HISTORICAL
# =====================================
st.markdown("<a name='historical'></a>", unsafe_allow_html=True)
st.header("📈 Historical")

cols = [c for c in ["2020", "2021", "2022", "2023", "2024", "TTM"] if c in df.columns]
rev_row = df.loc[df["Metric"].str.lower() == "revenue", cols].astype(float)
ni_row = df.loc[df["Metric"].str.lower() == "net income", cols].astype(float)
fcf_row = df.loc[df["Metric"].str.lower() == "fcf", cols].astype(float)

net_margin, fcf_margin = {}, {}
for c in cols:
    r = rev_row[c].iloc[0] if not rev_row.empty else np.nan
    n = ni_row[c].iloc[0] if not ni_row.empty else np.nan
    f = fcf_row[c].iloc[0] if not fcf_row.empty else np.nan
    net_margin[c] = round((n / r * 100), 1) if pd.notna(r) and r != 0 else np.nan
    fcf_margin[c] = round((f / r * 100), 1) if pd.notna(r) and r != 0 else np.nan

margin_rows = pd.DataFrame({
    "Metric": ["Net Margin (%)", "FCF Margin (%)"],
    **{c: [net_margin[c], fcf_margin[c]] for c in cols},
})
numeric_df = df.loc[df["Metric"].isin(["Revenue", "Net income", "FCF"]), ["Metric"] + cols].copy()
numeric_df = pd.concat([numeric_df, margin_rows], ignore_index=True)
st.dataframe(numeric_df, use_container_width=True)

# =====================================
# FORECAST (edit + inputs)
# =====================================
st.markdown("<a name='forecast'></a>", unsafe_allow_html=True)
st.header("🔮 Forecast")

colA, colB, colC, colD, colE, colF = st.columns(6)
with colA:
    operating_model = st.selectbox("Operating Model", ["Equity Model: via Net Income", "Equity Model: via FCF"], index=0)
    use_fcf = operating_model.endswith("FCF")

with colB:
    forecast_years = st.slider("Forecast Period", 3, 30, 5, step=1)

with colC:
    discount_rate = st.slider("Discount Rate (%)", 3.0, 40.0, 7.34, step=0.1)

with colD:
    exit_type = st.selectbox("Exit Type", ["Perpetual Growth Exit", "P/E Exit"], index=0)

with colE:
    if exit_type == "Perpetual Growth Exit":
        terminal_growth = st.slider("Terminal Growth (%)", -10.0, 10.0, 2.0, step=0.25)
        pe_multiple = None
    else:
        pe_multiple = st.number_input("P/E Exit Multiple", 5.0, 50.0, 15.0, step=0.5)
        terminal_growth = None

with colF:
    include_tbv = st.selectbox("Include Tangible Book Value", ["No", "Yes"], index=0) == "Yes"

# ---- Forecast data + edit logic ----
year_labels = [f"Year {i+1}" for i in range(forecast_years)] + ["Terminal"]
ttm_rev_val = float(rev or 0.0)
ttm_ni_val = float(ni or 0.0)
ttm_fcf_val = float(fcf or 0.0)

base_margin = (ttm_fcf_val / ttm_rev_val) if use_fcf else (ttm_ni_val / ttm_rev_val)
base_margin = base_margin if base_margin else 0.30

if "forecast_edit" not in st.session_state:
    st.session_state.forecast_edit = False

def _init_defaults():
    g = np.linspace(0.12, 0.05, forecast_years).round(4).tolist()
    m = np.linspace(base_margin, max(base_margin - 0.03, 0.05), forecast_years).round(4).tolist()
    st.session_state.baseline_growth = g[:]
    st.session_state.baseline_margin = m[:]
    st.session_state.buffer_growth = g[:]
    st.session_state.buffer_margin = m[:]

if "baseline_growth" not in st.session_state or len(st.session_state.baseline_growth) != forecast_years:
    _init_defaults()

# --- Toolbar (edit/save/reset) ---
toolbar = st.columns([0.12, 0.12, 0.12, 0.64])
with toolbar[0]:
    if not st.session_state.forecast_edit:
        if st.button("✏️ Edit", use_container_width=True):
            st.session_state.buffer_growth = st.session_state.baseline_growth[:]
            st.session_state.buffer_margin = st.session_state.baseline_margin[:]
            st.session_state.forecast_edit = True
    else:
        if st.button("💾 Save", use_container_width=True):
            st.session_state.baseline_growth = st.session_state.buffer_growth[:]
            st.session_state.baseline_margin = st.session_state.buffer_margin[:]
            st.session_state.forecast_edit = False
with toolbar[1]:
    if st.session_state.forecast_edit:
        if st.button("↩️ Cancel", use_container_width=True):
            st.session_state.forecast_edit = False
with toolbar[2]:
    if st.session_state.forecast_edit:
        if st.button("↺ Reset", use_container_width=True):
            _init_defaults()

# --- CAGR input tijdens edit ---
if st.session_state.forecast_edit:
    st.markdown("##### 🔢 Quick Adjust (apply to all years)")
    colx, coly = st.columns(2)
    with colx:
        cagr_rev = st.number_input("Revenue Growth CAGR (%)", value=round(np.mean(st.session_state.buffer_growth)*100,2))
    with coly:
        cagr_margin = st.number_input("Margin CAGR (%)", value=round(np.mean(st.session_state.buffer_margin)*100,2))
    if st.button("Apply CAGR to All Years"):
        st.session_state.buffer_growth = [cagr_rev/100.0]*forecast_years
        st.session_state.buffer_margin = [cagr_margin/100.0]*forecast_years

# --- Forecast berekeningen ---
growth_series = st.session_state.buffer_growth if st.session_state.forecast_edit else st.session_state.baseline_growth
margin_series = st.session_state.buffer_margin if st.session_state.forecast_edit else st.session_state.baseline_margin

if st.session_state.forecast_edit:
    edit_df = pd.DataFrame({
        "Metric": ["Revenue Growth (%)", "Margin (%)"],
        **{year_labels[i]: [growth_series[i]*100.0, margin_series[i]*100.0] for i in range(forecast_years)},
    }).set_index("Metric")
    edited = st.data_editor(edit_df, use_container_width=True, hide_index=False, disabled=["sort"])
    new_g = [float(pd.to_numeric(edited.iloc[0,i], errors="coerce") or 0.0)/100.0 for i in range(forecast_years)]
    new_m = [float(pd.to_numeric(edited.iloc[1,i], errors="coerce") or 0.0)/100.0 for i in range(forecast_years)]
    st.session_state.buffer_growth = new_g
    st.session_state.buffer_margin = new_m

revenues, net_incomes, fcfes = [], [], []
r = ttm_rev_val or 1.0
for i in range(forecast_years):
    r *= (1+growth_series[i])
    revenues.append(r)
    ni_est = r*margin_series[i]
    net_incomes.append(ni_est)
    fcfes.append(ni_est)

proj_values = fcfes if use_fcf else net_incomes
pv_stream = discount_values(proj_values, discount_rate)
terminal_val = gordon_terminal(proj_values[-1], discount_rate, terminal_growth) if exit_type=="Perpetual Growth Exit" else proj_values[-1]*pe_multiple
pv_terminal = terminal_val / ((1+discount_rate/100)**forecast_years)

def fmt_money(x): return f"{x:,.0f}"
def fmt_pct(x): return f"{x*100:,.2f}%"
table = pd.DataFrame({"": ["Revenue","Net Margin","Net Income","FCFE","Present Value"]})
for i in range(forecast_years):
    col = year_labels[i]
    table[col] = [
        fmt_money(revenues[i]), fmt_pct(margin_series[i]),
        fmt_money(net_incomes[i]), fmt_money(fcfes[i]),
        fmt_money(pv_stream[i]),
    ]
table["Terminal"] = [
    fmt_money(revenues[-1]*(1+growth_series[-1]*0.05)),
    fmt_pct(margin_series[-1]),
    fmt_money(proj_values[-1]), fmt_money(proj_values[-1]),
    fmt_money(pv_terminal),
]
st.dataframe(table.style.set_properties(**{"text-align":"center"}), use_container_width=True, hide_index=True)

# =====================================
# DCF CALCULATION (metrics only)
# =====================================
st.markdown("<a name='dcf'></a>", unsafe_allow_html=True)
st.header("💵 DCF Calculation")

enterprise_val = sum(pv_stream)+pv_terminal
fmp_profile = fmp_get_profile(ticker, FMP_API_KEY)
tbv, tbv_src = (None,None)
if include_tbv:
    tbv, tbv_src = get_tbv_from_fmp(fmp_profile)
enterprise_total = enterprise_val+(tbv or 0)
debt,cash,shares = get_balance_from_fmp(fmp_profile)
equity_val = enterprise_total-debt+cash
mcap_m = marketcap_from_fmp(fmp_profile)
upside = ((equity_val/mcap_m)-1)*100 if mcap_m else None

col1,col2,col3,col4 = st.columns(4)
col1.metric("Σ PV Cash Flows ($M)", f"{sum(pv_stream):,.0f}")
col2.metric("PV Terminal ($M)", f"{pv_terminal:,.0f}")
col3.metric("Enterprise Value ($M)", f"{enterprise_total:,.0f}")
if mcap_m:
    col4.metric("Market Cap ($M)", f"{mcap_m:,.0f}")

if upside is not None:
    color = "upside-positive" if upside>0 else ("upside-negative" if upside<0 else "upside-neutral")
    st.markdown(
        f"<div style='text-align:center;margin-top:20px;'>"
        f"<span class='{color}'>Upside/Downside vs Market Cap: {upside:+.2f}%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

if include_tbv and tbv:
    st.caption(f"Tangible Book Value included ({tbv_src}): {tbv:,.0f} $M")
