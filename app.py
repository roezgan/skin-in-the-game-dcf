import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
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

# --- Custom CSS & navigatie ---
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
        .muted {color:#666;font-size:12px;}
    </style>
    <div class="sticky-header">
        <a href="#historical">Historical</a>
        <a href="#forecast">Forecast</a>
        <a href="#dcf">DCF</a>
        <a href="#market">Market Comparison</a>
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

# =====================================
# FMP API HELPERS
# =====================================

@st.cache_data(ttl=3600)
def fmp_get_profile(ticker: str, apikey: str):
    """Haalt algemene profielinfo op (market cap, debt, cash, shares, book value, price)."""
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


@st.cache_data(ttl=3600)
def fmp_get_key_metrics_ttm(ticker: str, apikey: str):
    """Haalt TTM key metrics op via FMP (voor multiples)."""
    if not apikey:
        return {}
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={apikey}"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            row = data[0]
            return {
                "P/E": row.get("peRatioTTM"),
                "EV/EBITDA": row.get("evToEbitdaTTM"),
                "P/S": row.get("priceToSalesRatioTTM"),
                "P/B": row.get("pbRatioTTM"),
                "Debt/Equity": row.get("debtEquityRatioTTM"),
                "FCF Yield (%)": (
                    row.get("freeCashFlowPerShareTTM") / row.get("priceTTM") * 100
                    if row.get("freeCashFlowPerShareTTM") and row.get("priceTTM") else None
                ),
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
# SIDEBAR
# =====================================
ticker = st.sidebar.text_input("Ticker", "AAPL").upper()
st.sidebar.title("🔍 Navigation")
st.sidebar.markdown(
    "[Historical](#historical)  \n[Forecast](#forecast)  \n[DCF](#dcf)  \n[Market Comparison](#market)",
    unsafe_allow_html=True,
)

# =====================================
# MAIN TITLE
# =====================================
st.title(f"💰 DCF Valuation – {ticker}")

# =====================================
# LOAD DATA
# =====================================
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

cols = [c for c in ["2020","2021","2022","2023","2024","TTM"] if c in df.columns]
rev_row = df.loc[df["Metric"].str.lower()=="revenue", cols].astype(float)
ni_row  = df.loc[df["Metric"].str.lower()=="net income", cols].astype(float)
fcf_row = df.loc[df["Metric"].str.lower()=="fcf", cols].astype(float)

net_margin, fcf_margin = {}, {}
for c in cols:
    r = rev_row[c].iloc[0] if not rev_row.empty else np.nan
    n = ni_row[c].iloc[0] if not ni_row.empty else np.nan
    f = fcf_row[c].iloc[0] if not fcf_row.empty else np.nan
    net_margin[c] = round((n/r*100),1) if pd.notna(r) and r!=0 else np.nan
    fcf_margin[c] = round((f/r*100),1) if pd.notna(r) and r!=0 else np.nan

margin_rows = pd.DataFrame({"Metric":["Net Margin (%)","FCF Margin (%)"],
                            **{c:[net_margin[c],fcf_margin[c]] for c in cols}})
numeric_df = df.loc[df["Metric"].isin(["Revenue","Net income","FCF"]), ["Metric"]+cols].copy()
numeric_df = pd.concat([numeric_df, margin_rows], ignore_index=True)
st.dataframe(numeric_df, use_container_width=True)

# =====================================
# FORECAST
# =====================================
st.markdown("<a name='forecast'></a>", unsafe_allow_html=True)
st.header("🔮 Forecast")

model_type = st.selectbox("DCF Model Type", ["Free Cash Flow","Net Income"], index=0)
yrs = st.slider("Forecast years", 3, 10, 5)
current_year = datetime.now().year
years = [str(current_year+i+1) for i in range(yrs)]

ttm_rev_val = rev_row.get("TTM",pd.Series([np.nan])).iloc[0]
ttm_ni_val  = ni_row.get("TTM",pd.Series([np.nan])).iloc[0]
ttm_fcf_val = fcf_row.get("TTM",pd.Series([np.nan])).iloc[0]

ni_margin_ttm  = (ttm_ni_val/ttm_rev_val) if pd.notna(ttm_rev_val) and ttm_rev_val!=0 else np.nan
fcf_margin_ttm = (ttm_fcf_val/ttm_rev_val) if pd.notna(ttm_rev_val) and ttm_rev_val!=0 else np.nan
base_ni_margin  = ni_margin_ttm if pd.notna(ni_margin_ttm) else (ni/rev if (rev and ni) else 0)
base_fcf_margin = fcf_margin_ttm if pd.notna(fcf_margin_ttm) else (fcf/rev if (rev and fcf) else 0)
base_margin = base_fcf_margin if model_type=="Free Cash Flow" else base_ni_margin

st.subheader("⚙️ Input mode")
input_mode = st.radio("Kies hoe je de inputs wil ingeven:", ["CAGR","Year-by-Year"], horizontal=True)
if input_mode=="CAGR":
    rev_cagr = st.number_input("Revenue CAGR (%)", value=8.0, step=0.5)
    margin_const = st.number_input(f"{'FCF' if model_type=='Free Cash Flow' else 'Net'} Margin (%)",
                                   value=float(round((base_margin or 0)*100,1)), step=0.5)
    growth_list, margin_list = [rev_cagr]*len(years), [margin_const]*len(years)
else:
    growth_list = [8.0]*len(years)
    margin_list = [float(round((base_margin or 0)*100,1))]*len(years)

df_input = pd.DataFrame({"Year":years,"Revenue Growth (%)":growth_list,"Margin (%)":margin_list}).set_index("Year").T
st.subheader("📊 Input: Growth & Margin (%)")
edited_input = st.data_editor(df_input, key="forecast_input", hide_index=False, num_rows="fixed")

edited_df = pd.DataFrame(edited_input).T.reset_index(names="Year")
edited_df.columns = ["Year","Revenue Growth (%)","Margin (%)"]

proj_rev, proj_metric = [], []
r = ttm_rev_val if pd.notna(ttm_rev_val) else (rev or 0.0)
for _,row in edited_df.iterrows():
    g = float(pd.to_numeric(row["Revenue Growth (%)"], errors="coerce") or 0)
    m = float(pd.to_numeric(row["Margin (%)"], errors="coerce") or 0)/100
    r *= (1+g/100); proj_rev.append(r); proj_metric.append(r*m)

df_results = pd.DataFrame({"Year":years,
                           "Projected Revenue ($M)":proj_rev,
                           f"Projected {('FCF' if model_type=='Free Cash Flow' else 'Net Income')} ($M)":proj_metric}).set_index("Year").T
st.subheader(f"💵 Output: Projected Revenue & {('FCF' if model_type=='Free Cash Flow' else 'Net Income')}")
st.dataframe(df_results.style.format("{:,.1f}"), use_container_width=True)
proj_values = df_results.T.iloc[:,-1].tolist()

# =====================================
# DCF
# =====================================
st.markdown("<a name='dcf'></a>", unsafe_allow_html=True)
st.header("💵 DCF Calculation")

cA,cB,cC,cD = st.columns(4)
with cA: exit = st.selectbox("Terminal value", ["Perpetual Growth","P/E Exit"], index=0)
with cB: discount_rate = st.number_input("Discount rate (%)",3.0,20.0,9.0,step=0.25)
with cC: inc_tbv = st.selectbox("Include Tangible Book Value",["No","Yes"],index=0)=="Yes"
with cD:
    if exit=="P/E Exit": pe=st.number_input("P/E multiple",5.0,50.0,15.0,step=0.5); g=None
    else: g=st.number_input("Terminal growth (%)",0.0,5.0,2.0,step=0.25); pe=None

pv = discount_values(proj_values, discount_rate)
tv = gordon_terminal(proj_values[-1],discount_rate,g) if exit=="Perpetual Growth" else proj_values[-1]*pe
pv_tv = tv/((1+discount_rate/100)**yrs)
ev = sum(pv)+pv_tv

# ---- Balance & Market data via FMP ----
fmp_profile = fmp_get_profile(ticker, FMP_API_KEY)
tbv,tbv_src = (None,None)
if inc_tbv: tbv,tbv_src = get_tbv_from_fmp(fmp_profile)
ev_tot = ev+(tbv or 0)
debt,cash,shares = get_balance_from_fmp(fmp_profile)
eq = ev_tot - debt + cash

c1,c2,c3 = st.columns(3)
c1.metric("Σ PV CF ($M)",f"{sum(pv):,.0f}")
c2.metric("PV Terminal ($M)",f"{pv_tv:,.0f}")
c3.metric("Enterprise Value ($M)",f"{ev_tot:,.0f}")
if inc_tbv and tbv: st.caption(f"Tangible Book Value ({tbv_src}): {tbv:,.0f} $M")

# =====================================
# MARKET COMPARISON
# =====================================
st.markdown("<a name='market'></a>", unsafe_allow_html=True)
st.header("📊 Market Comparison")

mcap_m = marketcap_from_fmp(fmp_profile)
if mcap_m:
    st.metric("Market Cap ($M)", f"{mcap_m:,.0f}")
    try:
        upside = ((eq/mcap_m)-1)*100
        st.metric("Upside/Downside vs Market Cap", f"{upside:+.2f}%")
    except Exception:
        pass
else:
    st.caption("Market cap unavailable from FMP.")

if not FMP_API_KEY:
    st.info("Set FMP_API_KEY in .env or Streamlit Secrets to see valuation metrics.")
else:
    fmp_metrics = fmp_get_key_metrics_ttm(ticker, FMP_API_KEY)
    if fmp_metrics:
        keys=["P/E","EV/EBITDA","P/S","P/B","Debt/Equity","FCF Yield (%)"]
        cols = st.columns(len(keys))
        for i,k in enumerate(keys):
            v = fmp_metrics.get(k)
            val = "—" if v is None else (f"{v:,.2f}%" if "Yield" in k else f"{v:,.2f}")
            cols[i].metric(k, val)
        st.caption("Multiples via FMP (TTM) – cached 1 hour for stability.")
    else:
        st.caption("No valuation metrics available from FMP.")

st.markdown("<span class='muted'>Prices & balance via FMP (cached); valuation metrics via FMP (cached).</span>",
            unsafe_allow_html=True)
