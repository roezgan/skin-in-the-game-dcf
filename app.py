import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# =====================================
# CONFIG & PATHS
# =====================================
# Root van het project (1 niveau boven deze file)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# .env inladen
load_dotenv()

# Relatief pad naar je Excel-bestand
LOCAL_FILE = os.getenv("LOCAL_FINANCIALS_FILE", str(DATA_DIR / "FMP_all_tickers_financials.xlsx"))

# Streamlit instellingen
st.set_page_config(page_title="DCF Valuation", layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
        .main {background-color:#fafafa;font-family:'Segoe UI',sans-serif;}
        h2 {
            margin-top:40px;
            background-color:#f0f0f0;
            padding:10px 15px;
            border-radius:6px;
            font-size:20px;
        }
        table {
            border-collapse:collapse;
            width:100%;
            border-radius:8px;
            overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,0.1);
        }
        th {
            background-color:#222;
            color:white!important;
            text-align:center!important;
            padding:6px;
        }
        td {
            padding:6px;
            text-align:center!important;
        }
        tr:hover td {background-color:#eaf6ff;}
        .sticky-header {
            position: fixed; top: 0; left: 0; right: 0; height: 35px;
            background-color: #ffffffcc; backdrop-filter: blur(6px);
            display: flex; justify-content: center; align-items: center; gap: 15px;
            border-bottom: 1px solid #ddd; z-index: 100;
        }
        .sticky-header a {
            color:#007bff; text-decoration:none; font-size:14px; font-weight:600;
        }
        .sticky-header a:hover {text-decoration:underline;}
        body {padding-top:40px;}
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
def load_local_financials(ticker):
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


def get_tbv(ticker):
    try:
        info = yf.Ticker(ticker).get_info()
        bv = info.get("bookValue") or info.get("bookValuePerShare")
        so = info.get("sharesOutstanding")
        if bv and so:
            return bv * 1000 * so / 1e9, "Yahoo"
    except Exception:
        return None, None
    return None, None


def get_balance(ticker):
    try:
        info = yf.Ticker(ticker).get_info()
        return (
            (info.get("totalDebt", 0) or 0) / 1e9,
            (info.get("totalCash", 0) or 0) / 1e9,
            (info.get("sharesOutstanding", 0) or 0),
        )
    except Exception:
        return 0, 0, 0


def marketcap(ticker):
    try:
        mc = yf.Ticker(ticker).get_info().get("marketCap")
        if not mc:
            return None
        return mc / 1e9 if mc > 1e9 else mc / 1e6
    except Exception:
        return None


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

cols = [c for c in ["2020", "2021", "2022", "2023", "2024", "TTM"] if c in df.columns]

# ---- Bereken Net Margin (%) en FCF Margin (%) ----
rev_row = df.loc[df["Metric"].str.lower() == "revenue", cols].astype(float)
ni_row  = df.loc[df["Metric"].str.lower() == "net income", cols].astype(float)
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
    **{c: [net_margin[c], fcf_margin[c]] for c in cols}
})

numeric_df = df.loc[df["Metric"].isin(["Revenue", "Net income", "FCF"]), ["Metric"] + cols].copy()
numeric_df = pd.concat([numeric_df, margin_rows], ignore_index=True)

st.dataframe(numeric_df, use_container_width=True)

# =====================================
# FORECAST
# =====================================
st.markdown("<a name='forecast'></a>", unsafe_allow_html=True)
st.header("🔮 Forecast")

model_type = st.selectbox("DCF Model Type", ["Free Cash Flow", "Net Income"], index=0)
yrs = st.slider("Forecast years", 3, 10, 5)
current_year = datetime.now().year
years = [str(current_year + i + 1) for i in range(yrs)]

# Basis-marges uit TTM
ttm_rev_val = rev_row["TTM"].iloc[0] if "TTM" in rev_row.columns and not rev_row.empty else np.nan
ttm_ni_val  = ni_row["TTM"].iloc[0] if "TTM" in ni_row.columns and not ni_row.empty else np.nan
ttm_fcf_val = fcf_row["TTM"].iloc[0] if "TTM" in fcf_row.columns and not fcf_row.empty else np.nan

ni_margin_ttm  = (ttm_ni_val / ttm_rev_val) if pd.notna(ttm_rev_val) and ttm_rev_val != 0 else np.nan
fcf_margin_ttm = (ttm_fcf_val / ttm_rev_val) if pd.notna(ttm_rev_val) and ttm_rev_val != 0 else np.nan

base_ni_margin  = ni_margin_ttm if pd.notna(ni_margin_ttm) else (ni / rev if (rev and ni) else 0)
base_fcf_margin = fcf_margin_ttm if pd.notna(fcf_margin_ttm) else (fcf / rev if (rev and fcf) else 0)
base_margin = base_fcf_margin if model_type == "Free Cash Flow" else base_ni_margin

# ---- Input mode ----
st.subheader("⚙️ Input mode")
input_mode = st.radio("Kies hoe je de inputs wil ingeven:", ["CAGR", "Year-by-Year"], horizontal=True)

if input_mode == "CAGR":
    st.caption("Voer één CAGR (%) in voor Revenue Growth en één vaste Margin (%).")
    rev_cagr = st.number_input("Revenue CAGR (%)", value=8.0, step=0.5)
    margin_const = st.number_input(
        f"{'FCF' if model_type=='Free Cash Flow' else 'Net'} Margin (%)",
        value=round(base_margin * 100, 1),
        step=0.5
    )
    growth_list, margin_list = [rev_cagr for _ in years], [margin_const for _ in years]
else:
    growth_list = [8.0 for _ in years]
    margin_list = [round(base_margin * 100, 1) for _ in years]

# ---- Input tabel ----
df_input = pd.DataFrame({"Year": years, "Revenue Growth (%)": growth_list, "Margin (%)": margin_list}).set_index("Year").T
st.subheader("📊 Input: Growth & Margin (%)")
edited_input = st.data_editor(df_input, key="forecast_input", hide_index=False, num_rows="fixed")

# ---- Berekeningen ----
edited_df = pd.DataFrame(edited_input).T.reset_index(names="Year")
edited_df.columns = ["Year", "Revenue Growth (%)", "Margin (%)"]

proj_rev, proj_metric = [], []
r = ttm_rev_val if pd.notna(ttm_rev_val) else (rev or 0.0)

for _, row in edited_df.iterrows():
    g = pd.to_numeric(row["Revenue Growth (%)"], errors="coerce")
    m = pd.to_numeric(row["Margin (%)"], errors="coerce")
    g = 0.0 if pd.isna(g) else float(g)
    m = 0.0 if pd.isna(m) else float(m) / 100.0
    r *= (1 + g / 100.0)
    proj_rev.append(r)
    proj_metric.append(r * m)

df_results = pd.DataFrame({
    "Year": years,
    "Projected Revenue ($M)": proj_rev,
    f"Projected {('FCF' if model_type=='Free Cash Flow' else 'Net Income')} ($M)": proj_metric
}).set_index("Year").T

styled_results = df_results.style.format("{:,.1f}").set_properties(**{"color": "#333"})
st.subheader(f"💵 Output: Projected Revenue & {'FCF' if model_type=='Free Cash Flow' else 'Net Income'}")
st.dataframe(styled_results, use_container_width=True)

proj_values = df_results.T.iloc[:, -1].tolist()

# =====================================
# DCF
# =====================================
st.markdown("<a name='dcf'></a>", unsafe_allow_html=True)
st.header("💵 DCF Calculation")

cA, cB, cC, cD = st.columns(4)
with cA:
    exit = st.selectbox("Terminal value", ["Perpetual Growth", "P/E Exit"], index=0)
with cB:
    discount_rate = st.number_input("Discount rate (%)", 3.0, 20.0, 9.0, step=0.25)
with cC:
    inc_tbv_choice = st.selectbox("Include Tangible Book Value", ["No", "Yes"], index=0)
    inc_tbv = inc_tbv_choice == "Yes"
with cD:
    if exit == "P/E Exit":
        pe = st.number_input("P/E multiple", 5.0, 50.0, 15.0, step=0.5)
        g = None
    else:
        g = st.number_input("Terminal growth (%)", 0.0, 5.0, 2.0, step=0.25)
        pe = None

pv = discount_values(proj_values, discount_rate)
tv = gordon_terminal(proj_values[-1], discount_rate, g) if exit == "Perpetual Growth" else proj_values[-1] * pe
pv_tv = tv / ((1 + discount_rate / 100) ** yrs)
ev = sum(pv) + pv_tv

tbv, tbv_src = (None, None)
if inc_tbv:
    tbv, tbv_src = get_tbv(ticker)
ev_tot = ev + (tbv or 0)
debt, cash, shares = get_balance(ticker)
eq = ev_tot - debt + cash

c1, c2, c3 = st.columns(3)
c1.metric("Σ PV CF ($M)", f"{sum(pv):,.0f}")
c2.metric("PV Terminal ($M)", f"{pv_tv:,.0f}")
c3.metric("Enterprise Value ($M)", f"{ev_tot:,.0f}")

if inc_tbv and tbv:
    st.caption(f"Tangible Book Value ({tbv_src}): {tbv:,.0f} $M")

# =====================================
# MARKET COMPARISON
# =====================================
st.markdown("<a name='market'></a>", unsafe_allow_html=True)
st.header("📊 Market Comparison")

mcap_b = marketcap(ticker)
if mcap_b:
    mcap_m = mcap_b * 1000
    st.metric("Market Cap ($M)", f"{mcap_m:,.0f}")
    upside = ((eq / mcap_m) - 1) * 100
    st.metric("Upside/Downside", f"{upside:+.2f}%")
