from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants & IMF Country Directory (ISO3 + Euro Area G163)
# ---------------------------------------------------------------------------
IMF_SDMX_BASE = "https://api.imf.org/external/sdmx/3.0"
START_PERIOD_DEFAULT = "1976-M01"
REQUEST_TIMEOUT = 30

IMF_COUNTRIES: Dict[str, str] = {
    "G163": "Euro Area (G163)",
    "AFG": "Afghanistan (AFG)",
    "ALB": "Albania (ALB)",
    "DZA": "Algeria (DZA)",
    "AND": "Andorra (AND)",
    "AGO": "Angola (AGO)",
    "ATG": "Antigua and Barbuda (ATG)",
    "ARG": "Argentina (ARG)",
    "ARM": "Armenia (ARM)",
    "ABW": "Aruba (ABW)",
    "AUS": "Australia (AUS)",
    "AUT": "Austria (AUT)",
    "AZE": "Azerbaijan (AZE)",
    "BHS": "Bahamas, The (BHS)",
    "BHR": "Bahrain (BHR)",
    "BGD": "Bangladesh (BGD)",
    "BRB": "Barbados (BRB)",
    "BLR": "Belarus (BLR)",
    "BEL": "Belgium (BEL)",
    "BLZ": "Belize (BLZ)",
    "BEN": "Benin (BEN)",
    "BTN": "Bhutan (BTN)",
    "BOL": "Bolivia (BOL)",
    "BIH": "Bosnia and Herzegovina (BIH)",
    "BWA": "Botswana (BWA)",
    "BRA": "Brazil (BRA)",
    "BRN": "Brunei Darussalam (BRN)",
    "BGR": "Bulgaria (BGR)",
    "BFA": "Burkina Faso (BFA)",
    "BDI": "Burundi (BDI)",
    "CPV": "Cabo Verde (CPV)",
    "KHM": "Cambodia (KHM)",
    "CMR": "Cameroon (CMR)",
    "CAN": "Canada (CAN)",
    "CAF": "Central African Republic (CAF)",
    "TCD": "Chad (TCD)",
    "CHL": "Chile (CHL)",
    "CHN": "China, People's Republic of (CHN)",
    "COL": "Colombia (COL)",
    "COM": "Comoros (COM)",
    "COD": "Congo, Democratic Republic of the (COD)",
    "COG": "Congo, Republic of (COG)",
    "CRI": "Costa Rica (CRI)",
    "CIV": "Cote d'Ivoire (CIV)",
    "HRV": "Croatia (HRV)",
    "CYP": "Cyprus (CYP)",
    "CZE": "Czechia (CZE)",
    "DNK": "Denmark (DNK)",
    "DJI": "Djibouti (DJI)",
    "DMA": "Dominica (DMA)",
    "DOM": "Dominican Republic (DOM)",
    "ECU": "Ecuador (ECU)",
    "EGY": "Egypt (EGY)",
    "SLV": "El Salvador (SLV)",
    "GNQ": "Equatorial Guinea (GNQ)",
    "ERI": "Eritrea (ERI)",
    "EST": "Estonia (EST)",
    "SWZ": "Eswatini (SWZ)",
    "ETH": "Ethiopia (ETH)",
    "FJI": "Fiji (FJI)",
    "FIN": "Finland (FIN)",
    "FRA": "France (FRA)",
    "GAB": "Gabon (GAB)",
    "GMB": "Gambia, The (GMB)",
    "GEO": "Georgia (GEO)",
    "DEU": "Germany (DEU)",
    "GHA": "Ghana (GHA)",
    "GRC": "Greece (GRC)",
    "GRD": "Grenada (GRD)",
    "GTM": "Guatemala (GTM)",
    "GIN": "Guinea (GIN)",
    "GNB": "Guinea-Bissau (GNB)",
    "GUY": "Guyana (GUY)",
    "HTI": "Haiti (HTI)",
    "HND": "Honduras (HND)",
    "HKG": "Hong Kong SAR (HKG)",
    "HUN": "Hungary (HUN)",
    "ISL": "Iceland (ISL)",
    "IND": "India (IND)",
    "IDN": "Indonesia (IDN)",
    "IRN": "Iran (IRN)",
    "IRQ": "Iraq (IRQ)",
    "IRL": "Ireland (IRL)",
    "ISR": "Israel (ISR)",
    "ITA": "Italy (ITA)",
    "JAM": "Jamaica (JAM)",
    "JPN": "Japan (JPN)",
    "JOR": "Jordan (JOR)",
    "KAZ": "Kazakhstan (KAZ)",
    "KEN": "Kenya (KEN)",
    "KIR": "Kiribati (KIR)",
    "KOR": "Korea, Republic of (KOR)",
    "KOS": "Kosovo (KOS)",
    "KWT": "Kuwait (KWT)",
    "KGZ": "Kyrgyz Republic (KGZ)",
    "LAO": "Lao P.D.R. (LAO)",
    "LVA": "Latvia (LVA)",
    "LBN": "Lebanon (LBN)",
    "LSO": "Lesotho (LSO)",
    "LBR": "Liberia (LBR)",
    "LBY": "Libya (LBY)",
    "LTU": "Lithuania (LTU)",
    "LUX": "Luxembourg (LUX)",
    "MAC": "Macao SAR (MAC)",
    "MDG": "Madagascar (MDG)",
    "MWI": "Malawi (MWI)",
    "MYS": "Malaysia (MYS)",
    "MDV": "Maldives (MDV)",
    "MLI": "Mali (MLI)",
    "MLT": "Malta (MLT)",
    "MHL": "Marshall Islands (MHL)",
    "MRT": "Mauritania (MRT)",
    "MUS": "Mauritius (MUS)",
    "MEX": "Mexico (MEX)",
    "FSM": "Micronesia, Federated States of (FSM)",
    "MDA": "Moldova (MDA)",
    "MNG": "Mongolia (MNG)",
    "MNE": "Montenegro (MNE)",
    "MAR": "Morocco (MAR)",
    "MOZ": "Mozambique (MOZ)",
    "MMR": "Myanmar (MMR)",
    "NAM": "Namibia (NAM)",
    "NRU": "Nauru (NRU)",
    "NPL": "Nepal (NPL)",
    "NLD": "Netherlands (NLD)",
    "NZL": "New Zealand (NZL)",
    "NIC": "Nicaragua (NIC)",
    "NER": "Niger (NER)",
    "NGA": "Nigeria (NGA)",
    "MKD": "North Macedonia (MKD)",
    "NOR": "Norway (NOR)",
    "OMN": "Oman (OMN)",
    "PAK": "Pakistan (PAK)",
    "PLW": "Palau (PLW)",
    "PAN": "Panama (PAN)",
    "PNG": "Papua New Guinea (PNG)",
    "PRY": "Paraguay (PRY)",
    "PER": "Peru (PER)",
    "PHL": "Philippines (PHL)",
    "POL": "Poland (POL)",
    "PRT": "Portugal (PRT)",
    "QAT": "Qatar (QAT)",
    "ROU": "Romania (ROU)",
    "RUS": "Russian Federation (RUS)",
    "RWA": "Rwanda (RWA)",
    "WSM": "Samoa (WSM)",
    "SMR": "San Marino (SMR)",
    "STP": "Sao Tome and Principe (STP)",
    "SAU": "Saudi Arabia (SAU)",
    "SEN": "Senegal (SEN)",
    "SRB": "Serbia (SRB)",
    "SYC": "Seychelles (SYC)",
    "SLE": "Sierra Leone (SLE)",
    "SGP": "Singapore (SGP)",
    "SVK": "Slovak Republic (SVK)",
    "SVN": "Slovenia (SVN)",
    "SLB": "Solomon Islands (SLB)",
    "SOM": "Somalia (SOM)",
    "ZAF": "South Africa (ZAF)",
    "SSD": "South Sudan (SSD)",
    "ESP": "Spain (ESP)",
    "LKA": "Sri Lanka (LKA)",
    "KNA": "St. Kitts and Nevis (KNA)",
    "LCA": "St. Lucia (LCA)",
    "VCT": "St. Vincent and the Grenadines (VCT)",
    "SDN": "Sudan (SDN)",
    "SUR": "Suriname (SUR)",
    "SWE": "Sweden (SWE)",
    "CHE": "Switzerland (CHE)",
    "SYR": "Syrian Arab Republic (SYR)",
    "TJK": "Tajikistan (TJK)",
    "TZA": "Tanzania (TZA)",
    "THA": "Thailand (THA)",
    "TLS": "Timor-Leste (TLS)",
    "TGO": "Togo (TGO)",
    "TON": "Tonga (TON)",
    "TTO": "Trinidad and Tobago (TTO)",
    "TUN": "Tunisia (TUN)",
    "TUR": "Turkiye (TUR)",
    "TKM": "Turkmenistan (TKM)",
    "TUV": "Tuvalu (TUV)",
    "UGA": "Uganda (UGA)",
    "UKR": "Ukraine (UKR)",
    "ARE": "United Arab Emirates (ARE)",
    "GBR": "United Kingdom (GBR)",
    "USA": "United States (USA)",
    "URY": "Uruguay (URY)",
    "UZB": "Uzbekistan (UZB)",
    "VUT": "Vanuatu (VUT)",
    "VEN": "Venezuela (VEN)",
    "VNM": "Vietnam (VNM)",
    "YEM": "Yemen (YEM)",
    "ZMB": "Zambia (ZMB)",
    "ZWE": "Zimbabwe (ZWE)",
}


# ---------------------------------------------------------------------------
# IMF SDMX 3.0 Data Retrieval Functions
# ---------------------------------------------------------------------------
def _parse_time_period(series: pd.Series) -> pd.Series:
    # Convert IMF monthly strings (e.g. '1976-M01' or '1976-01') to Timestamp
    cleaned = series.astype(str).str.strip().str.replace("-M", "-", regex=False)
    return pd.to_datetime(cleaned, format="%Y-%m", errors="coerce")


def _fetch_sdmx_csv(url: str) -> Optional[pd.Series]:
    # Attempt to retrieve IMF SDMX 3.0 series as CSV
    headers = {
        "Accept": "text/csv",
        "User-Agent": "IMF-RER-Calculator/1.0 (Python/3.12)",
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200 or not resp.text.strip():
        return None

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
        return None

    df["date"] = _parse_time_period(df["TIME_PERIOD"])
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    if df.empty:
        return None
    return df.set_index("date")["value"]


def _fetch_sdmx_json(url: str) -> Optional[pd.Series]:
    # Fallback parser for SDMX-JSON 2.0 responses from IMF SDMX 3.0 API
    headers = {
        "Accept": "application/vnd.sdmx.data+json;version=2.0.0",
        "User-Agent": "IMF-RER-Calculator/1.0 (Python/3.12)",
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        return None

    payload = resp.json()
    data_block = payload.get("data", payload)
    datasets = data_block.get("dataSets", [])
    structures = data_block.get("structures", [])
    if not datasets or not structures:
        return None

    obs_dims = structures[0].get("dimensions", {}).get("observation", [])
    time_values = []
    for dim in obs_dims:
        if dim.get("id") in ("TIME_PERIOD", "TIME"):
            time_values = [v.get("id") for v in dim.get("values", [])]
            break

    if not time_values:
        return None

    series_dict = datasets[0].get("series", {})
    if not series_dict:
        return None

    records = []
    for _series_key, series_obj in series_dict.items():
        observations = series_obj.get("observations", {})
        for obs_idx_str, obs_val_list in observations.items():
            idx = int(obs_idx_str)
            if idx < len(time_values) and obs_val_list:
                raw_val = obs_val_list[0]
                if raw_val is not None:
                    records.append((time_values[idx], float(raw_val)))

    if not records:
        return None

    df = pd.DataFrame(records, columns=["TIME_PERIOD", "OBS_VALUE"])
    df["date"] = _parse_time_period(df["TIME_PERIOD"])
    df["value"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    if df.empty:
        return None
    return df.set_index("date")["value"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_imf_series(dataflow: str, key: str, start_period: str = START_PERIOD_DEFAULT) -> pd.Series:
    # Fetch a monthly series from the IMF SDMX 3.0 API starting from January 1976
    candidate_urls = [
        f"{IMF_SDMX_BASE}/data/dataflow/IMF.STA/{dataflow}/~/{key}?c[TIME_PERIOD]=ge:{start_period}",
        f"{IMF_SDMX_BASE}/data/dataflow/IMF.STA/{dataflow}/+/{key}?startPeriod={start_period.replace('-M', '-')}",
    ]

    for url in candidate_urls:
        try:
            series = _fetch_sdmx_csv(url)
            if series is not None and not series.empty:
                return series[series.index >= pd.Timestamp("1976-01-01")]
        except Exception:
            pass

        try:
            series = _fetch_sdmx_json(url)
            if series is not None and not series.empty:
                return series[series.index >= pd.Timestamp("1976-01-01")]
        except Exception:
            pass

    return pd.Series(dtype=float)


@st.cache_data(ttl=3600, show_spinner=False)
def load_country_pair_data(iso1: str, iso2: str) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    # Retrieve monthly CPI and XDC_USD exchange rate series for Country 1 and Country 2
    cpi_key_1 = f"{iso1}.CPI._T.IX.M"
    cpi_key_2 = f"{iso2}.CPI._T.IX.M"
    er_key_1 = f"{iso1}.XDC_USD.PA_RT.M"
    er_key_2 = f"{iso2}.XDC_USD.PA_RT.M"

    p1 = fetch_imf_series("CPI", cpi_key_1)
    p2 = fetch_imf_series("CPI", cpi_key_2)
    e1 = fetch_imf_series("ER", er_key_1)
    e2 = fetch_imf_series("ER", er_key_2)

    non_empty_indices = [s.index for s in (p1, p2, e1, e2) if not s.empty]
    if not non_empty_indices:
        return pd.DataFrame(), {
            "p1_ok": False,
            "p2_ok": False,
            "e1_ok": False,
            "e2_ok": False,
        }

    min_date = max(pd.Timestamp("1976-01-01"), min(idx.min() for idx in non_empty_indices))
    max_date = max(idx.max() for idx in non_empty_indices)
    full_index = pd.date_range(start=min_date, end=max_date, freq="MS")

    df = pd.DataFrame(index=full_index)
    df.index.name = "Date"
    df["P1_CPI"] = p1.reindex(full_index)
    df["P2_CPI"] = p2.reindex(full_index)
    df["E1_XDC_USD"] = e1.reindex(full_index)
    df["E2_XDC_USD"] = e2.reindex(full_index)

    if iso1 == "USA":
        df["E1_XDC_USD"] = df["E1_XDC_USD"].fillna(1.0)
    if iso2 == "USA":
        df["E2_XDC_USD"] = df["E2_XDC_USD"].fillna(1.0)

    df.loc[df["E1_XDC_USD"] <= 0, "E1_XDC_USD"] = np.nan
    df.loc[df["E2_XDC_USD"] <= 0, "E2_XDC_USD"] = np.nan
    df.loc[df["P1_CPI"] <= 0, "P1_CPI"] = np.nan
    df.loc[df["P2_CPI"] <= 0, "P2_CPI"] = np.nan

    # S(t): units of Country 2 currency per 1 unit of Country 1 currency
    df["S_Nominal"] = df["E2_XDC_USD"] / df["E1_XDC_USD"]

    # RER(t) = S(t) * P1(t) / P2(t)
    df["RER_Raw"] = df["S_Nominal"] * (df["P1_CPI"] / df["P2_CPI"])

    diagnostics = {
        "p1_ok": bool(df["P1_CPI"].notna().any()),
        "p2_ok": bool(df["P2_CPI"].notna().any()),
        "e1_ok": bool(df["E1_XDC_USD"].notna().any()),
        "e2_ok": bool(df["E2_XDC_USD"].notna().any()),
    }
    return df, diagnostics


def compute_sample_metrics(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> Tuple[pd.DataFrame, Optional[float]]:
    # Slice dataset to [start_date, end_date] and compute RER* and Real_Index(t)
    sample = df.loc[(df.index >= start_date) & (df.index <= end_date)].copy()
    valid_rer = sample["RER_Raw"].dropna()

    if valid_rer.empty:
        sample["RER_Star"] = np.nan
        sample["Real_Index"] = np.nan
        return sample, None

    rer_star = float(valid_rer.mean())
    sample["RER_Star"] = rer_star
    sample["Real_Index"] = 100.0 * (sample["RER_Raw"] / rer_star)
    return sample, rer_star


def build_interactive_chart(
    sample_df: pd.DataFrame,
    iso1: str,
    iso2: str,
    series_mode: str,
    rer_star: Optional[float],
) -> go.Figure:
    # Create an interactive Plotly chart for Real, Nominal, or Both series
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    real_label = "Real Exchange Rate Index (Sample Mean = 100)"
    nominal_label = f"Nominal S(t): {iso2} Currency per 1 {iso1} Currency"

    if series_mode in ("Real", "Both"):
        fig.add_trace(
            go.Scatter(
                x=sample_df.index,
                y=sample_df["Real_Index"],
                mode="lines",
                name=real_label,
                line=dict(color="#1f4e78", width=2.4),
                connectgaps=False,
                hovertemplate="<b>%{x|%Y-%m}</b><br>Real Index(t): %{y:.2f}<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="#6c757d",
            annotation_text="Sample Mean (100.0)",
            annotation_position="bottom right",
            secondary_y=False,
        )

    if series_mode in ("Nominal", "Both"):
        use_secondary = series_mode == "Both"
        fig.add_trace(
            go.Scatter(
                x=sample_df.index,
                y=sample_df["S_Nominal"],
                mode="lines",
                name=nominal_label,
                line=dict(
                    color="#c0392b",
                    width=2.2,
                    dash="solid" if not use_secondary else "dot",
                ),
                connectgaps=False,
                hovertemplate=(
                    f"<b>%{{x|%Y-%m}}</b><br>Nominal S(t): %{{y:.4f}} ({iso2}/{iso1})<extra></extra>"
                ),
            ),
            secondary_y=use_secondary,
        )

    chart_title = (
        f"Bilateral Exchange Rate Analysis: {IMF_COUNTRIES.get(iso1, iso1)} vs. "
        f"{IMF_COUNTRIES.get(iso2, iso2)}"
    )
    subtitle = (
        f"Sample Period: {sample_df.index.min().strftime('%Y-%m')} to "
        f"{sample_df.index.max().strftime('%Y-%m')}"
        + (f" | RER* = {rer_star:.4f}" if rer_star is not None else "")
    )

    fig.update_layout(
        title=dict(text=f"{chart_title}<br><sup>{subtitle}</sup>", x=0.0),
        template="plotly_white",
        hovermode="x unified",
        height=540,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=60, r=60, t=90, b=50),
    )

    fig.update_xaxes(title_text="Month", showgrid=True)

    if series_mode == "Real":
        fig.update_yaxes(
            title_text="Real Index(t) (100 x RER(t) / RER*)",
            secondary_y=False,
            showgrid=True,
        )
        fig.update_yaxes(showticklabels=False, showgrid=False, secondary_y=True)
    elif series_mode == "Nominal":
        fig.update_yaxes(
            title_text=f"Nominal S(t) ({iso2} units per 1 {iso1} unit)",
            secondary_y=False,
            showgrid=True,
        )
        fig.update_yaxes(showticklabels=False, showgrid=False, secondary_y=True)
    else:
        fig.update_yaxes(
            title_text="Real Index(t) (100 x RER(t) / RER*)",
            secondary_y=False,
            showgrid=True,
            title_font=dict(color="#1f4e78"),
            tickfont=dict(color="#1f4e78"),
        )
        fig.update_yaxes(
            title_text=f"Nominal S(t) ({iso2} units per 1 {iso1} unit)",
            secondary_y=True,
            showgrid=False,
            title_font=dict(color="#c0392b"),
            tickfont=dict(color="#c0392b"),
        )

    return fig


# ---------------------------------------------------------------------------
# Streamlit User Interface
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="IMF Real Exchange Rate Calculator",
        layout="wide",
    )

    st.title("Bilateral Real and Nominal Exchange Rate Calculator")
    st.markdown(
        "This application retrieves monthly Consumer Price Index (`{ISO3}.CPI._T.IX.M`) and "
        "US dollar exchange rate (`{ISO3}.XDC_USD.PA_RT.M`) observations from the "
        "**IMF SDMX 3.0 API** (`https://api.imf.org/external/sdmx/3.0`) beginning in January 1976. "
        "Nominal exchange rates **S(t)** are expressed as units of **Country 2** currency per one unit "
        "of **Country 1** currency. The real exchange rate is **RER(t) = S(t) x P1(t) / P2(t)**, "
        "and the indexed series is normalized to the sample-period average **RER***: "
        "**Real Index(t) = 100 x RER(t) / RER***."
    )

    country_codes = list(IMF_COUNTRIES.keys())
    default_c1_idx = country_codes.index("USA") if "USA" in country_codes else 0
    default_c2_idx = country_codes.index("G163") if "G163" in country_codes else 1

    with st.sidebar:
        st.header("Model Parameters")
        iso1 = st.selectbox(
            "Country 1 (Base Currency, P1)",
            options=country_codes,
            index=default_c1_idx,
            format_func=lambda code: IMF_COUNTRIES[code],
        )
        iso2 = st.selectbox(
            "Country 2 (Quote Currency, P2)",
            options=country_codes,
            index=default_c2_idx,
            format_func=lambda code: IMF_COUNTRIES[code],
        )

        series_mode = st.radio(
            "Chart Display Mode",
            options=["Both", "Real", "Nominal"],
            index=0,
            help="Select 'Both' to plot Real Index(t) on the left axis and unnormalized Nominal S(t) on the right axis.",
        )

    with st.spinner("Querying IMF SDMX 3.0 API for monthly CPI and exchange rate series..."):
        df, diagnostics = load_country_pair_data(iso1, iso2)

    if df.empty:
        st.error(
            "No monthly CPI or exchange rate observations were returned by the IMF API "
            "for the selected country pair."
        )
        return

    missing_components = []
    if not diagnostics["p1_ok"]:
        missing_components.append(f"Country 1 CPI ({iso1}.CPI._T.IX.M)")
    if not diagnostics["p2_ok"]:
        missing_components.append(f"Country 2 CPI ({iso2}.CPI._T.IX.M)")
    if not diagnostics["e1_ok"]:
        missing_components.append(f"Country 1 Exchange Rate ({iso1}.XDC_USD.PA_RT.M)")
    if not diagnostics["e2_ok"]:
        missing_components.append(f"Country 2 Exchange Rate ({iso2}.XDC_USD.PA_RT.M)")

    if missing_components:
        st.warning(
            "Limited IMF data coverage detected. The following series returned no valid monthly "
            "observations: " + ", ".join(missing_components)
        )

    valid_any = df.dropna(subset=["S_Nominal", "RER_Raw"], how="all")
    if valid_any.empty:
        min_available = df.index.min().to_pydatetime()
        max_available = df.index.max().to_pydatetime()
    else:
        min_available = valid_any.index.min().to_pydatetime()
        max_available = valid_any.index.max().to_pydatetime()

    all_months = df.index.to_pydatetime().tolist()

    with st.sidebar:
        st.subheader("Sample Window")
        selected_range = st.select_slider(
            "Select Start and End Months",
            options=all_months,
            value=(min_available, max_available),
            format_func=lambda dt: dt.strftime("%Y-%m"),
        )

    start_ts = pd.Timestamp(selected_range[0])
    end_ts = pd.Timestamp(selected_range[1])

    sample_df, rer_star = compute_sample_metrics(df, start_ts, end_ts)

    col1, col2, col3, col4 = st.columns(4)
    valid_rer_count = int(sample_df["RER_Raw"].notna().sum())
    valid_nom_count = int(sample_df["S_Nominal"].notna().sum())
    total_months = len(sample_df)

    col1.metric("Selected Sample Period", f"{start_ts.strftime('%Y-%m')} to {end_ts.strftime('%Y-%m')}")
    col2.metric(
        "Sample Mean RER*",
        f"{rer_star:.4f}" if rer_star is not None else "N/A",
    )
    latest_real = sample_df["Real_Index"].dropna()
    col3.metric(
        "Latest Real Index(t)",
        f"{latest_real.iloc[-1]:.2f}" if not latest_real.empty else "N/A",
    )
    latest_nom = sample_df["S_Nominal"].dropna()
    col4.metric(
        f"Latest Nominal S(t) ({iso2}/{iso1})",
        f"{latest_nom.iloc[-1]:.4f}" if not latest_nom.empty else "N/A",
    )

    if valid_rer_count < total_months or valid_nom_count < total_months:
        st.info(
            f"Data coverage note: Within the {total_months} selected months, "
            f"{valid_nom_count} months contain valid nominal exchange rates S(t) and "
            f"{valid_rer_count} months contain complete observations for RER(t)."
        )

    if sample_df["Real_Index"].notna().any() or sample_df["S_Nominal"].notna().any():
        fig = build_interactive_chart(sample_df, iso1, iso2, series_mode, rer_star)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No valid Real or Nominal exchange rate observations exist within the selected date range.")

    export_df = sample_df.copy()
    export_df.index = export_df.index.strftime("%Y-%m")
    export_df = export_df.reset_index().rename(
        columns={
            "Date": "Month",
            "P1_CPI": f"P1_CPI_{iso1}",
            "P2_CPI": f"P2_CPI_{iso2}",
            "E1_XDC_USD": f"E1_XDC_USD_{iso1}",
            "E2_XDC_USD": f"E2_XDC_USD_{iso2}",
            "S_Nominal": f"S_Nominal_{iso2}_per_{iso1}",
            "RER_Raw": "RER_t",
            "RER_Star": "RER_Star_Sample_Mean",
            "Real_Index": "Real_Index_t",
        }
    )

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    csv_filename = f"imf_rer_{iso1}_{iso2}_{start_ts.strftime('%Y%m')}_{end_ts.strftime('%Y%m')}.csv"

    st.download_button(
        label="Download Sample Observations and Calculations (CSV)",
        data=csv_bytes,
        file_name=csv_filename,
        mime="text/csv",
    )

    with st.expander("Inspect Sample Period Data Table"):
        st.dataframe(export_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
