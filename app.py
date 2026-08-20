import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# HOUSE VALUE · ML VALUATION STUDIO
# House Price Prediction Using Machine Learning
# ============================================================


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="House Value - ML Valuation Studio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. COLOR PALETTE
# ============================================================

BG = "#090B0F"
SIDEBAR_BG = "#0D0F14"
CARD_BG = "#12151B"
PANEL_BG = "#11141A"

TEXT_PRIMARY = "#F4F4F5"
TEXT_SECONDARY = "#A1A1AA"
TEXT_MUTED = "#7E828C"
TEXT_FAINT = "#5E626C"

PURPLE = "#A78BFA"
PURPLE_DARK = "#8B5CF6"
PURPLE_LIGHT = "#C4B5FD"

GREEN = "#4ADE80"

BORDER = "rgba(255,255,255,0.06)"
PURPLE_BORDER = "rgba(167,139,250,0.35)"
GRID = "rgba(255,255,255,0.06)"


# ============================================================
# 3. GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000'
        '&family=Space+Grotesk:wght@300..700&display=swap'
    );

    html, body, [class*="css"], .stApp {{
        font-family: 'DM Sans', sans-serif !important;
        background-color: {BG} !important;
        color: {TEXT_PRIMARY} !important;
    }}

    .stApp {{
        background-color: {BG} !important;
    }}

    /* -----------------------------
       SIDEBAR
       ----------------------------- */

    section[data-testid="stSidebar"] {{
        background-color: {SIDEBAR_BG} !important;
        border-right: 1px solid {BORDER} !important;
    }}

    section[data-testid="stSidebar"] > div {{
        padding-top: 1.5rem;
    }}

    .sidebar-brand {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: {TEXT_PRIMARY};
        margin-bottom: 2px;
    }}

    .sidebar-subtitle {{
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: {PURPLE};
        margin-bottom: 25px;
    }}

    .sidebar-section {{
        color: {TEXT_FAINT};
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}

    .sidebar-model-card {{
        background-color: {CARD_BG};
        border: 1px solid {PURPLE_BORDER};
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 25px;
    }}

    .sidebar-model-label {{
        color: {TEXT_FAINT};
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .sidebar-model-name {{
        color: {TEXT_PRIMARY};
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.92rem;
        font-weight: 600;
    }}

    .sidebar-model-metrics {{
        color: {GREEN};
        font-size: 0.76rem;
        margin-top: 5px;
    }}

    /* -----------------------------
       HEADINGS
       ----------------------------- */

    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: {TEXT_PRIMARY} !important;
    }}

    .page-kicker {{
        color: {PURPLE};
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}

    .page-title {{
        font-family: 'Space Grotesk', sans-serif;
        color: {TEXT_PRIMARY};
        font-size: 2.45rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.1;
        margin-bottom: 10px;
    }}

    .page-description {{
        color: {TEXT_MUTED};
        font-size: 0.95rem;
        max-width: 820px;
        line-height: 1.55;
        margin-bottom: 30px;
    }}

    /* -----------------------------
       CARDS
       ----------------------------- */

    .metric-card {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }}

    .metric-card-primary {{
        background-color: {PANEL_BG};
        border: 1px solid {PURPLE};
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 16px;
    }}

    .metric-card-highlight {{
        background-color: {PANEL_BG};
        border: 1px solid {PURPLE_BORDER};
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
    }}

    .metric-label {{
        color: {TEXT_MUTED};
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }}

    .metric-value {{
        font-family: 'Space Grotesk', sans-serif;
        color: {TEXT_PRIMARY};
        font-size: 1.8rem;
        font-weight: 700;
    }}

    .metric-value-large {{
        font-family: 'Space Grotesk', sans-serif;
        color: {GREEN};
        font-size: 2.9rem;
        font-weight: 700;
        line-height: 1.05;
    }}

    .metric-sub {{
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        margin-top: 5px;
    }}

    .section-title {{
        font-family: 'Space Grotesk', sans-serif;
        color: {TEXT_PRIMARY};
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 5px;
    }}

    .section-description {{
        color: {TEXT_MUTED};
        font-size: 0.78rem;
        margin-bottom: 15px;
    }}

    /* -----------------------------
       BUTTONS
       ----------------------------- */

    .stButton > button {{
        width: 100% !important;
        background-color: {PURPLE_DARK} !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 7px !important;
        padding: 0.72rem 1.5rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        background-color: {PURPLE} !important;
        color: {BG} !important;
        box-shadow: 0 4px 18px rgba(167,139,250,0.18) !important;
    }}

    /* -----------------------------
       INPUTS
       ----------------------------- */

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background-color: {CARD_BG} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT_PRIMARY} !important;
        border-radius: 7px !important;
    }}

    div[data-baseweb="select"] * {{
        color: {TEXT_PRIMARY} !important;
    }}

    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {{
        color: {TEXT_PRIMARY} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}

    /* -----------------------------
       TABLE
       ----------------------------- */

    .stDataFrame {{
        border: 1px solid {BORDER};
    }}

    /* -----------------------------
       DIVIDER
       ----------------------------- */

    hr {{
        border-color: {BORDER} !important;
        margin: 30px 0 !important;
    }}

    /* -----------------------------
       FOOTER
       ----------------------------- */

    .footer {{
        border-top: 1px solid {BORDER};
        margin-top: 45px;
        padding-top: 20px;
        padding-bottom: 20px;
        text-align: center;
        color: {TEXT_FAINT};
        font-size: 0.68rem;
        letter-spacing: 0.08em;
    }}

    /* -----------------------------
       STREAMLIT CLEANUP
       ----------------------------- */

    #MainMenu {{
        visibility: hidden;
    }}

    footer {{
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. NEIGHBORHOOD MAP
# ============================================================

NEIGHBORHOOD_MAP = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "Greens": "Greens",
    "GrnHill": "Green Hills",
    "IDOTRR": "Iowa DOT and Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NoRidge": "Northridge",
    "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights",
    "NWAmes": "Northwest Ames",
    "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker",
}

REVERSE_NEIGHBORHOOD_MAP = {
    value: key for key, value in NEIGHBORHOOD_MAP.items()
}


# ============================================================
# 5. DATA + MODEL LOADING
# ============================================================

@st.cache_data
def load_dataset():
    path = "cleaned_AmesHousing.csv"

    if not os.path.exists(path):
        return None, f"Dataset not found: {path}"

    try:
        data = pd.read_csv(path)
        data.columns = [str(c).strip() for c in data.columns]
        return data, None
    except Exception as exc:
        return None, f"Could not read dataset: {exc}"


@st.cache_resource
def load_model():
    path = "house_price_model.pkl"

    if not os.path.exists(path):
        return None, f"Model not found: {path}"

    try:
        loaded_model = joblib.load(path)
        return loaded_model, None
    except Exception as exc:
        return None, f"Could not load model: {exc}"


df, data_error = load_dataset()
model, model_error = load_model()


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

if data_error:
    st.error(data_error)
    st.info("Keep cleaned_AmesHousing.csv in the same folder as app.py.")
    st.stop()

if model_error:
    st.error(model_error)
    st.info("Keep house_price_model.pkl in the same folder as app.py.")
    st.stop()

if df is None or df.empty:
    st.error("The dataset is empty.")
    st.stop()

if "SalePrice" not in df.columns:
    st.error("The cleaned dataset does not contain the required 'SalePrice' column.")
    st.stop()


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def style_figure(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor=BG,
        plot_bgcolor=PANEL_BG,
        font=dict(
            family="DM Sans",
            color=TEXT_PRIMARY,
        ),
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(
            font=dict(
                family="DM Sans",
                color=TEXT_SECONDARY,
            )
        ),
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(family="DM Sans", color=TEXT_PRIMARY),
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont=dict(color=TEXT_MUTED),
        title_font=dict(family="DM Sans", color=TEXT_PRIMARY),
    )

    return fig


def add_engineered_features(data):
    result = data.copy()

    required_total_sf = ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]
    if all(col in result.columns for col in required_total_sf):
        result["TotalSF"] = (
            pd.to_numeric(result["Total Bsmt SF"], errors="coerce").fillna(0)
            + pd.to_numeric(result["1st Flr SF"], errors="coerce").fillna(0)
            + pd.to_numeric(result["2nd Flr SF"], errors="coerce").fillna(0)
        )

    required_bath = ["Full Bath", "Half Bath", "Bsmt Full Bath", "Bsmt Half Bath"]
    if all(col in result.columns for col in required_bath):
        result["TotalBathrooms"] = (
            pd.to_numeric(result["Full Bath"], errors="coerce").fillna(0)
            + 0.5 * pd.to_numeric(result["Half Bath"], errors="coerce").fillna(0)
            + pd.to_numeric(result["Bsmt Full Bath"], errors="coerce").fillna(0)
            + 0.5 * pd.to_numeric(result["Bsmt Half Bath"], errors="coerce").fillna(0)
        )

    porch_columns = ["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Wood Deck SF"]
    available_porch_columns = [col for col in porch_columns if col in result.columns]

    if available_porch_columns:
        porch_sum = pd.Series(0.0, index=result.index)
        for col in available_porch_columns:
            porch_sum += pd.to_numeric(result[col], errors="coerce").fillna(0)
        result["TotalPorchSF"] = porch_sum

    return result


def model_expected_columns(model_object):
    if hasattr(model_object, "feature_names_in_"):
        return list(model_object.feature_names_in_)
    return None


def build_prediction_input(
    source_df,
    model_object,
    overall_qual,
    gr_liv_area,
    lot_area,
    year_built,
    total_bsmt_sf,
    bedrooms,
    full_bath,
    half_bath,
    garage_cars,
    selected_neighborhood_code,
):
    work = add_engineered_features(source_df)

    if len(work) == 0:
        raise ValueError("The dataset contains no rows to use as a prediction template.")

    input_data = work.drop(columns=["SalePrice"], errors="ignore").iloc[[0]].copy()

    numeric_overrides = {
        "Overall Qual": overall_qual,
        "Gr Liv Area": gr_liv_area,
        "Lot Area": lot_area,
        "Year Built": year_built,
        "Total Bsmt SF": total_bsmt_sf,
        "Bedroom AbvGr": bedrooms,
        "Full Bath": full_bath,
        "Half Bath": half_bath,
        "Garage Cars": garage_cars,
    }

    for column, value in numeric_overrides.items():
        if column in input_data.columns:
            input_data.at[input_data.index[0], column] = value

    if "Neighborhood" in input_data.columns and selected_neighborhood_code is not None:
        input_data.at[input_data.index[0], "Neighborhood"] = selected_neighborhood_code

    if "TotalSF" in input_data.columns:
        first_floor = (
            pd.to_numeric(input_data["1st Flr SF"], errors="coerce").fillna(0).iloc[0]
            if "1st Flr SF" in input_data.columns else 0
        )
        second_floor = (
            pd.to_numeric(input_data["2nd Flr SF"], errors="coerce").fillna(0).iloc[0]
            if "2nd Flr SF" in input_data.columns else 0
        )
        input_data.at[input_data.index[0], "TotalSF"] = total_bsmt_sf + first_floor + second_floor

    if "TotalBathrooms" in input_data.columns:
        bsmt_full = (
            pd.to_numeric(input_data["Bsmt Full Bath"], errors="coerce").fillna(0).iloc[0]
            if "Bsmt Full Bath" in input_data.columns else 0
        )
        bsmt_half = (
            pd.to_numeric(input_data["Bsmt Half Bath"], errors="coerce").fillna(0).iloc[0]
            if "Bsmt Half Bath" in input_data.columns else 0
        )
        input_data.at[input_data.index[0], "TotalBathrooms"] = (
            full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half
        )

    if "TotalPorchSF" in input_data.columns:
        porch_total = 0.0
        for column in ["Open Porch SF", "Enclosed Porch", "3Ssn Porch", "Screen Porch", "Wood Deck SF"]:
            if column in input_data.columns:
                porch_total += float(
                    pd.to_numeric(input_data[column], errors="coerce").fillna(0).iloc[0]
                )
        input_data.at[input_data.index[0], "TotalPorchSF"] = porch_total

    expected = model_expected_columns(model_object)
    if expected is not None:
        missing = [col for col in expected if col not in input_data.columns]
        if missing:
            raise ValueError(
                "The saved model expects feature(s) that are not available in cleaned_AmesHousing.csv: "
                + ", ".join(missing)
            )
        input_data = input_data[expected]

    return input_data


def safe_numeric_series(series):
    return pd.to_numeric(series, errors="coerce")


def find_neighborhood_column(data):
    for column in ["Neighborhood", "neighborhood"]:
        if column in data.columns:
            return column
    return None


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="sidebar-brand">HOUSE VALUE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">ML Valuation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    navigation = st.radio(
        "Navigation",
        [
            "01  PROPERTY VALUATOR",
            "02  MARKET LENS",
            "03  MODEL BENCHMARK",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """
        <div class="sidebar-model-card">
            <div class="sidebar-model-label">Current Model</div>
            <div class="sidebar-model-name">
                Random Forest Regression
            </div>
            <div class="sidebar-model-metrics">
                R² 0.9231 · RMSE $24.8K
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# 9. PAGE 1 — PROPERTY VALUATOR
# ============================================================

if navigation == "01  PROPERTY VALUATOR":

    st.markdown('<div class="page-kicker">01 / PROPERTY VALUATOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Estimate market value.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="page-description">
        Enter the key characteristics of a property and use the
        trained Random Forest model to generate a market-value estimate.
        </div>
        """,
        unsafe_allow_html=True,
    )

    val_col1, val_col2 = st.columns([7, 5], gap="large")

    with val_col1:
        # REMOVED: st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Property inputs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-description">Configure the characteristics used for valuation.</div>',
            unsafe_allow_html=True,
        )

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            overall_qual = st.slider("Overall Quality (1–10)", min_value=1, max_value=10, value=6)
            gr_liv_area = st.number_input("Living Area (sq ft)", min_value=300, max_value=10000, value=1500, step=50)
            lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=100000, value=9600, step=100)
            year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=1980, step=1)
            total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=5000, value=1000, step=50)

        with input_col2:
            bedrooms = st.number_input("Bedrooms (AbvGrd)", min_value=0, max_value=15, value=3, step=1)
            full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=8, value=2, step=1)
            half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=5, value=0, step=1)
            garage_cars = st.number_input("Garage Capacity (Cars)", min_value=0, max_value=8, value=2, step=1)

            neighborhood_col = find_neighborhood_column(df)

            if neighborhood_col:
                actual_neighborhoods = sorted(
                    df[neighborhood_col].dropna().astype(str).unique().tolist()
                )
                readable_options = sorted([
                    NEIGHBORHOOD_MAP.get(code, code) for code in actual_neighborhoods
                ])
                selected_neighborhood_display = st.selectbox("Neighborhood", readable_options)
                selected_neighborhood_code = REVERSE_NEIGHBORHOOD_MAP.get(
                    selected_neighborhood_display, selected_neighborhood_display
                )
            else:
                selected_neighborhood_display = "Not available"
                selected_neighborhood_code = None
                st.warning("Neighborhood is not available in the dataset.")

        predict_btn = st.button("ESTIMATE MARKET VALUE")
        # REMOVED: st.markdown("</div>", unsafe_allow_html=True)

    predicted_price = None
    prediction_error = None

    if predict_btn:
        if model is None:
            prediction_error = "The trained model could not be loaded."
        else:
            try:
                input_data = build_prediction_input(
                    source_df=df,
                    model_object=model,
                    overall_qual=overall_qual,
                    gr_liv_area=gr_liv_area,
                    lot_area=lot_area,
                    year_built=year_built,
                    total_bsmt_sf=total_bsmt_sf,
                    bedrooms=bedrooms,
                    full_bath=full_bath,
                    half_bath=half_bath,
                    garage_cars=garage_cars,
                    selected_neighborhood_code=selected_neighborhood_code,
                )
                predicted_price = float(model.predict(input_data)[0])
            except Exception as exc:
                prediction_error = str(exc)

    with val_col2:
        if prediction_error:
            st.markdown(
                """
                <div class="metric-card-primary">
                    <div class="metric-label">PREDICTION ERROR</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.error("The trained model could not process this property.")
            with st.expander("Technical error details"):
                st.code(prediction_error)
            st.caption("No fallback or fabricated price was generated.")

        elif predicted_price is not None:
            st.markdown('<div class="metric-card-primary">', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">ESTIMATED MARKET VALUE</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value-large">${predicted_price:,.0f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-sub" style="color:#A78BFA;">Random Forest Regression · R² 0.9231</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<hr>", unsafe_allow_html=True)

            price_per_sqft = (
                predicted_price / gr_liv_area if gr_liv_area > 0 else 0
            )
            median_price = float(safe_numeric_series(df["SalePrice"]).median())
            difference = predicted_price - median_price
            percentage = (difference / median_price * 100) if median_price != 0 else 0

            p1, p2 = st.columns(2)

            with p1:
                st.markdown('<div class="metric-label">PRICE / SQ FT</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">${price_per_sqft:,.0f}</div>', unsafe_allow_html=True)

            with p2:
                st.markdown('<div class="metric-label">VS DATASET MEDIAN</div>', unsafe_allow_html=True)
                sign = "+" if difference >= 0 else ""
                st.markdown(
                    f'<div class="metric-value" style="font-size:1.35rem;">{sign}{percentage:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="metric-label">PROPERTY SUMMARY</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="metric-sub">
                • <b>Quality:</b> {overall_qual}/10<br>
                • <b>Size:</b> {gr_liv_area:,} sq ft living area | {total_bsmt_sf:,} sq ft bsmt<br>
                • <b>Layout:</b> {bedrooms} beds, {full_bath} full / {half_bath} half baths<br>
                • <b>Built:</b> {year_built} | <b>Garage:</b> {garage_cars} cars<br>
                • <b>Location:</b> {selected_neighborhood_display}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.markdown(
                """
                <div class="metric-card-highlight">
                    <div class="metric-label">VALUATION READY</div>
                    <div class="metric-value" style="font-size: 1.2rem; margin-top: 10px;">
                        Configure parameters & click estimate
                    </div>
                    <div class="metric-sub">
                        Adjust property attributes on the left to generate real-time AI valuation estimates.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ============================================================
# 10. PAGE 2 — MARKET LENS
# ============================================================

elif navigation == "02  MARKET LENS":

    st.markdown('<div class="page-kicker">02 / MARKET LENS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Explore market trends.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="page-description">
        Analyze price distribution patterns, feature relationships, and neighborhood trends across the Ames dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    m_col1, m_col2 = st.columns(2, gap="large")

    with m_col1:
        st.markdown('<div class="section-title">Sale Price Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-description">Frequency distribution of historical property prices.</div>', unsafe_allow_html=True)
        
        fig_hist = px.histogram(
            df,
            x="SalePrice",
            nbins=40,
            color_discrete_sequence=[PURPLE_DARK],
            labels={"SalePrice": "Sale Price ($)"},
        )
        fig_hist = style_figure(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)

    with m_col2:
        st.markdown('<div class="section-title">Living Area vs. Price</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-description">Ground living area compared against price, colored by overall quality.</div>', unsafe_allow_html=True)
        
        if "Gr Liv Area" in df.columns and "Overall Qual" in df.columns:
            fig_scatter = px.scatter(
                df,
                x="Gr Liv Area",
                y="SalePrice",
                color="Overall Qual",
                color_continuous_scale="Purples",
                labels={"Gr Liv Area": "Living Area (sq ft)", "SalePrice": "Sale Price ($)", "Overall Qual": "Quality"},
            )
            fig_scatter = style_figure(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Required columns ('Gr Liv Area', 'Overall Qual') not found in dataset.")

    st.markdown("---")

    # Neighborhood analysis
    st.markdown('<div class="section-title">Top Neighborhoods by Median Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-description">Comparison of median property values across top locations.</div>', unsafe_allow_html=True)

    neighborhood_col = find_neighborhood_column(df)
    if neighborhood_col:
        neigh_summary = (
            df.groupby(neighborhood_col)["SalePrice"]
            .median()
            .reset_index()
            .sort_values(by="SalePrice", ascending=False)
            .head(12)
        )
        neigh_summary["Display_Name"] = neigh_summary[neighborhood_col].map(
            lambda c: NEIGHBORHOOD_MAP.get(str(c), str(c))
        )

        fig_bar = px.bar(
            neigh_summary,
            x="SalePrice",
            y="Display_Name",
            orientation="h",
            color="SalePrice",
            color_continuous_scale="Purples",
            labels={"SalePrice": "Median Sale Price ($)", "Display_Name": "Neighborhood"},
        )
        fig_bar.update_layout(yaxis={"categoryorder": "total ascending"})
        fig_bar = style_figure(fig_bar, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Neighborhood data is not available.")


# ============================================================
# 11. PAGE 3 — MODEL BENCHMARK
# ============================================================

elif navigation == "03  MODEL BENCHMARK":

    st.markdown('<div class="page-kicker">03 / MODEL BENCHMARK</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Model Diagnostics.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="page-description">
        Inspect feature importance rankings and residual distributions for model explainability and performance monitoring.
        </div>
        """,
        unsafe_allow_html=True,
    )

    b_col1, b_col2 = st.columns(2, gap="large")

    def extract_estimator_and_features(model_obj):
        estimator = model_obj
        if hasattr(estimator, "named_steps"):
            estimator = list(estimator.named_steps.values())[-1]
        if hasattr(estimator, "regressor_"):
            estimator = estimator.regressor_

        feature_names = None
        if hasattr(model_obj, "feature_names_in_"):
            feature_names = model_obj.feature_names_in_
        elif hasattr(estimator, "feature_names_in_"):
            feature_names = estimator.feature_names_in_

        return estimator, feature_names

    raw_estimator, feature_names = extract_estimator_and_features(model)

    with b_col1:
        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-description">Relative impact of key features in predicting house prices.</div>', unsafe_allow_html=True)

        if hasattr(raw_estimator, "feature_importances_"):
            importances = raw_estimator.feature_importances_
            
            if feature_names is not None and len(feature_names) == len(importances):
                cols = list(feature_names)
            else:
                cols = [f"Feature {i}" for i in range(len(importances))]

            fi_df = pd.DataFrame({
                "Feature": cols,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(10)

            fig_fi = px.bar(
                fi_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Purples",
            )
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})
            fig_fi = style_figure(fig_fi, height=380)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance data is unavailable for this estimator.")

    with b_col2:
        st.markdown('<div class="section-title">Actual vs. Predicted (Sample)</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-description">Model prediction alignment relative to actual prices in training data.</div>', unsafe_allow_html=True)

        try:
            processed_df = add_engineered_features(df)
            expected_cols = model_expected_columns(model)

            if expected_cols is None and feature_names is not None:
                expected_cols = list(feature_names)

            if expected_cols:
                sample_data = processed_df.head(200).copy()
                preds = model.predict(sample_data)
                
                eval_df = pd.DataFrame({
                    "Actual": sample_data["SalePrice"],
                    "Predicted": preds
                })

                fig_diag = px.scatter(
                    eval_df,
                    x="Actual",
                    y="Predicted",
                    labels={"Actual": "Actual Price ($)", "Predicted": "Predicted Price ($)"},
                )
                
                max_val = max(eval_df["Actual"].max(), eval_df["Predicted"].max())
                fig_diag.add_shape(
                    type="line",
                    x0=0, y0=0, x1=max_val, y1=max_val,
                    line=dict(color=PURPLE, dash="dash", width=1.5)
                )
                
                fig_diag = style_figure(fig_diag, height=380)
                st.plotly_chart(fig_diag, use_container_width=True)
            else:
                st.info("Could not extract feature specifications from the model.")
        except Exception as exc:
            st.warning("Could not render prediction plot.")
            with st.expander("Error details"):
                st.code(str(exc))