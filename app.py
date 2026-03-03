import re
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------
# Helpers
# ----------------------------
def _to_number(x):
    """Extract numeric value from messy strings like '$54,973' or '12,345'."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[^0-9.]", "", s)  # keep digits + dot
    return float(s) if s != "" else np.nan


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv("Ateek_Ford.csv")

    # normalize column names (handle different spellings)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    col_year = pick("Year")
    col_mileage = pick("Mileage.Done", "Mileage Done", "Mileage")
    col_price = pick("Price.in.USD", "Price in USD", "Price")

    # Optional categorical cols if present
    col_model = pick("MODEL", "Vehicle_Model", "Model")
    col_trim = pick("Trim")
    col_drive = pick("Drive.Train", "Drive Train")
    col_state = pick("Location..State.", "Location (State)", "Location")

    # Basic required
    if col_year is None or col_mileage is None or col_price is None:
        raise ValueError(
            f"CSV must include Year + Mileage + Price columns. Found columns: {list(df.columns)}"
        )

    # Clean numeric
    df["Year"] = pd.to_numeric(df[col_year], errors="coerce").astype("Int64")
    df["Mileage"] = df[col_mileage].apply(_to_number)
    df["Price"] = df[col_price].apply(_to_number)

    # Vehicle age
    current_year = datetime.now().year
    df["Vehicle_Age"] = (current_year - df["Year"]).clip(lower=1)

    # Extra derived feature
    df["Mileage_per_Year"] = df["Mileage"] / df["Vehicle_Age"].replace(0, 1)

    # Standardize drivetrain (optional)
    if col_drive is not None:
        s = df[col_drive].astype(str).str.lower().str.strip()
        s = s.replace(
            {
                "four-wheel drive": "awd",
                "four wheel drive": "awd",
                "4wd": "awd",
                "all-wheel drive": "awd",
                "awd": "awd",
                "front-wheel drive": "fwd",
                "fwd": "fwd",
                "rear-wheel drive": "rwd",
                "rwd": "rwd",
            }
        )
        df["DriveTrain"] = s
    else:
        df["DriveTrain"] = np.nan

    # Optional cats
    df["MODEL"] = df[col_model] if col_model is not None else np.nan
    df["Trim"] = df[col_trim] if col_trim is not None else np.nan
    df["State"] = df[col_state] if col_state is not None else np.nan

    # Keep only rows with required numeric targets
    df = df.dropna(subset=["Year", "Mileage", "Price", "Vehicle_Age", "Mileage_per_Year"])

    return df


def build_price_model(df: pd.DataFrame):
    # Features to use (numeric + any available categoricals)
    numeric_features = ["Vehicle_Age", "Mileage", "Mileage_per_Year"]
    categorical_features = ["MODEL", "Trim", "DriveTrain", "State"]

    # Only keep categoricals that have at least some non-null values
    cat_use = [c for c in categorical_features if df[c].notna().any()]

    X = df[numeric_features + cat_use]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, cat_use),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=123,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return pipe, {"mae": mae, "rmse": rmse, "r2": r2, "cat_features_used": cat_use}


def estimate_maintenance(year: int, mileage: float, drivetrain: str | None):
    """
    Heuristic maintenance estimate (because most listing datasets don't have real repair cost).
    Returns an annual estimate range.
    """
    current_year = datetime.now().year
    age = max(current_year - int(year), 1)
    miles_per_year = mileage / age

    base = 600
    age_component = 120 * age
    usage_component = 140 * (miles_per_year / 10000)

    dt = (drivetrain or "").lower().strip()
    awd_premium = 250 if dt in {"awd", "4wd", "four-wheel drive", "four wheel drive", "all-wheel drive"} else 0

    est = base + age_component + usage_component + awd_premium
    # add uncertainty band
    low = max(250, est * 0.75)
    high = est * 1.25
    return float(low), float(high)


def deal_label(asking_price: float, fair_price: float, mae: float):
    """
    Label deal quality using the model's MAE as a rough tolerance band.
    """
    # band widens if model is noisier
    band = max(1000.0, mae)
    diff = asking_price - fair_price

    if diff <= -band:
        return "🔥 Good deal", diff
    if abs(diff) < band:
        return "✅ Fair", diff
    return "⚠️ Overpriced", diff


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CarJudo – Used Car Decision Assistant", layout="centered")
st.title("🚗 CarJudo – Used Car Decision Assistant")
st.write("Enter basic listing info → get **fair price**, **deal rating**, and **maintenance / 5-year cost** estimate.")

with st.sidebar:
    st.header("Data + Model")
    csv_path = st.text_input("CSV path", value="Ford_150.csv")
    retrain = st.button("Train / Reload Model")

@st.cache_data(show_spinner=False)
def _load_df(path):
    return load_and_prepare(path)

@st.cache_resource(show_spinner=False)
def _train(df):
    return build_price_model(df)

# Train model on load or when user clicks retrain
try:
    df = _load_df(csv_path)
    model, metrics = _train(df) if (retrain or True) else _train(df)
except Exception as e:
    st.error(f"Could not load/train from CSV: {e}")
    st.stop()

st.caption(f"Training rows used: **{len(df):,}** | R²: **{metrics['r2']:.3f}** | MAE: **${metrics['mae']:,.0f}**")
if metrics["cat_features_used"]:
    st.caption(f"Using categoricals: {', '.join(metrics['cat_features_used'])}")

st.divider()
st.subheader("🧾 Listing Inputs")

current_year = datetime.now().year
year = st.number_input("Year", min_value=1990, max_value=current_year, value=max(2000, current_year - 5), step=1)
mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=30000, step=1000)
asking_price = st.number_input("Asking price ($)", min_value=0, max_value=500000, value=25000, step=500)

# Optional inputs (only show if model used them)
model_in = None
trim_in = None
dt_in = None
state_in = None

cols = st.columns(2)
if "MODEL" in metrics["cat_features_used"]:
    model_in = cols[0].text_input("Model (optional)", value="")
if "Trim" in metrics["cat_features_used"]:
    trim_in = cols[1].text_input("Trim (optional)", value="")

cols2 = st.columns(2)
if "DriveTrain" in metrics["cat_features_used"]:
    dt_in = cols2[0].selectbox("Drive train (optional)", options=["", "fwd", "rwd", "awd"], index=0)
else:
    # still collect for maintenance heuristic
    dt_in = cols2[0].selectbox("Drive train (for maintenance estimate)", options=["", "fwd", "rwd", "awd"], index=0)

if "State" in metrics["cat_features_used"]:
    state_in = cols2[1].text_input("State/Location (optional)", value="")

if st.button("Evaluate listing"):
    age = max(current_year - int(year), 1)
    miles_per_year = mileage / age

    row = {
        "Vehicle_Age": age,
        "Mileage": float(mileage),
        "Mileage_per_Year": float(miles_per_year),
    }
    if "MODEL" in metrics["cat_features_used"]:
        row["MODEL"] = model_in if model_in else np.nan
    if "Trim" in metrics["cat_features_used"]:
        row["Trim"] = trim_in if trim_in else np.nan
    if "DriveTrain" in metrics["cat_features_used"]:
        row["DriveTrain"] = dt_in if dt_in else np.nan
    if "State" in metrics["cat_features_used"]:
        row["State"] = state_in if state_in else np.nan

    X_new = pd.DataFrame([row])
    fair_price = float(model.predict(X_new)[0])

    label, diff = deal_label(float(asking_price), fair_price, metrics["mae"])
    low_m, high_m = estimate_maintenance(int(year), float(mileage), dt_in)

    # 5-year rough ownership cost (simple version)
    ownership_years = 5
    mpg = 22  # rough default; can be improved later
    fuel_price = 3.8
    annual_miles = miles_per_year  # use inferred usage
    fuel_cost_5y = (annual_miles / mpg) * fuel_price * ownership_years
    maint_5y_mid = ((low_m + high_m) / 2.0) * ownership_years
    resale = fair_price * 0.5  # rough depreciation proxy

    tco_5y = asking_price + fuel_cost_5y + maint_5y_mid - resale

    st.subheader("✅ Results")
    st.metric("Fair price estimate", f"${fair_price:,.0f}")
    st.metric("Deal rating", label, f"{diff:+,.0f} vs fair price")

    st.write("**Maintenance estimate (annual):** "
             f"${low_m:,.0f} – ${high_m:,.0f}")

    st.write("**5-year rough ownership cost (TCO):** "
             f"${tco_5y:,.0f}")
    st.caption("Note: Maintenance + TCO are heuristic estimates unless you train on real repair-cost data.")