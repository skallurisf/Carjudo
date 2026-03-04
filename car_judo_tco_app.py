import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# ------------------------------
# Modern App Styling
# ------------------------------
def set_app_style():
    st.markdown("""
    <style>

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .stApp {
        background: linear-gradient(180deg,#eef4ff,#dce9ff);
        font-family: 'Inter', sans-serif;
        color:#1a1a1a;
    }

    /* Center app like real product */

    .block-container {
        max-width: 850px;
        margin: auto;
        padding-top: 2rem;
    }

    /* Header */

    .app-header{
        background: linear-gradient(135deg,#1e3a8a,#3b82f6);
        color:white;
        padding:2rem;
        border-radius:18px;
        text-align:center;
        margin-bottom:2rem;
        box-shadow:0 10px 35px rgba(0,0,0,0.15);
    }

    .app-header h1{
        font-size:3rem;
        margin-bottom:5px;
        color:white;
    }

    .app-header p{
        font-size:1.2rem;
        color:#e0e7ff;
    }

    /* Cards */

    .card{
        background:white;
        padding:1.5rem;
        border-radius:14px;
        box-shadow:0 8px 30px rgba(0,0,0,0.08);
        margin-bottom:1.5rem;
        border:1px solid #e5e7eb;
    }

    /* Metric cards */

    .metric-card{
        background: linear-gradient(135deg,#3b82f6,#60a5fa);
        padding:1.5rem;
        border-radius:14px;
        text-align:center;
        color:white;
        box-shadow:0 8px 25px rgba(59,130,246,0.4);
    }

    .metric-card:hover{
        transform: translateY(-4px);
        transition:0.2s;
    }

    .metric-value{
        font-size:2.5rem;
        font-weight:700;
    }

    .metric-label{
        font-size:1rem;
        opacity:.9;
    }

    /* Buttons */

    .stButton button{
        background: linear-gradient(135deg,#2563eb,#3b82f6);
        color:white;
        border-radius:10px;
        font-size:1.1rem;
        padding:.8rem 1.6rem;
        font-weight:600;
        border:none;
        box-shadow:0 4px 18px rgba(37,99,235,.4);
    }

    .stButton button:hover{
        background: linear-gradient(135deg,#1d4ed8,#2563eb);
        transform:translateY(-1px);
    }

    /* Input styling */

    input, select{
        border-radius:10px !important;
    }

    /* Fix label readability */

label, .stSelectbox label, .stNumberInput label, .stSlider label {
    font-size:25px !important;
    font-weight:900 !important;
    color:#111827 !important;
}

    </style>
    """, unsafe_allow_html=True)


# ------------------------------
# Core ML System
# ------------------------------
class CarJudoF150System:

    def __init__(self, csv_path):

        self.csv_path = csv_path
        self.df = None
        self.df_clean = None

        self.year_model = None
        self.mileage_model = None
        self.trim_encoder = None

        self.load_data()
        self.setup_models()
        self.setup_tco()

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)

        if 'Mileage Done' in self.df.columns:
            self.df['Mileage Done'] = self.df['Mileage Done'].astype(str).str.replace(',', '')
            self.df['Mileage Done'] = pd.to_numeric(self.df['Mileage Done'], errors='coerce')

        if 'Price in USD' in self.df.columns:
            self.df['Price in USD'] = self.df['Price in USD'].astype(str).str.replace('$','').str.replace(',','')
            self.df['Price in USD'] = pd.to_numeric(self.df['Price in USD'], errors='coerce')

    def setup_models(self):

        self.df_clean = self.df.dropna()

        for col in ['Price in USD','Mileage Done']:

            Q1 = self.df_clean[col].quantile(.25)
            Q3 = self.df_clean[col].quantile(.75)

            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            self.df_clean = self.df_clean[
                (self.df_clean[col]>=lower) &
                (self.df_clean[col]<=upper)
            ]

        self.trim_encoder = LabelEncoder()
        self.df_clean["Trim_encoded"] = self.trim_encoder.fit_transform(self.df_clean["Trim"].astype(str))

        X = self.df_clean[['Price in USD','Trim_encoded']]

        y_year = self.df_clean['Year']
        y_mileage = self.df_clean['Mileage Done']

        self.year_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.mileage_model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.year_model.fit(X,y_year)
        self.mileage_model.fit(X,y_mileage)

    def setup_tco(self):

        self.tco_rates = {

            'maintenance_per_mile':0.08,
            'maintenance_base':500,
            'insurance_base':1200,
            'insurance_per_1000':10,
            'fuel_per_mile':0.15,
            'depreciation':0.12

        }

    def predict(self,budget,trim):

        trim_encoded = self.trim_encoder.transform([trim])[0]

        X = np.array([[budget,trim_encoded]])

        year = int(self.year_model.predict(X)[0])
        mileage = int(self.mileage_model.predict(X)[0])

        return year,mileage

    def calculate_tco(self,budget,year,mileage,annual_mileage):

        rates = self.tco_rates

        maint = rates["maintenance_base"] + annual_mileage*rates["maintenance_per_mile"]
        ins = rates["insurance_base"] + (budget/1000)*rates["insurance_per_1000"]
        fuel = annual_mileage*rates["fuel_per_mile"]
        dep = budget*rates["depreciation"]

        annual = maint + ins + fuel + dep

        total5 = budget + annual*5

        return {
            "purchase":budget,
            "maintenance":maint*5,
            "insurance":ins*5,
            "fuel":fuel*5,
            "depreciation":dep*5,
            "annual":annual,
            "total5":total5
        }


# ------------------------------
# Streamlit App
# ------------------------------
def main():

    st.set_page_config(
        page_title="Car Judo",
        layout="wide",
        page_icon="🚗"
    )

    set_app_style()

    st.markdown("""
    <div class="app-header">
        <h1>🚗 Car Judo</h1>
        <p>Stop Guessing. Know Your True Car Cost.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;background:white;padding:10px 16px;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,0.05);">
<div style="font-size:24px;">👤</div>
<div style="font-size:16px;font-weight:600;">Guest User</div>
</div>
""", unsafe_allow_html=True)

    csv = "Ford_150.csv"
    car = CarJudoF150System(csv)

    st.markdown("""
<div class="card">
<h2 style="margin-bottom:0;color:#111827;">💰 Your Budget</h2>
</div>
""",unsafe_allow_html=True)

    trim = st.selectbox("🚙 Preferred Trim",sorted(car.df_clean["Trim"].unique()))
    budget = st.number_input("💰 Budget ($)",5000,100000,25000,1000)
    annual_miles = st.slider("🛣️ Annual Miles",5000,30000,15000,1000)

    if st.button("🚗 Find My Truck",use_container_width=True):

        with st.spinner("Running analysis..."):

            year,mileage = car.predict(budget,trim)
            tco = car.calculate_tco(budget,year,mileage,annual_miles)

        st.markdown(f"""
<div style="background:white;padding:20px;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,0.08);font-size:20px;color:black;margin-top:20px;">

🚘 <b>Your Recommended Truck</b><br><br>

<span style="font-size:22px;font-weight:700;">
{year} Ford F-150 {trim}
</span><br>

Mileage: <b>{mileage:,}</b> miles

</div>
""", unsafe_allow_html=True)

        st.image("F150.png",caption=f"Estimated vehicle: {year} Ford F-150 {trim}",use_container_width=True)

        col1,col2,col3 = st.columns(3)

        col1.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">${tco['annual']:,.0f}</div>
        <div class="metric-label">Annual Cost</div>
        </div>
        """,unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">${tco['total5']:,.0f}</div>
        <div class="metric-label">5-Year Cost</div>
        </div>
        """,unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card">
        <div class="metric-value">${tco['annual']/12:,.0f}</div>
        <div class="metric-label">Monthly Cost</div>
        </div>
        """,unsafe_allow_html=True)

        cost_data = {
            "Purchase":tco["purchase"],
            "Maintenance":tco["maintenance"],
            "Insurance":tco["insurance"],
            "Fuel":tco["fuel"],
            "Depreciation":tco["depreciation"]
        }

        df = pd.DataFrame({
            "Category":list(cost_data.keys()),
            "Cost":list(cost_data.values())
        })

        fig = px.bar(df,x="Category",y="Cost",color="Category",
                     title="True Cost of Ownership (5 Years)")

        fig.update_layout(template="plotly_white")

        st.plotly_chart(fig,use_container_width=True)

    st.markdown("""
    <div style='text-align:center;padding:30px'>
    <b>Car Judo</b><br>
    Stop Overpaying. Drive Smarter.
    </div>
    """,unsafe_allow_html=True)


if __name__ == "__main__":
    main()
