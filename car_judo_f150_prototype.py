import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Mobile app styling
def set_mobile_style():
    st.markdown("""
    <style>
    /* Hide sidebar and header */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    .stApp header {
        display: none !important;
    }
    
    /* Mobile app container */
    .stApp {
        background: #000000;
        min-height: 100vh;
        max-width: 428px;
        margin: 0 auto;
    }
    
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* Status bar */
    .status-bar {
        background: #000000;
        color: white;
        padding: 8px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 14px;
        font-weight: 500;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        padding: 20px;
        text-align: center;
        color: white;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .app-header h1 {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-header p {
        font-size: 16px;
        margin: 8px 0 0 0;
        opacity: 0.9;
    }
    
    /* Input section */
    .input-section {
        background: #1C1C1E;
        padding: 20px;
        color: white;
    }
    
    .input-section h3 {
        color: #FF6B35;
        font-size: 18px;
        font-weight: 600;
        margin: 0 0 20px 0;
    }
    
    /* Input cards */
    .input-card {
        background: #2C2C2E;
        border: 2px solid #3A3A3C;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        color: white;
        font-size: 16px;
    }
    
    .input-label {
        color: #98989D;
        font-size: 12px;
        margin-bottom: 4px;
        display: block;
    }
    
    .input-value {
        font-size: 18px;
        font-weight: 600;
        color: white;
    }
    
    /* Button */
    .analyze-btn {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 18px;
        font-size: 18px;
        font-weight: 600;
        width: calc(100% - 40px);
        margin: 20px;
        cursor: pointer;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        text-align: center;
    }
    
    /* Results section */
    .results-section {
        background: #1C1C1E;
        padding: 20px;
        color: white;
        min-height: calc(100vh - 400px);
    }
    
    .result-card {
        background: #2C2C2E;
        border: 2px solid #3A3A3C;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .result-card h3 {
        color: #FF6B35;
        font-size: 18px;
        font-weight: 600;
        margin: 0 0 16px 0;
    }
    
    /* Spec grid */
    .spec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin: 16px 0;
    }
    
    .spec-item {
        background: #1C1C1E;
        border: 2px solid #3A3A3C;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    
    .spec-label {
        color: #98989D;
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    .spec-value {
        color: white;
        font-size: 20px;
        font-weight: 700;
        margin: 0;
    }
    
    .spec-detail {
        color: #FF6B35;
        font-size: 11px;
        margin-top: 4px;
    }
    
    /* TCO display */
    .tco-display {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        margin-bottom: 16px;
    }
    
    .tco-label {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    
    .tco-value {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    
    .tco-subtitle {
        font-size: 12px;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    /* Cost breakdown */
    .cost-item {
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #3A3A3C;
    }
    
    .cost-item:last-child {
        border-bottom: none;
    }
    
    .cost-label {
        color: #98989D;
        font-size: 14px;
    }
    
    .cost-value {
        color: white;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* Hide streamlit elements */
    .stSelectbox, .stNumberInput, .stSlider, .stButton, .stSuccess, .stSpinner, .stMetric, .stPlotlyChart, .stInfo {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

class CarJudoF150System:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.df_clean = None
        self.year_model = None
        self.mileage_model = None
        self.trim_encoder = None
        
        # Load and setup
        self.load_data()
        self.setup_models()
        self.setup_tco_calculators()
    
    def load_data(self):
        """Load F-150 data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self._fix_data_types()
        except Exception as e:
            raise Exception(f"Could not load CSV file: {str(e)}")
    
    def _fix_data_types(self):
        """Fix data types for F-150 data"""
        # Convert string numbers with commas to numeric
        if 'Mileage Done' in self.df.columns:
            self.df['Mileage Done'] = self.df['Mileage Done'].astype(str).str.replace(',', '')
            self.df['Mileage Done'] = pd.to_numeric(self.df['Mileage Done'], errors='coerce')
        
        if 'Price in USD' in self.df.columns:
            self.df['Price in USD'] = self.df['Price in USD'].astype(str).str.replace('$', '').str.replace(',', '')
            self.df['Price in USD'] = pd.to_numeric(self.df['Price in USD'], errors='coerce')
    
    def setup_models(self):
        """Setup prediction models for F-150"""
        # Clean data
        self.df_clean = self.df.dropna()
        
        # Remove outliers
        for col in ['Price in USD', 'Mileage Done']:
            Q1 = self.df_clean[col].quantile(0.25)
            Q3 = self.df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df_clean = self.df_clean[(self.df_clean[col] >= lower_bound) & (self.df_clean[col] <= upper_bound)]
        
        # Reset index
        self.df_clean = self.df_clean.reset_index(drop=True)
        
        # Encode trim
        self.trim_encoder = LabelEncoder()
        self.df_clean['Trim_encoded'] = self.trim_encoder.fit_transform(self.df_clean['Trim'].astype(str))
        
        # Features and targets
        features = ['Price in USD', 'Trim_encoded']
        X = self.df_clean[features]
        y_year = self.df_clean['Year']
        y_mileage = self.df_clean['Mileage Done']
        
        # Train models
        self.year_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.mileage_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.year_model.fit(X, y_year)
        self.mileage_model.fit(X, y_mileage)
    
    def setup_tco_calculators(self):
        """Setup TCO calculation parameters"""
        self.tco_rates = {
            'maintenance_per_mile': 0.08,
            'maintenance_base_per_year': 500,
            'insurance_base_per_year': 1200,
            'insurance_per_1000_value': 10,
            'fuel_per_mile': 0.15,
            'depreciation_per_year': 0.12,
        }
        
        self.trim_adjustments = {
            'XL': {'insurance': 0.9, 'maintenance': 0.8},
            'XLT': {'insurance': 1.0, 'maintenance': 1.0},
            'Lariat': {'insurance': 1.2, 'maintenance': 1.1},
            'King Ranch': {'insurance': 1.3, 'maintenance': 1.2},
            'Platinum': {'insurance': 1.4, 'maintenance': 1.2},
            'Limited': {'insurance': 1.5, 'maintenance': 1.3},
        }
    
    def predict_car_specifications(self, budget, trim):
        """Predict year and mileage for given budget and trim"""
        try:
            # Encode trim
            if trim in self.trim_encoder.classes_:
                trim_encoded = self.trim_encoder.transform([trim])[0]
            else:
                # Use most common trim if specified trim not found
                most_common_trim = self.df_clean['Trim'].mode()[0]
                trim_encoded = self.trim_encoder.transform([most_common_trim])[0]
                trim = most_common_trim
            
            # Prepare input
            input_data = np.array([[budget, trim_encoded]])
            
            # Predict
            predicted_year = self.year_model.predict(input_data)[0]
            predicted_mileage = self.mileage_model.predict(input_data)[0]
            
            # Round to reasonable values
            predicted_year = int(round(predicted_year))
            predicted_mileage = int(round(predicted_mileage))
            
            # Calculate confidence ranges
            year_range = (predicted_year - 1, predicted_year + 1)
            mileage_range = (predicted_mileage - 10000, predicted_mileage + 10000)
            
            return {
                'vehicle': 'F-150',
                'trim': trim,
                'year': predicted_year,
                'mileage': predicted_mileage,
                'year_range': year_range,
                'mileage_range': mileage_range
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_tco(self, predictions, budget, annual_mileage=15000):
        """Calculate Total Cost of Ownership"""
        try:
            year = predictions['year']
            mileage = predictions['mileage']
            trim = predictions['trim']
            current_year = 2025
            car_age = current_year - year
            
            # Get trim adjustments
            trim_adj = self.trim_adjustments.get(trim, {'insurance': 1.0, 'maintenance': 1.0})
            
            # Purchase price
            purchase_price = budget
            
            # Annual maintenance costs
            maintenance_base = self.tco_rates['maintenance_base_per_year'] * trim_adj['maintenance']
            maintenance_mileage = annual_mileage * self.tco_rates['maintenance_per_mile'] * trim_adj['maintenance']
            annual_maintenance = maintenance_base + maintenance_mileage
            
            # Annual insurance
            insurance_base = self.tco_rates['insurance_base_per_year'] * trim_adj['insurance']
            insurance_value = (purchase_price / 1000) * self.tco_rates['insurance_per_1000_value'] * trim_adj['insurance']
            annual_insurance = insurance_base + insurance_value
            
            # Annual fuel costs
            annual_fuel = annual_mileage * self.tco_rates['fuel_per_mile']
            
            # Annual depreciation
            annual_depreciation = purchase_price * self.tco_rates['depreciation_per_year']
            
            # Calculate 5-year TCO
            years_ownership = 5
            total_maintenance = annual_maintenance * years_ownership
            total_insurance = annual_insurance * years_ownership
            total_fuel = annual_fuel * years_ownership
            total_depreciation = annual_depreciation * years_ownership
            
            total_tco = purchase_price + total_maintenance + total_insurance + total_fuel + total_depreciation
            
            return {
                'purchase_price': purchase_price,
                'annual_maintenance': annual_maintenance,
                'annual_insurance': annual_insurance,
                'annual_fuel': annual_fuel,
                'annual_depreciation': annual_depreciation,
                'annual_tco': annual_maintenance + annual_insurance + annual_fuel + annual_depreciation,
                'total_5yr_tco': total_tco,
                'total_5yr_costs': total_maintenance + total_insurance + total_fuel + total_depreciation,
                'total_maintenance': total_maintenance,
                'total_insurance': total_insurance,
                'total_fuel': total_fuel,
                'total_depreciation': total_depreciation
            }
        except Exception as e:
            return {'error': str(e)}

def main():
    st.set_page_config(
        page_title="Car Judo - F-150 Intelligence", 
        layout="wide",
        page_icon="🚗"
    )
    
    # Apply mobile styling
    set_mobile_style()
    
    # Status bar
    st.markdown("""
    <div class="status-bar">
        <span>9:41</span>
        <span>📶 🔋</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🚗 Car Judo</h1>
        <p>What Can You REALLY Afford?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the system
    try:
        csv_path = "Ford_150.csv"
        car_judo = CarJudoF150System(csv_path)
    except FileNotFoundError:
        st.markdown("""
        <div class="results-section">
            <div class="result-card">
                <h3>❌ Error</h3>
                <p>Could not find Ford_150.csv file</p>
                <p>Please make sure the data file is in the same directory</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    except Exception as e:
        st.markdown(f"""
        <div class="results-section">
            <div class="result-card">
                <h3>❌ Error</h3>
                <p>{str(e)}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Input section
    st.markdown("""
    <div class="input-section">
        <h3>💰 Your Budget</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get available trims
    available_trims = sorted(car_judo.df_clean['Trim'].unique())
    
    # Hidden inputs
    budget = st.number_input("Budget", min_value=5000, max_value=100000, value=25000, step=1000, key="budget")
    trim = st.selectbox("Trim", available_trims, key="trim")
    annual_mileage = st.slider("Mileage", 5000, 30000, 15000, 1000, key="mileage")
    
    # Display inputs with mobile styling
    st.markdown(f"""
    <div class="input-section">
        <div class="input-card">
            <span class="input-label">VEHICLE TYPE</span>
            <div class="input-value">F-150</div>
        </div>
        <div class="input-card">
            <span class="input-label">PREFERRED TRIM</span>
            <div class="input-value">{trim}</div>
        </div>
        <div class="input-card">
            <span class="input-label">BUDGET</span>
            <div class="input-value">${budget:,}</div>
        </div>
        <div class="input-card">
            <span class="input-label">ANNUAL MILEAGE</span>
            <div class="input-value">{annual_mileage:,} miles</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Analyze button
    analyze_clicked = st.button("ANALYZE", key="analyze")
    
    if analyze_clicked:
        # Show loading
        st.markdown("""
        <div class="analyze-btn">
            Analyzing your options...
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Analyzing your F-150 options..."):
            # Get predictions
            predictions = car_judo.predict_car_specifications(budget, trim)
            
            if 'error' not in predictions:
                # Calculate TCO
                tco_analysis = car_judo.calculate_tco(predictions, budget, annual_mileage)
                
                if 'error' not in tco_analysis:
                    # Results section
                    monthly_cost = tco_analysis['annual_tco'] / 12
                    
                    st.markdown(f"""
                    <div class="results-section">
                        <div class="result-card">
                            <h3>🎯 BOOM! Your ${budget:,} budget gets you:</h3>
                        </div>
                        
                        <div class="result-card">
                            <h3>📋 Expected Specifications</h3>
                            <div class="spec-grid">
                                <div class="spec-item">
                                    <div class="spec-label">VEHICLE</div>
                                    <div class="spec-value">F-150</div>
                                    <div class="spec-detail">{predictions['trim']}</div>
                                </div>
                                <div class="spec-item">
                                    <div class="spec-label">YEAR</div>
                                    <div class="spec-value">{predictions['year']}</div>
                                    <div class="spec-detail">{predictions['year_range'][0]}-{predictions['year_range'][1]}</div>
                                </div>
                                <div class="spec-item">
                                    <div class="spec-label">MILEAGE</div>
                                    <div class="spec-value">{predictions['mileage']:,}</div>
                                    <div class="spec-detail">{predictions['mileage_range'][0]:,}-{predictions['mileage_range'][1]:,}</div>
                                </div>
                                <div class="spec-item">
                                    <div class="spec-label">BUDGET</div>
                                    <div class="spec-value">${budget:,}</div>
                                    <div class="spec-detail">Your budget</div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="tco-display">
                            <div class="tco-label">5-YEAR TOTAL COST</div>
                            <div class="tco-value">${tco_analysis['total_5yr_tco']:,.0f}</div>
                            <div class="tco-subtitle">${monthly_cost:,.0f}/month ownership cost</div>
                        </div>
                        
                        <div class="result-card">
                            <h3>💸 Annual Costs</h3>
                            <div class="cost-item">
                                <span class="cost-label">🔧 Maintenance</span>
                                <span class="cost-value">${tco_analysis['annual_maintenance']:,.0f}</span>
                            </div>
                            <div class="cost-item">
                                <span class="cost-label">🛡️ Insurance</span>
                                <span class="cost-value">${tco_analysis['annual_insurance']:,.0f}</span>
                            </div>
                            <div class="cost-item">
                                <span class="cost-label">⛽ Fuel</span>
                                <span class="cost-value">${tco_analysis['annual_fuel']:,.0f}</span>
                            </div>
                            <div class="cost-item">
                                <span class="cost-label">📉 Depreciation</span>
                                <span class="cost-value">${tco_analysis['annual_depreciation']:,.0f}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Analyze button (visible)
    st.markdown(f"""
    <div class="analyze-btn" onclick="document.querySelector('[data-testid=\"stButton\"] button').click()">
        🚗 ANALYZE MY BUDGET
    </div>
    """, unsafe_allow_html=True)
    
    # Add JavaScript to handle button clicks
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const mobileButton = document.querySelector('.analyze-btn');
        if (mobileButton) {
            mobileButton.addEventListener('click', function() {
                const hiddenButton = document.querySelector('[data-testid="stButton"] button');
                if (hiddenButton) {
                    hiddenButton.click();
                }
            });
        }
    });
    </script>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
