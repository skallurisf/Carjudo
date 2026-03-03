import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Modern App Styling
def set_app_style():
    st.markdown("""
    <style>
    /* Import system font stack for consistency */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Header styling */
    .app-header {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .app-header h1 {
        color: #1a1a1a;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .app-header p {
        color: #4a4a4a;
        font-size: 1.4rem;
        margin: 0;
        font-weight: 500;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.2);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .card h2 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 1rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .card h3 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.8rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .card p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .metric-label {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 500;
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Input styling */
    .stSelectbox > div > div > select {
        background: white;
        border: 2px solid #e1e8ed;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 500;
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stNumberInput > div > div > input {
        background: white;
        border: 2px solid #e1e8ed;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 500;
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: #1a1a1a;
        border: none;
        border-radius: 10px;
        padding: 1.2rem 2rem;
        font-weight: 700;
        font-size: 1.3rem;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.4);
        transition: all 0.3s ease;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(79, 172, 254, 0.6);
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: #1a1a1a;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Streamlit text elements */
    .stMarkdown {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1a1a1a;
    }
    
    .stMarkdown h1 {
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stMarkdown h2 {
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stMarkdown h3 {
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stMarkdown p {
        color: #1a1a1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Labels and descriptions */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #1a1a1a;
        font-weight: 600;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Expander styling */
    .stExpander {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stExpander > div > div > button {
        color: #1a1a1a !important;
        font-weight: 600;
        font-size: 1.1rem;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .stExpander button {
        color: #1a1a1a !important;
    }
    
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
    }
    
    /* Hide streamlit elements */
    .stDeployButton {
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
    
    # Apply modern styling
    set_app_style()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🚗 Car Judo</h1>
        <p>What Can You REALLY Afford? Stop guessing. Know exactly what your budget gets you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the system
    try:
        csv_path = "Ford_150.csv"
        car_judo = CarJudoF150System(csv_path)
    except FileNotFoundError:
        st.markdown(f"""
        <div class="card">
            <h2>❌ Data File Missing</h2>
            <p>Could not find <code>{csv_path}</code></p>
            <p>Please make sure the data file is in the same directory as this app.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    except Exception as e:
        st.markdown(f"""
        <div class="card">
            <h2>❌ System Error</h2>
            <p>Error loading system: {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main interface with columns
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("")  # Empty spacer
    
    with col2:
        # Input card
        st.markdown("""
        <div class="card">
            <h2>💰 Your Budget Configuration</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Get available trims
        available_trims = sorted(car_judo.df_clean['Trim'].unique())
        
        # Vehicle type (fixed)
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <strong>🚗 Vehicle Type:</strong> F-150
        </div>
        """, unsafe_allow_html=True)
        
        # User inputs
        trim = st.selectbox("🎯 Preferred Trim", available_trims)
        budget = st.number_input("💵 Budget ($)", min_value=5000, max_value=100000, value=25000, step=1000)
        annual_mileage = st.slider("📊 Annual Mileage", 5000, 30000, 15000, 1000)
        
        # Budget validation
        if budget < 10000:
            st.markdown("""
            <div style="background: #fff3cd; color: #856404; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                ⚠️ Low budget may limit options to older/higher-mileage vehicles
            </div>
            """, unsafe_allow_html=True)
        elif budget > 50000:
            st.markdown("""
            <div style="background: #d4edda; color: #155724; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                ✨ Great budget! You should find excellent options
            </div>
            """, unsafe_allow_html=True)
        
        # Analyze button
        analyze_clicked = st.button("🚗 ANALYZE MY BUDGET", type="primary", use_container_width=True)
    
    with col3:
        st.markdown("")  # Empty spacer
    
    # Results section
    if analyze_clicked:
        with st.spinner("🔍 Analyzing your F-150 options..."):
            # Get predictions
            predictions = car_judo.predict_car_specifications(budget, trim)
            
            if 'error' in predictions:
                st.markdown(f"""
                <div class="card">
                    <h2>❌ Prediction Error</h2>
                    <p>{predictions['error']}</p>
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Calculate TCO
            tco_analysis = car_judo.calculate_tco(predictions, budget, annual_mileage)
            
            if 'error' in tco_analysis:
                st.markdown(f"""
                <div class="card">
                    <h2>❌ TCO Calculation Error</h2>
                    <p>{tco_analysis['error']}</p>
                </div>
                """, unsafe_allow_html=True)
                return
            
            # Success message
            st.markdown(f"""
            <div class="success-message">
                🎯 BOOM! Your ${budget:,} budget gets you:
            </div>
            """, unsafe_allow_html=True)
            
            # Main results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="card">
                    <h2>📋 Expected Specifications</h2>
                    <div style="display: grid; gap: 1rem;">
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px;">
                            <strong>🚗 Vehicle:</strong> {vehicle}<br>
                            <strong>🎨 Trim:</strong> {trim}<br>
                            <strong>📅 Year:</strong> {year} ({year_range[0]}-{year_range[1]})<br>
                            <strong>📊 Mileage:</strong> {mileage:,} miles ({mileage_range[0]:,}-{mileage_range[1]:,})<br>
                            <strong>💰 Budget:</strong> ${budget:,}
                        </div>
                    </div>
                </div>
                """.format(
                    vehicle=predictions['vehicle'],
                    trim=predictions['trim'],
                    year=predictions['year'],
                    year_range=predictions['year_range'],
                    mileage=predictions['mileage'],
                    mileage_range=predictions['mileage_range'],
                    budget=budget
                ), unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <h2>💸 Total Cost Breakdown</h2>
                    <div style="display: grid; gap: 1rem;">
                        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px;">
                            <strong>💵 Purchase Price:</strong> ${tco_analysis['purchase_price']:,}<br>
                            <strong>🔧 Annual Maintenance:</strong> ${tco_analysis['annual_maintenance']:,.0f}<br>
                            <strong>🛡️ Annual Insurance:</strong> ${tco_analysis['annual_insurance']:,.0f}<br>
                            <strong>⛽ Annual Fuel:</strong> ${tco_analysis['annual_fuel']:,.0f}<br>
                            <strong>📉 Annual Depreciation:</strong> ${tco_analysis['annual_depreciation']:,.0f}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # TCO Summary with metric cards
            st.markdown('<h2 style="text-align: center; color: #1a1a1a; margin-bottom: 2rem; font-family: \"Inter\", -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif;">💰 Total Cost of Ownership</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${tco_analysis['annual_tco']:,.0f}</div>
                    <div class="metric-label">Annual Ownership Cost<br><small>(excluding purchase price)</small></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${tco_analysis['total_5yr_tco']:,.0f}</div>
                    <div class="metric-label">5-Year Total Cost<br><small>(including purchase)</small></div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                monthly_cost = tco_analysis['annual_tco'] / 12
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">${monthly_cost:,.0f}</div>
                    <div class="metric-label">Monthly Cost<br><small>(ownership only)</small></div>
                </div>
                """, unsafe_allow_html=True)
            
            # Visual breakdown
            st.markdown("""
            <div class="card">
                <h2>📊 5-Year Cost Breakdown</h2>
            </div>
            """, unsafe_allow_html=True)
            
            cost_data = {
                'Purchase Price': tco_analysis['purchase_price'],
                'Maintenance': tco_analysis['total_maintenance'],
                'Insurance': tco_analysis['total_insurance'],
                'Fuel': tco_analysis['total_fuel'],
                'Depreciation': tco_analysis['total_depreciation']
            }
            
            # Create bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#4facfe', '#00f2fe', '#43e97b', '#38f9d7', '#fa709a']
            bars = ax.bar(cost_data.keys(), cost_data.values(), color=colors)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:,.0f}',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
            ax.set_title('5-Year Total Cost of Ownership Breakdown', fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, fontsize=11)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Smart Insights
            st.markdown("""
            <div class="card">
                <h2>🧠 Smart Insights</h2>
            </div>
            """, unsafe_allow_html=True)
            
            insights = []
            
            # Purchase vs total cost ratio
            purchase_ratio = tco_analysis['purchase_price'] / tco_analysis['total_5yr_tco']
            if purchase_ratio < 0.4:
                insights.append("💰 Purchase price is less than 40% of total cost - focus on maintenance and fuel efficiency!")
            elif purchase_ratio > 0.6:
                insights.append("🚗 Purchase price dominates total cost - consider negotiating harder on price!")
            
            # Age analysis
            if predictions['year'] < 2015:
                insights.append("⚠️ Expected older vehicle - budget more for maintenance and repairs")
            elif predictions['year'] > 2022:
                insights.append("✨ Nearly new vehicle - lower maintenance but higher depreciation")
            
            # Mileage analysis
            if predictions['mileage'] > 100000:
                insights.append("🔧 High mileage expected - factor in potential major repairs")
            
            for insight in insights:
                st.markdown(f"""
                <div class="card" style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 4px solid #2196f3;">
                    <p style="margin: 0; font-size: 1.1rem;">{insight}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Educational section
    with st.expander("📚 What is Total Cost of Ownership?"):
        st.markdown("""
        <div class="card">
            <h3>TCO includes all costs over 5 years:</h3>
            <ul>
                <li><strong>💵 Purchase Price:</strong> One-time cost to buy the vehicle</li>
                <li><strong>🔧 Maintenance:</strong> Oil changes, repairs, tires, brakes</li>
                <li><strong>🛡️ Insurance:</strong> Coverage costs (varies by trim and value)</li>
                <li><strong>⛽ Fuel:</strong> Gas costs based on your annual mileage</li>
                <li><strong>📉 Depreciation:</strong> Value loss over time</li>
            </ul>
            <p><strong>🎯 Car Judo shows you the REAL cost</strong> - not just the sticker price!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; color: #1a1a1a; padding: 2rem; margin-top: 3rem; font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <strong style='font-size: 1.5rem; font-weight: 700;'>🚗 Car Judo</strong><br>
        <span style='font-size: 1.2rem; font-weight: 500;'>Stop Overpaying. Start Driving Smarter.</span><br>
        <em style='opacity: 0.8; font-weight: 400;'>F-150 Budget Intelligence for Smart Truck Buyers</em>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
