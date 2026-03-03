import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

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
    
    st.title("🚗 Car Judo - What Can You REALLY Afford?")
    st.markdown("Stop guessing. Know exactly what your budget gets you - **including all hidden costs.**")
    
    # Initialize the system
    try:
        csv_path = "Ford_150.csv"
        car_judo = CarJudoF150System(csv_path)
    except FileNotFoundError:
        st.error(f"❌ Could not find file: {csv_path}")
        st.info("Please make sure Ford_150.csv is in the same directory as this app.")
        return
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        return
    
    # Main interface
    st.sidebar.markdown("### 🚗 Car Judo")
    st.sidebar.markdown("*F-150 Budget Intelligence*")
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Your Budget")
    
    # User inputs
    # Fixed vehicle type for prototype
    st.sidebar.markdown("**Vehicle Type:** F-150")
    
    # Get available trims
    available_trims = sorted(car_judo.df_clean['Trim'].unique())
    trim = st.sidebar.selectbox("Preferred Trim", available_trims)
    
    budget = st.sidebar.number_input("Budget ($)", min_value=5000, max_value=100000, value=25000, step=1000)
    annual_mileage = st.sidebar.slider("Annual Mileage", 5000, 30000, 15000, 1000)
    
    # Budget validation
    if budget < 10000:
        st.sidebar.warning("⚠️ Low budget may limit options to older/higher-mileage vehicles")
    elif budget > 50000:
        st.sidebar.success("✨ Great budget! You should find excellent options")
    
    # Analyze button
    if st.sidebar.button("🚗 ANALYZE", type="primary"):
        with st.spinner("Analyzing your F-150 options..."):
            # Get predictions
            predictions = car_judo.predict_car_specifications(budget, trim)
            
            if 'error' in predictions:
                st.error(f"Prediction error: {predictions['error']}")
                return
            
            # Calculate TCO
            tco_analysis = car_judo.calculate_tco(predictions, budget, annual_mileage)
            
            if 'error' in tco_analysis:
                st.error(f"TCO calculation error: {tco_analysis['error']}")
                return
            
            # Display results
            st.success(f"🎯 BOOM! Your ${budget:,} budget gets you:")
            
            # Main results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Expected Specifications")
                st.write(f"**Vehicle:** {predictions['vehicle']}")
                st.write(f"**Trim:** {predictions['trim']}")
                st.write(f"**Year:** {predictions['year']} (range: {predictions['year_range'][0]}-{predictions['year_range'][1]})")
                st.write(f"**Mileage:** {predictions['mileage']:,} miles (range: {predictions['mileage_range'][0]:,}-{predictions['mileage_range'][1]:,})")
                st.write(f"**Budget:** ${budget:,}")
            
            with col2:
                st.subheader("Total Cost Breakdown")
                st.write(f"**Purchase Price:** ${tco_analysis['purchase_price']:,}")
                st.write(f"**Annual Maintenance:** ${tco_analysis['annual_maintenance']:,.0f}")
                st.write(f"**Annual Insurance:** ${tco_analysis['annual_insurance']:,.0f}")
                st.write(f"**Annual Fuel:** ${tco_analysis['annual_fuel']:,.0f}")
                st.write(f"**Annual Depreciation:** ${tco_analysis['annual_depreciation']:,.0f}")
            
            # TCO Summary
            st.subheader("Total Cost of Ownership")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Annual Ownership Cost", f"${tco_analysis['annual_tco']:,.0f}")
                st.write("*(excluding purchase price)*")
            
            with col2:
                st.metric("5-Year Total Cost", f"${tco_analysis['total_5yr_tco']:,.0f}")
                st.write("*(including purchase)*")
            
            with col3:
                monthly_cost = tco_analysis['annual_tco'] / 12
                st.metric("Monthly Cost", f"${monthly_cost:,.0f}")
                st.write("*(ownership only)*")
            
            # Visual breakdown
            st.subheader("5-Year Cost Breakdown")
            
            cost_data = {
                'Purchase Price': tco_analysis['purchase_price'],
                'Maintenance': tco_analysis['total_maintenance'],
                'Insurance': tco_analysis['total_insurance'],
                'Fuel': tco_analysis['total_fuel'],
                'Depreciation': tco_analysis['total_depreciation']
            }
            
            # Create bar chart
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            ax.bar(cost_data.keys(), cost_data.values(), color=colors)
            ax.set_ylabel('Cost ($)')
            ax.set_title('5-Year Total Cost of Ownership Breakdown')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Insights
            st.subheader("Smart Insights")
            
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
                st.info(insight)
    
    # Educational section
    with st.expander("What is Total Cost of Ownership?"):
        st.markdown("""
        **TCO includes all costs over 5 years:**
        - **Purchase Price:** One-time cost to buy the vehicle
        - **Maintenance:** Oil changes, repairs, tires, brakes
        - **Insurance:** Coverage costs (varies by trim and value)
        - **Fuel:** Gas costs based on your annual mileage
        - **Depreciation:** Value loss over time
        
        **Car Judo shows you the REAL cost** - not just the sticker price!
        """)
    
    # Branded footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <strong>Car Judo</strong> - Stop Overpaying. Start Driving Smarter.<br>
            <em>F-150 Budget Intelligence for Smart Truck Buyers</em>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
