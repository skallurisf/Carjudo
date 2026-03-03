import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CarJudoTCOSystem:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.model = None
        self.year_model = None
        self.mileage_model = None
        self.label_encoders = {}
        self.feature_columns = []
        
        # Fix data types first
        self._fix_data_types()
        self.setup_models()
        self.setup_tco_calculators()
    
    def _fix_data_types(self):
        """Fix string data types to numeric"""
        if 'Mileage Done' in self.df.columns:
            self.df['Mileage Done'] = self.df['Mileage Done'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['Mileage Done'] = pd.to_numeric(self.df['Mileage Done'], errors='coerce')
        
        if 'Price in USD' in self.df.columns:
            self.df['Price in USD'] = self.df['Price in USD'].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
            self.df['Price in USD'] = pd.to_numeric(self.df['Price in USD'], errors='coerce')
    
    def setup_models(self):
        """Setup prediction models for reverse engineering"""
        # Clean data
        df_clean = self.df.dropna().reset_index(drop=True)
        self.df_clean = df_clean
        
        # Encode categorical variables
        categorical_columns = ['MODEL', 'Trim', 'Color', 'Drive Train', 'Location (State)']
        
        for col in categorical_columns:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
                self.label_encoders[col] = le
        
        # Features for prediction
        self.feature_columns = ['Price in USD', 'Trim_encoded']
        X = df_clean[self.feature_columns]
        
        # Target variables
        y_year = df_clean['Year']
        y_mileage = df_clean['Mileage Done']
        
        # Train models
        self.year_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.mileage_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        self.year_model.fit(X, y_year)
        self.mileage_model.fit(X, y_mileage)
    
    def setup_tco_calculators(self):
        """Setup TCO calculation parameters"""
        # Base TCO rates (can be refined with real data)
        self.tco_rates = {
            'maintenance_per_mile': 0.08,  # $0.08 per mile average
            'maintenance_base_per_year': 500,  # Base maintenance cost
            'insurance_base_per_year': 1200,  # Base insurance
            'insurance_per_1000_value': 10,  # $10 per $1000 of car value
            'fuel_per_mile': 0.15,  # Average fuel cost
            'depreciation_per_year': 0.12,  # 12% depreciation per year
        }
        
        # Trim-specific adjustments
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
            if trim in self.label_encoders['Trim'].classes_:
                trim_encoded = self.label_encoders['Trim'].transform([trim])[0]
            else:
                # Find closest trim or use default
                trim_encoded = 0  # Default to first trim
            
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
                'year': predicted_year,
                'mileage': predicted_mileage,
                'year_range': year_range,
                'mileage_range': mileage_range
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_tco(self, predicted_specs, budget, trim, annual_mileage=15000):
        """Calculate Total Cost of Ownership"""
        year = predicted_specs['year']
        mileage = predicted_specs['mileage']
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

def main():
    st.set_page_config(
    page_title="Car Judo - TCO Intelligence", 
    layout="wide",
    page_icon="🚗"
)
    
    st.title("🚗 Car Judo - What Can You REALLY Afford?")
    st.markdown("Stop guessing. Know exactly what your budget gets you - **including all hidden costs.**")
    
    # Initialize the system
    try:
        csv_path = "Ford_150.csv"
        car_judo = CarJudoTCOSystem(csv_path)
        st.success("✅ Car Judo system loaded successfully!")
    except FileNotFoundError:
        st.error(f"❌ Could not find file: {csv_path}")
        return
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        return
    
    # Main interface
    st.sidebar.markdown("### 🚗 Car Judo")
    st.sidebar.markdown("*Budget-First Intelligence*")
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Your Budget")
    
    # User inputs
    budget = st.sidebar.number_input("Budget ($)", min_value=5000, max_value=100000, value=25000, step=1000)
    
    # Budget validation
    if budget < 10000:
        st.sidebar.warning("⚠️ Low budget may limit options to older/high-mileage vehicles")
    elif budget > 50000:
        st.sidebar.success("✨ Great budget! You should find excellent options")
    
    # Get available trims
    available_trims = sorted(car_judo.df_clean['Trim'].unique())
    trim = st.sidebar.selectbox("Preferred Trim", available_trims)
    
    annual_mileage = st.sidebar.slider("Annual Mileage", 5000, 30000, 15000, 1000)
    
    # Analyze button
    if st.sidebar.button("🚗 ANALYZE", type="primary"):
        with st.spinner("Analyzing your options..."):
            # Get predictions
            predictions = car_judo.predict_car_specifications(budget, trim)
            
            if 'error' in predictions:
                st.error(f"Prediction error: {predictions['error']}")
                return
            
            # Calculate TCO
            tco_analysis = car_judo.calculate_tco(predictions, budget, trim, annual_mileage)
            
            # Display results
            st.success(f"BOOM! Your ${budget:,} budget gets you:")
            
            # Main results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Expected Specifications")
                st.write(f"**Trim:** {trim}")
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
            
            # What If Scenarios
            st.subheader("What If Scenarios")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"Add $5,000 to Budget (${budget+5000:,} total)"):
                    with st.spinner("Calculating upgraded options..."):
                        upgraded_predictions = car_judo.predict_car_specifications(budget + 5000, trim)
                        if 'error' not in upgraded_predictions:
                            st.success(f"🚀 For ${budget+5000:,}: {upgraded_predictions['year']} model with {upgraded_predictions['mileage']:,} miles")
                            st.write("That's **{upgraded_predictions['year'] - predictions['year']} years newer** and **{predictions['mileage'] - upgraded_predictions['mileage']:,} fewer miles**!")
            
            with col2:
                if st.button("Reduce Annual Mileage to 10,000"):
                    with st.spinner("Recalculating costs..."):
                        lower_mileage_tco = car_judo.calculate_tco(predictions, budget, trim, 10000)
                        annual_savings = tco_analysis['annual_tco'] - lower_mileage_tco['annual_tco']
                        st.success(f"💸 You'd save ${annual_savings:,.0f} per year (${annual_savings*5:,.0f} over 5 years)")
    
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
            <em>Budget-First Intelligence for Smart Car Buyers</em>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
