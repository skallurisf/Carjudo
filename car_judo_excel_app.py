import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MultiVehicleCarJudo:
    def __init__(self, excel_file):
        self.excel_file = excel_file
        self.all_data = {}
        self.models = {}
        self.label_encoders = {}
        
        # Load all vehicle data
        self.load_all_vehicles()
        
        # Setup TCO calculator
        self.setup_tco_calculators()
    
    def load_all_vehicles(self):
        """Load all vehicle types from Excel tabs"""
        try:
            xls = pd.ExcelFile(self.excel_file)
            vehicle_types = xls.sheet_names
            
            for vehicle in vehicle_types:
                df = pd.read_excel(self.excel_file, sheet_name=vehicle)
                self.all_data[vehicle] = df
                self.train_vehicle_model(df, vehicle)
                
        except Exception as e:
            st.error(f"❌ Error loading Excel file: {str(e)}")
            st.stop()
    
    def train_vehicle_model(self, df, vehicle_type):
        """Train model for specific vehicle type"""
        try:
            # Clean data with enhanced debugging (silent mode)
            df_clean = self.clean_vehicle_data(df, vehicle_type)
            
            if df_clean is None:
                # Skip silently - don't show warnings for bad data
                return
            
            # Encode categorical variables
            le_trim = LabelEncoder()
            if 'Trim' in df_clean.columns:
                df_clean['Trim_encoded'] = le_trim.fit_transform(df_clean['Trim'].astype(str))
            
            # Store label encoder
            self.label_encoders[vehicle_type] = le_trim
            
            # Prepare features and targets
            features = ['Price in USD', 'Trim_encoded']
            X = df_clean[features]
            y_year = df_clean['Year']
            y_mileage = df_clean['Mileage Done']
            
            # Train models
            year_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            mileage_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            year_model.fit(X, y_year)
            mileage_model.fit(X, y_mileage)
            
            # Store models
            self.models[vehicle_type] = {
                'year_model': year_model,
                'mileage_model': mileage_model,
                'data': df_clean,
                'feature_columns': features
            }
            
        except Exception as e:
            st.error(f"❌ Error training model for {vehicle_type}: {str(e)}")
    
    def clean_vehicle_data(self, df, vehicle_type):
        """Clean and preprocess vehicle data"""
        df_clean = df.copy()
        
        try:
            # Flexible column name matching
            column_mappings = {
                'Price in USD': ['Price in USD', 'Price', 'price', 'Price ($)', 'Cost', 'Amount'],
                'Year': ['Year', 'year', 'Model Year', 'Model Year', 'YEAR'],
                'Mileage Done': ['Mileage Done', 'Mileage', 'miles', 'Odometer', 'Mileage (mi)', 'Miles'],
                'Trim': ['Trim', 'trim', 'TRIM', 'Model', 'Version', 'Package']
            }
            
            # Find actual column names
            actual_columns = {}
            for standard_name, possible_names in column_mappings.items():
                found = False
                for possible_name in possible_names:
                    if possible_name in df_clean.columns:
                        actual_columns[standard_name] = possible_name
                        found = True
                        break
                if not found:
                    return None
            
            # Fix data types using actual column names
            if 'Mileage Done' in actual_columns:
                mileage_col = actual_columns['Mileage Done']
                df_clean[mileage_col] = df_clean[mileage_col].astype(str).str.replace(',', '').str.replace(' ', '').str.replace('$', '').str.replace('mi', '').str.replace('miles', '')
                df_clean[mileage_col] = pd.to_numeric(df_clean[mileage_col], errors='coerce')
                df_clean = df_clean.rename(columns={mileage_col: 'Mileage Done'})
            
            if 'Price in USD' in actual_columns:
                price_col = actual_columns['Price in USD']
                df_clean[price_col] = df_clean[price_col].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                df_clean[price_col] = pd.to_numeric(df_clean[price_col], errors='coerce')
                df_clean = df_clean.rename(columns={price_col: 'Price in USD'})
            
            if 'Year' in actual_columns:
                year_col = actual_columns['Year']
                df_clean[year_col] = pd.to_numeric(df_clean[year_col], errors='coerce')
                df_clean = df_clean.rename(columns={year_col: 'Year'})
            
            if 'Trim' in actual_columns:
                trim_col = actual_columns['Trim']
                df_clean[trim_col] = df_clean[trim_col].astype(str).fillna('Unknown')
                df_clean = df_clean.rename(columns={trim_col: 'Trim'})
            
            # Keep only essential columns
            essential_cols = ['Price in USD', 'Year', 'Mileage Done', 'Trim']
            df_clean = df_clean[essential_cols]
            
            # Remove missing values
            df_clean = df_clean.dropna()
            
            # Remove outliers using IQR
            for col in ['Price in USD', 'Mileage Done']:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            # Reset index
            df_clean = df_clean.reset_index(drop=True)
            
            # Check if we have enough data
            if len(df_clean) < 10:
                return None
            
            return df_clean
            
        except Exception as e:
            return None
    
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
    
    def predict_for_budget(self, budget, vehicle_type, trim=None):
        """Predict specifications for given budget and vehicle"""
        try:
            if vehicle_type not in self.models:
                error_msg = f'Vehicle {vehicle_type} not available'
                return {'error': error_msg}
            
            model = self.models[vehicle_type]
            le_trim = self.label_encoders[vehicle_type]
            
            # Handle trim selection
            if trim and trim in le_trim.classes_:
                trim_encoded = le_trim.transform([trim])[0]
            else:
                # Use most common trim
                most_common_trim = model['data']['Trim'].mode()[0]
                trim_encoded = le_trim.transform([most_common_trim])[0]
                trim = most_common_trim
            
            # Prepare input
            input_data = np.array([[budget, trim_encoded]])
            
            # Predict
            predicted_year = model['year_model'].predict(input_data)[0]
            predicted_mileage = model['mileage_model'].predict(input_data)[0]
            
            # Round to reasonable values
            predicted_year = int(round(predicted_year))
            predicted_mileage = int(round(predicted_mileage))
            
            # Calculate confidence ranges
            year_range = (predicted_year - 1, predicted_year + 1)
            mileage_range = (predicted_mileage - 10000, predicted_mileage + 10000)
            
            result = {
                'vehicle': vehicle_type,
                'trim': trim,
                'year': predicted_year,
                'mileage': predicted_mileage,
                'year_range': year_range,
                'mileage_range': mileage_range
            }
            
            return result
            
        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'
            return {'error': error_msg}
    
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
        page_title="Car Judo - Multi-Vehicle Intelligence", 
        layout="wide",
        page_icon="🚗"
    )
    
    st.title("🚗 Car Judo - Multi-Vehicle Budget Intelligence")
    st.markdown("Discover what your budget gets you across **multiple vehicle types** - including all hidden costs!")
    
    # Initialize the system
    try:
        excel_file = "Car_Data.xlsx"  # Your Excel file
        car_judo = MultiVehicleCarJudo(excel_file)
        
        # Quick stats
        total_vehicles = len(car_judo.all_data)
        total_listings = sum(len(car_judo.models[v]['data']) for v in car_judo.models if v in car_judo.models)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚗 Vehicle Types", total_vehicles)
        with col2:
            st.metric("📊 Total Listings", total_listings)
        with col3:
            st.metric("🤖 AI Models", len(car_judo.models))
        
        st.markdown("---")
        
    except FileNotFoundError:
        st.error(f"❌ Could not find Excel file: {excel_file}")
        st.info("Please make sure your Excel file is in the same directory as this app.")
        return
    except Exception as e:
        st.error(f"❌ Error loading system: {str(e)}")
        return
    
    # Main interface
    st.sidebar.markdown("### 🚗 Car Judo")
    st.sidebar.markdown("*Multi-Vehicle Intelligence*")
    st.sidebar.markdown("---")
    st.sidebar.header("💰 Your Budget")
    
    # User inputs
    available_vehicles = list(car_judo.all_data.keys())
    if not available_vehicles:
        st.error("No vehicle data available")
        return
    
    # Vehicle selection with data counts
    st.sidebar.subheader("🚗 Choose Your Vehicle")
    vehicle_options = []
    for vehicle in available_vehicles:
        if vehicle in car_judo.models:
            count = len(car_judo.models[vehicle]['data'])
            vehicle_options.append(f"{vehicle} ({count} listings)")
        else:
            vehicle_options.append(f"{vehicle} (insufficient data)")
    
    selected_option = st.sidebar.selectbox("Vehicle Type", vehicle_options)
    vehicle_type = selected_option.split(" (")[0]  # Extract vehicle name
    
    # Get available trims for selected vehicle
    if vehicle_type in car_judo.models:
        available_trims = list(car_judo.models[vehicle_type]['data']['Trim'].unique())
        trim = st.sidebar.selectbox("Preferred Trim (Optional)", ['Recommended'] + list(available_trims))
    else:
        trim = 'Recommended'
    
    budget = st.sidebar.number_input("Budget ($)", min_value=5000, max_value=100000, value=25000, step=1000)
    annual_mileage = st.sidebar.slider("Annual Mileage", 5000, 30000, 15000, 1000)
    
    # Budget validation
    if budget < 10000:
        st.sidebar.warning("⚠️ Low budget may limit options to older/high-mileage vehicles")
    elif budget > 50000:
        st.sidebar.success("✨ Great budget! You should find excellent options")
    
    # Analyze button
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        analyze_button = st.button("🚗 ANALYZE", type="primary")
    
    with col2:
        compare_button = st.button("🔄 COMPARE")
    
    # Store comparison results
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = []
    
    if analyze_button:
        with st.spinner("Analyzing your options across all vehicles..."):
            # Get predictions
            trim_to_use = None if trim == 'Recommended' else trim
            predictions = car_judo.predict_for_budget(budget, vehicle_type, trim_to_use)
            
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
            
            # Add to comparison button
            if st.button(f"📊 Add {vehicle_type} to Comparison"):
                comparison_data = {
                    'vehicle': predictions['vehicle'],
                    'trim': predictions['trim'],
                    'year': predictions['year'],
                    'mileage': predictions['mileage'],
                    'budget': budget,
                    'total_5yr_tco': tco_analysis['total_5yr_tco']
                }
                st.session_state.comparison_results.append(comparison_data)
                st.success(f"✅ Added {vehicle_type} to comparison!")
    
    # Handle compare button
    if compare_button:
        if len(st.session_state.comparison_results) < 2:
            st.warning("📊 Add at least 2 vehicles to compare")
        else:
            st.subheader("🔄 Vehicle Comparison")
            
            # Create comparison table
            comparison_df = pd.DataFrame(st.session_state.comparison_results)
            
            # Display comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Vehicles Compared", len(comparison_df))
            
            with col2:
                avg_tco = comparison_df['total_5yr_tco'].mean()
                st.metric("Avg 5-Year TCO", f"${avg_tco:,.0f}")
            
            with col3:
                best_value = comparison_df.loc[comparison_df['total_5yr_tco'].idxmin()]
                st.metric("Best Value", f"{best_value['vehicle']}")
            
            # Detailed comparison table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Clear comparison button
            if st.button("🗑️ Clear Comparison"):
                st.session_state.comparison_results = []
                st.experimental_rerun()
    
    # Show current comparison count
    if st.session_state.comparison_results:
        st.sidebar.info(f"📊 Comparison: {len(st.session_state.comparison_results)} vehicles")
    
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
            <em>Multi-Vehicle Intelligence for Smart Car Buyers</em>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
