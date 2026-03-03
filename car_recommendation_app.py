import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class CarRecommendationSystem:
    def __init__(self, csv_path):
        self.df = pd.read_csv("Ford_150.csv")
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        
        # Fix data types first
        self._fix_data_types()
        self.setup_model()
    
    def _fix_data_types(self):
        """Fix string data types to numeric"""
        if 'Mileage Done' in self.df.columns:
            # Remove commas and convert to numeric
            self.df['Mileage Done'] = self.df['Mileage Done'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['Mileage Done'] = pd.to_numeric(self.df['Mileage Done'], errors='coerce')
        
        if 'Price in USD' in self.df.columns:
            # Remove $, commas and convert to numeric
            self.df['Price in USD'] = self.df['Price in USD'].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
            self.df['Price in USD'] = pd.to_numeric(self.df['Price in USD'], errors='coerce')
    
    def setup_model(self):
        """Setup the ML model and encoders"""
        # Preprocess data
        df_clean = self.df.dropna().reset_index(drop=True)  # Reset index after dropping rows
        
        # Store the cleaned dataframe for predictions
        self.df_clean = df_clean
        
        # Encode categorical variables
        categorical_columns = ['MODEL', 'Trim', 'Color', 'Drive Train', 'Location (State)']
        
        for col in categorical_columns:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col + '_encoded'] = le.fit_transform(df_clean[col])
                self.label_encoders[col] = le
        
        # Prepare features
        self.feature_columns = ['Year', 'Mileage Done'] + [col + '_encoded' for col in categorical_columns if col in df_clean.columns]
        X = df_clean[self.feature_columns]
        y = df_clean['Price in USD']
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X, y)
        
        # Add predicted prices to CLEANED dataframe
        self.df_clean['Predicted_Price'] = self.model.predict(df_clean[self.feature_columns])
        self.df_clean['Value_Score'] = (self.df_clean['Predicted_Price'] - self.df_clean['Price in USD']) / self.df_clean['Predicted_Price'] * 100
    
    def get_recommendations(self, preferences):
        """Get car recommendations based on user preferences"""
        filtered_df = self.df_clean.copy()  # Use cleaned dataframe with predictions
        
        # Apply filters
        if preferences.get('car_type') and preferences['car_type'].strip():
            filtered_df = filtered_df[filtered_df['MODEL'].str.contains(preferences['car_type'], case=False, na=False)]
        
        if preferences.get('budget_max'):
            filtered_df = filtered_df[filtered_df['Price in USD'] <= preferences['budget_max']]
        
        if preferences.get('budget_min'):
            filtered_df = filtered_df[filtered_df['Price in USD'] >= preferences['budget_min']]
        
        if preferences.get('year_min'):
            filtered_df = filtered_df[filtered_df['Year'] >= preferences['year_min']]
        
        if preferences.get('mileage_max'):
            filtered_df = filtered_df[filtered_df['Mileage Done'] <= preferences['mileage_max']]
        
        if preferences.get('color') and preferences['color'].strip():
            filtered_df = filtered_df[filtered_df['Color'].str.contains(preferences['color'], case=False, na=False)]
        
        if preferences.get('location') and preferences['location'].strip():
            filtered_df = filtered_df[filtered_df['Location (State)'].str.contains(preferences['location'], case=False, na=False)]
        
        # Sort by value score (best deals first)
        recommendations = filtered_df.sort_values('Value_Score', ascending=False)
        
        return recommendations.head(10)  # Return top 10 recommendations

def main():
    st.set_page_config(page_title="Car Price Recommender", layout="wide")
    
    st.title("🚗 Smart Car Price Recommender")
    st.markdown("Find the best value cars based on your preferences and our AI price predictions!")
    
    # Initialize the system (you'll need to update this path)
    try:
        csv_path = "Ford_150.csv"  # Using your Ford F-150 CSV file
        recommender = CarRecommendationSystem(csv_path)
        st.success("✅ Car data loaded successfully!")
    except FileNotFoundError:
        st.error(f"❌ Could not find file: {csv_path}")
        st.info("Please update the `csv_path` variable in the code to point to your actual CSV file.")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Sidebar for user preferences
    st.sidebar.header("🎯 Your Preferences")
    
    preferences = {}
    
    # Car Type
    preferences['car_type'] = st.sidebar.text_input("Car Type (e.g., F-150, Camry, etc.)", "")
    
    # Budget Range
    st.sidebar.subheader("💰 Budget Range (USD)")
    preferences['budget_min'] = st.sidebar.number_input("Minimum Budget", min_value=0, value=0, step=1000)
    preferences['budget_max'] = st.sidebar.number_input("Maximum Budget", min_value=0, value=100000, step=1000)
    
    # Car Specifications
    st.sidebar.subheader("🔧 Car Specifications")
    preferences['year_min'] = st.sidebar.number_input("Minimum Year", min_value=2000, value=2020, step=1)
    preferences['mileage_max'] = st.sidebar.number_input("Maximum Mileage", min_value=0, value=50000, step=1000)
    
    # Additional Preferences
    st.sidebar.subheader("🎨 Additional Preferences")
    preferences['color'] = st.sidebar.text_input("Preferred Color", "")
    preferences['location'] = st.sidebar.text_input("Preferred State", "")
    
    # Purchase Type
    purchase_type = st.sidebar.selectbox("💳 Purchase Type", ["Buy", "Lease", "Either"])
    
    # Get recommendations button
    if st.sidebar.button("🔍 Find Best Deals", type="primary"):
        with st.spinner("🤖 AI is analyzing the best deals for you..."):
            recommendations = recommender.get_recommendations(preferences)
        
        if len(recommendations) == 0:
            st.warning("⚠️ No cars found matching your criteria. Try adjusting your filters.")
        else:
            st.success(f"🎉 Found {len(recommendations)} great deals for you!")
            
            # Display recommendations
            for idx, (_, car) in enumerate(recommendations.iterrows(), 1):
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.subheader(f"🏆 Deal #{idx}: {car.get('MODEL', 'Unknown')}")
                        st.write(f"**Trim:** {car.get('Trim', 'N/A')}")
                        st.write(f"**Year:** {car.get('Year', 'N/A')}")
                        st.write(f"**Mileage:** {car.get('Mileage Done', 'N/A'):,} miles")
                        st.write(f"**Color:** {car.get('Color', 'N/A')}")
                        st.write(f"**Location:** {car.get('Location (State)', 'N/A')}")
                    
                    with col2:
                        st.metric("Actual Price", f"${car.get('Price in USD', 0):,.2f}")
                        st.metric("AI Predicted Value", f"${car.get('Predicted_Price', 0):,.2f}")
                        
                        value_score = car.get('Value_Score', 0)
                        if value_score > 0:
                            savings = abs(value_score * car.get('Predicted_Price', 0) / 100)
                            st.success(f"💰 You Save: ${savings:,.2f} ({value_score:.1f}% below market)")
                        else:
                            overprice = abs(value_score * car.get('Predicted_Price', 0) / 100)
                            st.warning(f"⚠️ Above Market: ${overprice:,.2f} ({value_score:.1f}% above market)")
                    
                    with col3:
                        if value_score > 5:
                            st.markdown("🟢 **Excellent Deal**")
                        elif value_score > 0:
                            st.markdown("🟡 **Good Deal**")
                        else:
                            st.markdown("🔴 **Overpriced**")
                    
                    st.divider()
    
    # Show market insights
    if st.checkbox("📊 Show Market Insights"):
        st.subheader("Market Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Average Prices by Model:**")
            if 'MODEL' in recommender.df_clean.columns:
                model_prices = recommender.df_clean.groupby('MODEL')['Price in USD'].agg(['mean', 'count']).round(2)
                st.dataframe(model_prices.sort_values('mean', ascending=False))
        
        with col2:
            st.write("**Price Distribution:**")
            st.write(f"Average Price: ${recommender.df_clean['Price in USD'].mean():,.2f}")
            st.write(f"Median Price: ${recommender.df_clean['Price in USD'].median():,.2f}")
            st.write(f"Price Range: ${recommender.df_clean['Price in USD'].min():,.2f} - ${recommender.df_clean['Price in USD'].max():,.2f}")

if __name__ == "__main__":
    main()
