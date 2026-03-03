import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CarModelEvaluator:
    def __init__(self, csv_path):
        self.df = pd.read_csv("Ford_150.csv")
        self.original_shape = self.df.shape
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def data_quality_report(self):
        """Generate comprehensive data quality report"""
        print("="*60)
        print("📊 DATA QUALITY REPORT")
        print("="*60)
        
        print(f"Original dataset shape: {self.original_shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Fix data types first
        print("\n🔧 FIXING DATA TYPES:")
        if 'Mileage Done' in self.df.columns:
            # Remove commas and convert to numeric
            self.df['Mileage Done'] = self.df['Mileage Done'].astype(str).str.replace(',', '').str.replace(' ', '')
            self.df['Mileage Done'] = pd.to_numeric(self.df['Mileage Done'], errors='coerce')
            print(f"  Converted 'Mileage Done' to numeric")
        
        if 'Price in USD' in self.df.columns:
            # Remove $, commas and convert to numeric
            self.df['Price in USD'] = self.df['Price in USD'].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
            self.df['Price in USD'] = pd.to_numeric(self.df['Price in USD'], errors='coerce')
            print(f"  Converted 'Price in USD' to numeric")
        
        # Missing values analysis
        print("\n🔍 MISSING VALUES ANALYSIS:")
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        for col in self.df.columns:
            if missing_data[col] > 0:
                print(f"  {col}: {missing_data[col]} missing ({missing_percentage[col]:.1f}%)")
        
        if missing_data.sum() == 0:
            print("  ✅ No missing values found!")
        
        # Data types analysis
        print("\n📋 DATA TYPES AFTER CLEANING:")
        print(self.df.dtypes)
        
        # Unique values for categorical columns
        print("\n🏷️ CATEGORICAL COLUMNS ANALYSIS:")
        categorical_columns = ['MODEL', 'Trim', 'Color', 'Drive Train', 'Location (State)']
        
        for col in categorical_columns:
            if col in self.df.columns:
                unique_count = self.df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {list(self.df[col].unique())}")
        
        # Numerical columns statistics
        print("\n📈 NUMERICAL COLUMNS STATISTICS:")
        numerical_cols = ['Year', 'Mileage Done', 'Price in USD']
        print(self.df[numerical_cols].describe())
        
        # Outlier detection
        print("\n⚠️ OUTLIER DETECTION:")
        numerical_cols = ['Year', 'Mileage Done', 'Price in USD']
        for col in numerical_cols:
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                print(f"  {col}: {len(outliers)} outliers detected ({len(outliers)/len(self.df)*100:.1f}%)")
            elif col in self.df.columns:
                print(f"  {col}: Skipped - not numeric (type: {self.df[col].dtype})")
        
        return self.df
    
    def clean_data(self, remove_outliers=True, handle_missing='drop'):
        """Clean the dataset"""
        print("\n🧹 DATA CLEANING PROCESS:")
        print(f"Starting with {len(self.df)} rows")
        
        df_clean = self.df.copy()
        
        # Handle missing values
        if handle_missing == 'drop':
            df_clean = df_clean.dropna()
            print(f"  Dropped rows with missing values: {len(self.df) - len(df_clean)}")
        elif handle_missing == 'fill':
            # Fill numerical missing values with median
            numerical_cols = ['Year', 'Mileage Done', 'Price in USD']
            for col in numerical_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
            # Fill categorical missing values with mode
            categorical_cols = ['MODEL', 'Trim', 'Color', 'Drive Train', 'Location (State)']
            for col in categorical_cols:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        
        # Remove outliers
        if remove_outliers:
            numerical_cols = ['Year', 'Mileage Done', 'Price in USD']
            outlier_mask = pd.Series([True] * len(df_clean))
            outlier_mask.index = df_clean.index  # Align the index
            
            for col in numerical_cols:
                if col in df_clean.columns:
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    col_outlier_mask = (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                    outlier_mask = outlier_mask & col_outlier_mask
            
            outliers_removed = len(df_clean) - outlier_mask.sum()
            df_clean = df_clean[outlier_mask]
            print(f"  Removed outliers: {outliers_removed}")
        
        # Remove duplicates
        duplicates_removed = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        print(f"  Removed duplicates: {duplicates_removed}")
        
        # Data validation
        # Ensure year is reasonable
        if 'Year' in df_clean.columns:
            df_clean = df_clean[(df_clean['Year'] >= 1990) & (df_clean['Year'] <= 2030)]
        
        # Ensure mileage is non-negative
        if 'Mileage Done' in df_clean.columns:
            df_clean = df_clean[df_clean['Mileage Done'] >= 0]
        
        # Ensure price is positive
        if 'Price in USD' in df_clean.columns:
            df_clean = df_clean[df_clean['Price in USD'] > 0]
        
        print(f"Final dataset: {len(df_clean)} rows ({len(df_clean)/len(self.df)*100:.1f}% retained)")
        self.df = df_clean
        return df_clean
    
    def train_and_evaluate(self):
        """Train model and comprehensive evaluation"""
        print("\n🤖 MODEL TRAINING AND EVALUATION:")
        print("="*60)
        
        # Encode categorical variables
        categorical_columns = ['MODEL', 'Trim', 'Color', 'Drive Train', 'Location (State)']
        
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
        
        # Prepare features
        self.feature_columns = ['Year', 'Mileage Done'] + [col + '_encoded' for col in categorical_columns if col in self.df.columns]
        X = self.df[self.feature_columns]
        y = self.df['Price in USD']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        print("\n📊 PERFORMANCE METRICS:")
        print("-" * 40)
        
        # Training set performance
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_train_pred)
        
        print("TRAINING SET:")
        print(f"  MAE: ${train_mae:,.2f}")
        print(f"  MSE: ${train_mse:,.2f}")
        print(f"  RMSE: ${train_rmse:,.2f}")
        print(f"  R²: {train_r2:.4f}")
        
        # Test set performance
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("\nTEST SET:")
        print(f"  MAE: ${test_mae:,.2f}")
        print(f"  MSE: ${test_mse:,.2f}")
        print(f"  RMSE: ${test_rmse:,.2f}")
        print(f"  R²: {test_r2:.4f}")
        
        # Cross-validation
        print("\n🔄 CROSS-VALIDATION:")
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"  CV MAE: ${cv_mae:,.2f} (±${cv_std:,.2f})")
        
        # Error analysis
        print("\n📉 ERROR ANALYSIS:")
        errors = np.abs(self.y_test - y_test_pred)
        print(f"  Mean error: ${errors.mean():,.2f}")
        print(f"  Median error: ${np.median(errors):,.2f}")
        print(f"  Max error: ${errors.max():,.2f}")
        print(f"  Std error: ${errors.std():,.2f}")
        
        # Percentage error analysis
        percentage_errors = (errors / self.y_test) * 100
        print(f"  Mean % error: {percentage_errors.mean():.1f}%")
        print(f"  Median % error: {np.median(percentage_errors):.1f}%")
        
        # Sensitivity analysis (feature importance)
        print("\n🎯 FEATURE IMPORTANCE (SENSITIVITY):")
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'feature_importance': feature_importance
        }
    
    def generate_predictions_analysis(self):
        """Analyze predictions vs actual values"""
        print("\n🔍 PREDICTIONS ANALYSIS:")
        print("-" * 40)
        
        y_pred = self.model.predict(self.X_test)
        
        # Create prediction intervals
        residuals = self.y_test - y_pred
        std_residual = np.std(residuals)
        
        print(f"Prediction intervals (95% confidence): ±${1.96*std_residual:,.2f}")
        
        # Accuracy within different thresholds
        thresholds = [1000, 2500, 5000, 10000]
        for threshold in thresholds:
            accuracy = np.mean(np.abs(residuals) <= threshold) * 100
            print(f"Accuracy within ${threshold:,}: {accuracy:.1f}%")
        
        return y_pred, residuals

def main():
    # Update this path to your CSV file
    csv_path = "your_car_data.csv"
    
    try:
        # Initialize evaluator
        evaluator = CarModelEvaluator(csv_path)
        
        # Data quality report
        evaluator.data_quality_report()
        
        # Clean data
        evaluator.clean_data(remove_outliers=True, handle_missing='drop')
        
        # Train and evaluate
        results = evaluator.train_and_evaluate()
        
        # Generate predictions analysis
        y_pred, residuals = evaluator.generate_predictions_analysis()
        
        print("\n" + "="*60)
        print("✅ EVALUATION COMPLETE!")
        print("="*60)
        print(f"Model Performance Summary:")
        print(f"  • Mean Absolute Error: ${results['test_mae']:,.2f}")
        print(f"  • Root Mean Square Error: ${results['test_rmse']:,.2f}")
        print(f"  • R² Score: {results['test_r2']:.4f}")
        print(f"  • Cross-validated MAE: ${results['cv_mae']:,.2f}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find file '{csv_path}'")
        print("Please update the csv_path variable to point to your actual CSV file.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
