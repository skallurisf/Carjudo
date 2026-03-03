import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("car_data.csv")

df['Vehicle_Age'] = 2024 - df['Year']
df['Mileage_per_Year'] = df['Mileage'] / df['Vehicle_Age']

X = df[['Price','Mileage','MPG','Reliability_Score','Vehicle_Age','Mileage_per_Year']]
y = df['Annual_Repair_Cost']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor()
model.fit(X_train,y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test,preds))

df['Predicted_Repair_Cost'] = model.predict(X)

ownership_years = 5
fuel_price = 3.5
annual_miles = 12000

df['Fuel_Cost'] = (annual_miles / df['MPG']) * fuel_price * ownership_years
df['Maintenance_Total'] = df['Predicted_Repair_Cost'] * ownership_years
df['Resale_Value'] = df['Price'] * 0.5

df['TCO'] = df['Price'] + df['Maintenance_Total'] + df['Fuel_Cost'] - df['Resale_Value']

ranked = df.sort_values('TCO')

print("\nTop 5 Lowest TCO Vehicles:\n")
print(ranked[['Make','Model','Year','TCO']].head())

