# model_training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import pickle

# Load data
df = pd.read_csv('forestfires.csv')

# Create labels
df['Classes'] = df['area'].apply(lambda x: 'LOW RISK' if x <= 5 else 'HIGH RISK')
df['SpreadTime'] = df['area'] / 30

features = ['temp', 'RH', 'wind', 'rain']
X = df[features]
y_fire_size = df['Classes']
y_time = df['SpreadTime']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_fire_size, test_size=0.2, random_state=42)

# GridSearchCV for XGBoost Classifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

grid_clf = GridSearchCV(xgb_clf, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

print("🔥 Best Classifier Parameters:", grid_clf.best_params_)

# XGBoost Regressor for Spread Time
xgb_reg = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1)
xgb_reg.fit(X_scaled, y_time)

# Save models
pickle.dump(grid_clf.best_estimator_, open('fire_size_model.pkl', 'wb'))
pickle.dump(xgb_reg, open('spread_time_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("✅ Models and scaler saved.")
