import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def to_int_year(x):
    try:
        s = str(x)
        m = re.search(r'(\d{4})', s)
        return int(m.group(1)) if m else np.nan
    except:
        return np.nan

def to_int_minutes(x):
    try:
        s = str(x)
        m = re.search(r'(\d+)', s)
        return int(m.group(1)) if m else np.nan
    except:
        return np.nan

def to_int_votes(x):
    try:
        if pd.isna(x): return np.nan
        s = str(x).replace(',', '').strip()
        return int(s) if s.isdigit() else np.nan
    except:
        return np.nan

df = pd.read_csv("movies.csv", encoding='latin1')

df = df.rename(columns=lambda c: c.strip())

expected_cols = ['Year','Duration','Genre','Votes','Director','Actor 1','Actor 2','Actor 3','Rating']
available = [c for c in expected_cols if c in df.columns]

df = df[available].copy()

if 'Rating' not in df.columns:
    raise SystemExit("No 'Rating' column found. Check your CSV.")

df['Year'] = df['Year'].apply(to_int_year)
df['Duration'] = df['Duration'].apply(to_int_minutes)
df['Votes'] = df['Votes'].apply(to_int_votes)

df['Genre'] = df['Genre'].fillna('Unknown').astype(str).apply(lambda s: s.split(',')[0].strip())

df = df.dropna(subset=['Rating']).reset_index(drop=True)

num_cols = []
if 'Year' in df.columns: num_cols.append('Year')
if 'Duration' in df.columns: num_cols.append('Duration')
if 'Votes' in df.columns: num_cols.append('Votes')

cat_cols = [c for c in ['Genre','Director','Actor 1','Actor 2','Actor 3'] if c in df.columns]

for nc in num_cols:
    df[nc] = df[nc].fillna(df[nc].median())

for cc in cat_cols:
    df[cc] = df[cc].fillna('Unknown').astype(str)

encoders = {}
for cc in cat_cols:
    le = LabelEncoder()
    df[cc] = le.fit_transform(df[cc])
    encoders[cc] = le

X = df[num_cols + cat_cols]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Model Performance")
print(f"R2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

feat_names = num_cols + cat_cols
importances = model.feature_importances_
fi = pd.Series(importances, index=feat_names).sort_values(ascending=True)

plt.figure(figsize=(8,6))
fi.plot.barh()
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()

joblib.dump(model, "movie_rating_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("Saved movie_rating_model.pkl, scaler.pkl, label_encoders.pkl and plots.")
