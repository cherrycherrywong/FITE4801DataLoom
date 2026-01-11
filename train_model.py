import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_brain():
    print("🧠 Initializing Credit Scoring Brain...")

    # 1. Create Synthetic Training Data
    # This data teaches the model what a "Good" vs "Bad" payer looks like.
    # It mirrors the structure used in the main.py (trips, spend, completeness).
    data = {
        'trips_count': [
            1, 2, 5, 20, 15, 8,   # User A, B, C...
            50, 0, 1, 3, 2, 0     # User G, H, I...
        ],
        'avg_spend': [
            200, 300, 1000, 5000, 3500, 1200, 
            10000, 0, 50, 400, 250, 0
        ],
        'completeness_score': [
            30, 45, 60, 99, 95, 80, 
            100, 0, 20, 50, 40, 0
        ],
        # 0 = High Risk (Poor), 1 = Low Risk (Good/Excellent)
        'is_good_payer': [
            0, 0, 0, 1, 1, 1, 
            1, 0, 0, 0, 0, 0
        ]
    }

    df = pd.DataFrame(data)
    
    # 2. Define Features (Inputs) and Target (Output)
    # These must match the inputs in the main.py (trips, spend, completeness)
    X = df[['trips_count', 'avg_spend', 'completeness_score']]
    y = df['is_good_payer']

    print(f"📊 Training on {len(df)} synthetic records...")

    # 3. Train the Model
    # n_estimators=100 means we use 100 "decision trees" to vote on the result
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 4. Save the Brain
    output_filename = 'credit_brain.pkl'
    joblib.dump(model, output_filename)
    print(f"✅ Success! Model saved as '{output_filename}'")
    print("🚀 You can now run 'uvicorn main:app --reload'")

if __name__ == "__main__":
    train_brain()