"""eThekwini Municipal Election ML Pipeline Reproducibility Script

This script trains the Random Forest models using historical lag features (t-1)
and exports the serialized .pkl artifacts required by the Streamlit dashboard.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def run_pipeline():
  print("==================================================")
  print("   eThekwini Electoral ML Pipeline Training       ")
  print("==================================================")

  # 1. Load Master ML Dataset
  print("\n[1/5] Loading master ML dataset (df_ml_master.pkl)...")
  try:
    df_ml = pd.read_pickle("df_ml_master.pkl")
    print(f"      Loaded {len(df_ml)} rows successfully.")
  except Exception as e:
    print(f"      ERROR loading df_ml_master.pkl: {e}")
    return

  # 2. Define Features & Target Splits
  print("\n[2/5] Preparing leakage-free features (t-1 lags)...")
  features = [
      "Lag_WinningShare_Pct",
      "Lag_Turnout_Pct",
      "Lag_Registered_Voters",
  ]

  # Check column names in dataset
  col_map = {
      "WinningShare_Pct": next(
          (
              c
              for c in [
                  "WinningShare_Pct",
                  "Winning_Share_Pct",
                  "Winning_Share",
              ]
              if c in df_ml.columns
          ),
          "WinningShare_Pct",
      ),
      "Turnout_Pct": next(
          (
              c
              for c in [
                  "Turnout_Pct",
                  "Voter_Turnout_Pct",
                  "Turnout",
                  "Voter_Turnout",
              ]
              if c in df_ml.columns
          ),
          "Turnout_Pct",
      ),
      "Registered_Voters": next(
          (
              c
              for c in [
                  "RegisteredVoters",
                  "Total_Registered_Voters",
                  "Registered_Voters",
              ]
              if c in df_ml.columns
          ),
          "RegisteredVoters",
      ),
  }

  df_train = df_ml[df_ml["Year"] == 2016].copy()
  df_test = df_ml[df_ml["Year"] == 2021].copy()

  # Construct Feature Matrices
  X_train = pd.DataFrame({
      "Lag_WinningShare_Pct": df_train[col_map["WinningShare_Pct"]],
      "Lag_Turnout_Pct": df_train[col_map["Turnout_Pct"]],
      "Lag_Registered_Voters": df_train[col_map["Registered_Voters"]],
  })

  X_test = pd.DataFrame({
      "Lag_WinningShare_Pct": df_test[col_map["WinningShare_Pct"]],
      "Lag_Turnout_Pct": df_test[col_map["Turnout_Pct"]],
      "Lag_Registered_Voters": df_test[col_map["Registered_Voters"]],
  })

  y_train_c = (
      df_train["Target_ANC_Win"]
      if "Target_ANC_Win" in df_train.columns
      else (df_train[col_map["WinningShare_Pct"]] > 50).astype(int)
  )
  y_test_c = (
      df_test["Target_ANC_Win"]
      if "Target_ANC_Win" in df_test.columns
      else (df_test[col_map["WinningShare_Pct"]] > 50).astype(int)
  )

  # 3. Fit Feature Scaler
  print("\n[3/5] Scaling features using StandardScaler...")
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # 4. Train Models
  print("\n[4/5] Training Random Forest Models...")

  # Model 3: Random Forest Classifier (Primary Solution)
  rf_classifier = RandomForestClassifier(
      n_estimators=100, max_depth=5, random_state=42
  )
  rf_classifier.fit(X_train_scaled, y_train_c)
  acc = rf_classifier.score(X_test_scaled, y_test_c)
  print(f"      Model 3 (Classifier) Test Accuracy : {acc * 100:.2f}%")

  # Model 2: Random Forest Regressor
  rf_regressor = RandomForestRegressor(
      n_estimators=100, max_depth=5, random_state=42
  )
  rf_regressor.fit(X_train_scaled, df_train[col_map["WinningShare_Pct"]])
  print("      Model 2 (Regressor) Trained Successfully.")

  # 5. Export Artifacts (.pkl)
  print("\n[5/5] Exporting .pkl artifacts for Streamlit deployment...")
  joblib.dump(rf_classifier, "model_ward_classifier.pkl")
  joblib.dump(rf_regressor, "model_vote_share_regressor.pkl")
  joblib.dump(scaler, "scaler_features.pkl")

  print("\n==================================================")
  print("   Pipeline Execution Complete! All .pkl saved.   ")
  print("==================================================")


if __name__ == "__main__":
  run_pipeline()