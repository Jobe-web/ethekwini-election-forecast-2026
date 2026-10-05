import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# PAGE CONFIGURATION & THEME SETTINGS
# ==============================================================================
st.set_page_config(
    page_title="eThekwini Election Analytics & 2026 Forecast",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-header { font-size: 28px; font-weight: bold; color: #1E3A8A; }
    .sub-header { font-size: 18px; color: #4B5563; }
    .metric-box { background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #1E3A8A; }
    .warning-box { background-color: #FEF3C7; padding: 15px; border-radius: 8px; border-left: 5px solid #F59E0B; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# HELPER FUNCTIONS & COLUMN RESOLUTION
# ==============================================================================
def find_column(df, possible_names, default=None):
  """Find the first matching column name from a list of possibilities."""
  if df is None:
    return default
  for col in possible_names:
    if col in df.columns:
      return col
  return default


# ==============================================================================
# DATA & ARTIFACT LOADING (WITH CACHING)
# ==============================================================================
@st.cache_resource
def load_ml_artifacts():
  """Load trained machine learning models and feature scalers."""
  try:
    classifier = joblib.load("model_ward_classifier.pkl")
    regressor = joblib.load("model_vote_share_regressor.pkl")
    scaler = joblib.load("scaler_features.pkl")
    return classifier, regressor, scaler
  except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    return None, None, None


@st.cache_data
def load_datasets():
  """Load cleaned master datasets for EDA and ML predictions."""
  try:
    df_ml = pd.read_pickle("df_ml_master.pkl")
    df_eda = pd.read_pickle("df_eda_master.pkl")
    return df_ml, df_eda
  except Exception as e:
    st.error(f"Error loading master datasets: {e}")
    return None, None


rf_classifier, rf_regressor, feature_scaler = load_ml_artifacts()
df_ml_master, df_eda_master = load_datasets()

# ==============================================================================
# SIDEBAR CONTROL PANEL
# ==============================================================================
st.sidebar.title("🗳️ Navigation & Parameters")
page_selection = st.sidebar.radio(
    "Select Section:",
    [
        "1. Metro Macro Findings (2011–2021)",
        "2. 04 Nov 2026 Model Forecasts",
        "3. Model Diagnostics & Performance Evidence",
        "4. Limitations & Structural Risk",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Scenario Simulation (2026)")
turnout_adjustment = st.sidebar.slider(
    "Global Turnout Shift Factor (%)",
    min_value=-15.0,
    max_value=15.0,
    value=0.0,
    step=1.0,
    help="Simulates macro voter mobilization or further apathy relative to 2021.",
)


# ==============================================================================
# SECTION 1: METRO-LEVEL MACRO FINDINGS (HISTORICAL 2011-2021)
# ==============================================================================
if page_selection == "1. Metro Macro Findings (2011–2021)":
  st.markdown(
      '<div class="main-header">eThekwini Metro Electoral Findings'
      " (2011–2021)</div>",
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Historical evidence of voter participation,'
      " party vote shares, and competitive dynamics.</div>",
      unsafe_allow_html=True,
  )
  st.markdown("---")

  col1, col2, col3, col4 = st.columns(4)
  col1.metric("2011 Mean Turnout", "61.8%", "Baseline")
  col2.metric("2016 Mean Turnout", "61.2%", "-0.6%")
  col3.metric("2021 Mean Turnout", "45.1%", "-16.1% (Collapse)")
  col4.metric("Total Wards Evaluated", "141 Wards", "MDB Bound")

  st.markdown("### 📊 Macro Electoral Trends")
  tab1, tab2 = st.tabs(
      ["Major Party Vote Trajectory", "Voter Turnout Collapse"]
  )

  with tab1:
    party_col = find_column(
        df_eda_master,
        [
            "WinningParty",
            "Winning_Party",
            "Party",
            "PartyName",
            "Party_Name",
            "Leading_Party",
            "Winner",
            "Party_Winner",
        ],
    )
    votes_col = find_column(
        df_eda_master,
        [
            "WinningPartyVotes",
            "TotalWardVotes",
            "Total_Valid_Votes",
            "Valid_Votes",
            "Total_Votes",
            "Votes",
            "ValidVotes",
            "TotalValidVotes",
            "Votes_Cast",
        ],
    )

    if party_col and votes_col:
      party_votes = df_eda_master.pivot_table(
          index="Year",
          columns=party_col,
          values=votes_col,
          aggfunc="sum",
          fill_value=0,
      )
      st.markdown("**Total Winning Party Votes across Election Cycles**")
      st.bar_chart(party_votes)
    else:
      st.warning("Party vote trajectory columns not found in EDA dataset.")

  with tab2:
    turnout_col_eda = find_column(
        df_eda_master,
        [
            "Turnout_Pct",
            "Voter_Turnout_Pct",
            "Turnout",
            "Voter_Turnout",
            "Turnout_%",
            "Turnout_Percentage",
            "VoterTurnoutPct",
        ],
    )
    if turnout_col_eda:
      turnout_trend = (
          df_eda_master.groupby("Year")[turnout_col_eda].mean().reset_index()
      )
      st.markdown("**eThekwini Average Ward Voter Turnout (%) Trajectory**")
      st.line_chart(turnout_trend.set_index("Year")[turnout_col_eda])
    else:
      st.warning("Turnout column not found in EDA dataset.")


# ==============================================================================
# SECTION 2: 04 NOVEMBER 2026 MODEL FORECASTS (PROJECTIONS)
# ==============================================================================
elif page_selection == "2. 04 Nov 2026 Model Forecasts":
  st.markdown(
      '<div class="main-header">🔮 04 November 2026 Ward-Level Model'
      " Forecasts</div>",
      unsafe_allow_html=True,
  )
  st.info(
      "📌 **Historical vs. Forecast Distinction:** The figures below are"
      " machine learning predictions generated by Random Forest models trained"
      " on historical lag features (t-1). They do not represent past confirmed"
      " election results."
  )

  if df_ml_master is None or rf_classifier is None:
    st.error("ML Master Dataset or Model Classifier artifact is missing.")
  else:
    # Filter 2021 election data as base for 2026 projection
    df_2021 = df_ml_master[df_ml_master["Year"] == 2021].copy()

    # Dynamically resolve columns
    turnout_col = find_column(
        df_2021,
        [
            "Turnout_Pct",
            "Voter_Turnout_Pct",
            "Turnout",
            "Voter_Turnout",
            "Turnout_%",
        ],
    )
    winning_share_col = find_column(
        df_2021,
        [
            "WinningShare_Pct",
            "Winning_Share_Pct",
            "Winning_Share",
            "WinningShare",
        ],
    )
    voters_col = find_column(
        df_2021,
        [
            "RegisteredVoters",
            "Total_Registered_Voters",
            "Registered_Voters",
            "Total_Registered",
            "Lag_Registered_Voters",
        ],
    )
    ward_col = find_column(df_2021, ["Ward_Key", "Ward", "WardNo", "Ward_ID"])

    if not all([turnout_col, winning_share_col, voters_col]):
      st.error(
          "Could not automatically resolve all required feature columns in"
          f" df_ml_master. Available columns: {list(df_2021.columns)}"
      )
    else:
      # Apply dynamic scenario simulation
      df_2021["Simulated_Turnout"] = (
          df_2021[turnout_col] + turnout_adjustment
      ).clip(10, 100)

      # Construct feature matrix matching ML training columns using Simulated_Turnout
      X_2026 = df_2021[
          [winning_share_col, "Simulated_Turnout", voters_col]
      ].rename(
          columns={
              winning_share_col: "Lag_WinningShare_Pct",
              "Simulated_Turnout": "Lag_Turnout_Pct",
              voters_col: "Lag_Registered_Voters",
          }
      )

      # Make Predictions
      pred_winner_binary = rf_classifier.predict(X_2026)
      pred_winner_prob = rf_classifier.predict_proba(X_2026)[:, 1]

      # Predict Vote Share if regressor exists, otherwise default
      if rf_regressor is not None:
        pred_vote_share = rf_regressor.predict(X_2026)
      else:
        pred_vote_share = np.zeros(len(df_2021))

      # Build 2026 Forecast DataFrame
      df_2026_forecast = pd.DataFrame()
      df_2026_forecast["Ward_Key"] = (
          df_2021[ward_col] if ward_col else df_2021.index
      )
      df_2026_forecast["Predicted_ANC_Win"] = pred_winner_binary
      df_2026_forecast["ANC_Win_Probability_%"] = np.round(
          pred_winner_prob * 100, 2
      )
      df_2026_forecast["Predicted_Winning_Share_%"] = np.round(
          pred_vote_share, 2
      )
      df_2026_forecast["Forecast_Outcome"] = np.where(
          pred_winner_binary == 1, "ANC Hold / Win", "Opposition Hold / Win"
      )

      # Classify confidence for swing ward analysis
      df_2026_forecast["Classification_Confidence"] = np.where(
          (df_2026_forecast["ANC_Win_Probability_%"] >= 40)
          & (df_2026_forecast["ANC_Win_Probability_%"] <= 60),
          "Uncertain (Swing Ward)",
          "High Confidence",
      )

      # Display Top-Level Metrics
      c1, c2, c3 = st.columns(3)
      c1.metric(
          "Projected ANC Wards Won",
          f"{sum(pred_winner_binary == 1)}",
          f"Out of {len(df_2026_forecast)} Wards",
      )
      c2.metric(
          "Projected Opposition Wards",
          f"{sum(pred_winner_binary == 0)}",
          f"Out of {len(df_2026_forecast)} Wards",
      )
      c3.metric(
          "High-Risk Swing Wards",
          f"{sum(df_2026_forecast['Classification_Confidence'] == 'Uncertain (Swing Ward)')}",
          "Probability 40% - 60%",
      )

      st.markdown("---")
      st.markdown("### 🔎 Ward-Level Forecast Lookup")

      ward_options = df_2026_forecast["Ward_Key"].unique()
      selected_ward = st.selectbox("Select Ward to Inspect:", ward_options)

      ward_detail = df_2026_forecast[
          df_2026_forecast["Ward_Key"] == selected_ward
      ].iloc[0]

      w1, w2, w3, w4 = st.columns(4)
      w1.metric("Selected Ward", f"Ward {ward_detail['Ward_Key']}")
      w2.metric("Predicted Outcome", ward_detail["Forecast_Outcome"])
      w3.metric("ANC Win Probability", f"{ward_detail['ANC_Win_Probability_%']}%")
      w4.metric(
          "Predicted Winning Share",
          f"{ward_detail['Predicted_Winning_Share_%']}%",
      )

      st.markdown("---")
      st.markdown("### 📋 Complete 2026 Municipal Ward Forecast Table")
      st.dataframe(df_2026_forecast, use_container_width=True)


# ==============================================================================
# SECTION 3: MODEL DIAGNOSTICS & PERFORMANCE EVIDENCE
# ==============================================================================
elif page_selection == "3. Model Diagnostics & Performance Evidence":
  st.markdown(
      '<div class="main-header">📈 Model Diagnostics & Empirical Evidence</div>',
      unsafe_allow_html=True,
  )
  st.markdown(
      '<div class="sub-header">Evaluation of Model 1 (Turnout), Model 2 (Vote'
      " Share), and Model 3 (Winner Classification) on 2021 hidden test"
      " set.</div>",
      unsafe_allow_html=True,
  )
  st.markdown("---")

  st.markdown("### 🏆 Summary Evaluation Matrix")
  metrics_data = {
      "Model Objective": [
          "Model 1: Voter Turnout",
          "Model 2: Party Vote Share",
          "Model 3: Ward Classifier",
      ],
      "Algorithm": [
          "Random Forest Regressor",
          "Random Forest Regressor",
          "Random Forest Classifier",
      ],
      "Primary Metric": [
          "MAE: 16.90% | R²: -5.13",
          "MAE: 14.32% | R²: -0.87",
          "Accuracy: 88.6% | F1: 0.89",
      ],
      "Leakage Prevention": [
          "Historical Lag (t-1)",
          "Historical Lag (t-1)",
          "Historical Lag (t-1)",
      ],
      "Diagnostic Status": [
          "Negative R² (Macro Shock)",
          "Negative R² (Vote Splitting)",
          "Optimal / Retained Model",
      ],
  }
  st.table(pd.DataFrame(metrics_data))

  st.markdown("### 🔍 Model 3 Feature Importance (Gini Impurity Reduction)")
  importance_data = pd.DataFrame({
      "Feature": [
          "Lag_WinningShare_Pct",
          "Lag_Registered_Voters",
          "Lag_Turnout_Pct",
      ],
      "Importance": [0.62, 0.23, 0.15],
  })
  st.markdown("**Random Forest Classifier Feature Importances**")
  st.bar_chart(importance_data.set_index("Feature"))


# ==============================================================================
# SECTION 4: LIMITATIONS & STRUCTURAL RISK
# ==============================================================================
elif page_selection == "4. Limitations & Structural Risk":
  st.markdown(
      '<div class="main-header">⚠️ Model Assumptions, Limitations & Critical'
      " Judgment</div>",
      unsafe_allow_html=True,
  )
  st.markdown("---")

  st.markdown("""
    ### 📌 Technical Justification & Diagnostic Explanations
    
    #### 1. Explanation of Negative $R^2$ in Regression Models (Models 1 & 2)
    - **Data Drift / Distributional Shift:** The 2021 election experienced a severe, non-stationary turnout collapse (~61% down to ~45%) driven by nationwide macro-shocks (COVID-19 pandemic restrictions, voter disillusionment, and the July 2021 civil unrest in KZN).
    - **Stationarity Violation:** Models trained strictly on historical local lags ($t-1$) assumed stationary turnout baselines around ~60%. Consequently, regression models overshot actual 2021 values, producing negative $R^2$ scores.
    
    #### 2. Why Model 3 (Categorical Classification) Is the Retained Solution
    - Decision tree decision boundaries in **Random Forest Classification** are resilient to uniform baseline shifts in turnout.
    - Model 3 successfully isolates relative party strongholds from competitive wards, achieving high classification accuracy without violating temporal data leakage rules.
    
    #### 3. Responsible Treatment of Close Cases (Swing Wards)
    - Rather than outputting deterministic predictions, Model 3 outputs predicted win probabilities.
    - **38 Wards (26.9%)** were identified in the uncertainty band ($0.40 \\le P \\le 0.60$). These swing wards are explicitly flagged for human analyst review rather than treated as guaranteed outcomes.
    """)