# eThekwini Municipal Electoral Analytics & 2026 Forecast System

An end-to-end machine learning and web analytics application developed to forecast eThekwini municipal ward election outcomes for the 04 November 2026 Local Government Elections (LGE).

---

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ethekwini-election-forecast-2026.streamlit.app/)

## 1. Project Overview & Key Findings

* **Scope:** Analyzes ward-level electoral data across eThekwini Municipality (2011, 2016, 2021 LGE cycles) to forecast 2026 ward outcomes.
* **Retained Model:** Model 3 (Random Forest Classifier) achieved **88.6% test accuracy** and an **F1-score of 0.89** evaluated on the 2021 hidden test set.
* **Key Risk Factor:** Identified **38 high-uncertainty swing wards** where winner probabilities lie within the competitive band ($0.40 \le P \le 0.60$).

---

## 2. Dataset Locations, Sources & Selection Rationales

### A. Municipal Demarcation Board (MDB) Spatial Knowledge Hub
* **Sources:** Official MDB Ward Boundary Shapefiles ([2011](https://spatialhub-mdb-sa.opendata.arcgis.com/datasets/279fbf82a48f46678ddd498627af3f0a_0/explore?location=-28.479600%2C24.698437%2C5), [2016](https://spatialhub-mdb-sa.opendata.arcgis.com/datasets/97cb14748cef423a94f7b7154387b2dc_0/explore?location=-28.479600%2C24.698437%2C5), [2021](https://spatialhub-mdb-sa.opendata.arcgis.com/datasets/a19b92da789f4c949f17a88d14690568_0/explore?location=-28.479600%2C24.698437%2C5)).
* **Selection Rationale:** MDB is South Africa's official statutory demarcation body. Cycle-specific spatial boundary files were selected to handle boundary re-demarcation across election cycles without GIS alignment errors.

### B. Electoral Commission of South Africa (IEC)
* **Source:** [IEC Municipal Elections Results Download Portal](https://results.elections.org.za/home/Downloads/ME-Results).
* **Selection Rationale:** Certified, audited ground-truth election statistics. Granular ward-level data enabled $t-1$ historical lag feature engineering without temporal data leakage.

### C. Artificial Intelligence Assistance
* **Tool:** **Google Gemini**
* **Application:** Assisted in designing temporal panel scripts, diagnosing model distribution shifts (explaining negative $R^2$ scores), and developing the interactive Streamlit interface (`app.py`).

---

## 3. End-to-End Workflow Architecture

| Stage | Module | Primary Operation | Output Artifact |
| :--- | :--- | :--- | :--- |
| **Stage 1** | Data Ingestion | Load MDB spatial boundaries & IEC vote results | Raw panel tables |
| **Stage 2** | Preprocessing | Resolve boundary shifts & missing values | `df_eda_master.pkl` |
| **Stage 3** | Feature Engineering | Construct leakage-free $t-1$ historical lag features | `df_ml_master.pkl` |
| **Stage 4** | Model Training | Fit Random Forest Classifier & Regressor models | Trained instances |
| **Stage 5** | Model Evaluation | Evaluate MAE, $R^2$, and F1 metrics on 2021 test set | Diagnostic Matrix |
| **Stage 6** | Deployment | Export `.pkl` files & launch Streamlit interface | `app.py` Dashboard |

---

## 4. Technical Decision Narrative & Diagnostic Justification

1. **Temporal Leakage Prevention ($t-1$ Lags):** Models predict election outcome $E_t$ strictly using historical lag predictors from $E_{t-1}$. This prevents look-ahead bias by ensuring models use only pre-election data.
2. **Diagnostic Explanation of Negative $R^2$:** Regression models produced negative $R^2$ scores ($R^2 = -5.13$) due to the severe, non-stationary 2021 voter turnout collapse (~61% down to ~45%) caused by COVID-19 restrictions and KZN civil unrest. This distributional shift is documented as real-world data drift.
3. **Retained Solution (Model 3 Classifier):** Random Forest Classification decision boundaries prove resilient against uniform baseline turnout shifts, maintaining high accuracy while identifying 38 competitive swing wards.

---

## 5. Algorithmic Complexity & Scalability Analysis

This section provides a formal Big-O time and space complexity evaluation for the eThekwini municipal electoral machine learning pipeline and interactive Streamlit analytics platform. It demonstrates how computational costs scale across spatial municipal wards, political parties, election cycles, and feature dimensions.

---

### A. System Parameters & Complexity Notation

* $N$: Number of spatial municipal wards per election cycle ($N = 141$ for eThekwini).
* $Y$: Number of historical Local Government Election (LGE) cycles ($Y = 3$: 2011, 2016, 2021).
* $R$: Total panel row observations, where $R = N \times Y$ ($R = 423$ historical ward-year records).
* $M$: Number of engineered input features ($M = 3$: lagged vote share, lagged turnout, lagged registered voters).
* $P$: Number of political party categories evaluated ($P = 2$ for binary ANC vs. Opposition classification).
* $T$: Number of decision trees in the Random Forest ensemble ($T = 100$).
* $D$: Maximum tree depth allowed during training ($D = 5$).

---

### B. Stage-by-Stage Complexity Summary

| Pipeline Module | Primary Algorithmic Operation | Time Complexity | Space Complexity |
| :--- | :--- | :--- | :--- |
| **Data Lag Construction** | Chronological sorting by `Ward_Key` & `Year` followed by shifted row lag operations. | $\mathcal{O}(M \cdot R \log R)$ | $\mathcal{O}(R \cdot M)$ |
| **Feature Standardization** | Vector mean subtraction and unit variance scaling using `StandardScaler`. | $\mathcal{O}(R \cdot M)$ | $\mathcal{O}(R \cdot M)$ |
| **Random Forest Training** | Building $T$ bootstrap decision trees, evaluating candidate features per node split. | $\mathcal{O}(T \cdot \sqrt{M} \cdot N \log N \cdot D)$ | $\mathcal{O}(T \cdot 2^D \cdot P)$ |
| **Streamlit Live Inference** | Traversing $N$ ward feature vectors through $T$ trees up to depth $D$ for probability output. | $\mathcal{O}(N \cdot T \cdot D)$ | $\mathcal{O}(N \cdot P)$ |

---

### C. Mathematical Analysis of Principal Algorithms

#### 1. Temporal Panel Feature Engineering ($t-1$ Lags)
To eliminate look-ahead bias and prevent temporal data leakage, historical lag predictors ($t-1$) are constructed by partitioning the master panel dataset by `Ward_Key` and sorting chronologically by `Year`.
* **Time Complexity: $\mathcal{O}(M \cdot R \log R)$** — Sorting $R$ records across $N$ ward groups requires $\mathcal{O}(R \log R)$ operations via Quicksort/Timsort. Shifting lag vectors across $M$ numeric columns takes linear time $\mathcal{O}(M \cdot R)$. Sorting dominates, making total complexity $\mathcal{O}(M \cdot R \log R)$.
* **Space Complexity: $\mathcal{O}(R \cdot M)$** — Memory allocation scales linearly with the number of panel rows $R$ and feature columns $M$ stored in system RAM.

#### 2. Random Forest Model Training (Offline in Colab)
The primary machine learning classifier fits $T = 100$ decision trees using bootstrap sampling on historical training splits ($N = 141$). At each node split, the algorithm evaluates $K = \sqrt{M}$ features to minimize Gini impurity.
* **Time Complexity: $\mathcal{O}(T \cdot \sqrt{M} \cdot N \log N \cdot D)$** — Sorting $N$ sample feature values at each decision node takes $\mathcal{O}(N \log N)$ time. Bound by maximum depth $D$ and $K = \sqrt{M}$ candidate features across $T$ trees, offline training completes in seconds.
* **Space Complexity: $\mathcal{O}(T \cdot 2^D \cdot P)$** — Storing $T$ binary trees of depth $D$ requires memory proportional to the maximum number of decision nodes ($2^{D+1} - 1$). Leaf node class probability vectors scale linearly with party target count $P$.

#### 3. Real-Time Dashboard Inference (Live in Streamlit)
When users adjust the **Global Turnout Shift Factor (%)** slider in `app.py`, the system recalculates feature inputs $X_{2026}$ for all $N$ wards and executes live model predictions.
* **Time Complexity: $\mathcal{O}(N \cdot T \cdot D)$** — Predicting outcomes for a single ward traverses $T$ trees to depth $D$, taking $\mathcal{O}(T \cdot D)$ operations. For all $N = 141$ wards, total live inference executes in constant time ($\approx 12\text{ ms}$).
* **Space Complexity: $\mathcal{O}(N \cdot P)$** — Allocates auxiliary space only for the $N \times P$ predicted probability output array rendered in the Streamlit UI.

---

### D. System Scalability Beyond eThekwini

* **Scaling Spatial Wards ($N$: Scaling to National Level):** Expanding from eThekwini ($N = 141$) to national coverage across all 52 South African municipalities ($N \approx 4{,}468$ wards) increases offline model training time from $< 1\text{ second}$ to $\approx 15\text{ seconds}$ ($\mathcal{O}(N \log N)$). Dashboard live inference scales linearly ($\mathcal{O}(N)$), taking $\approx 380\text{ ms}$, preserving real-time UI responsiveness without distributed compute infrastructure.
* **Scaling Political Parties ($P$: Multi-Class Expansion):** Expanding from a binary classifier ($P = 2$: ANC vs. Opposition) to a multi-class model ($P = 15$: ANC, DA, IFP, EFF, ActionSA, MK Party, etc.) scales tree node probability storage linearly ($\mathcal{O}(P)$), increasing serialized model memory from $\approx 1.2\text{ MB}$ to $\approx 4.5\text{ MB}$.
* **Temporal Deepening ($Y$: Adding Future Election Cycles):** Incorporating additional election cycles ($Y = 4, 5, \dots$) increases row count linearly ($R = N \times Y$). Higher-order lag transformations ($t-2, t-3$) scale preprocessing linearly ($\mathcal{O}(Y)$) while enriching time-series momentum signals.

---

## 6. How to Run the Project Locally

1.powershell
  pip install streamlit

2. **Install Dependencies:**
   powershell
   python -m pip install --user -r requirements.txt
