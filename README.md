# Corporate Distress Prediction Using Machine Learning

## 🎯 Research Question

Can machine learning models predict corporate financial distress 12 months in advance using CDS spreads, accounting fundamentals, and market data?

**Target Variable**: Binary classification (1 = firm experiences distress in next 12 months, 0 = otherwise)

**Final Performance**: 
- **Recommended Model (Exp 16):** AUC 0.640, Recall 72%, F1 0.420 (Top 10 features)
- **Alternative (Calibrated):** AUC 0.662, Recall 69.1% (29 features, better calibration)
- **Improvement:** 58% better than CDS-only baseline

---

## 🚀 Quick Start

### 1. Set Up Environment

**Option A: Using Conda (Recommended for Nuvolos)**
```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate cds-distress-prediction
```

**Option B: Using pip (Local development)**
```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

**Default: Pipeline + Experiments (Recommended)**
```bash
python main.py
```
This runs all 15 pipeline steps + 7 optimization experiments.

**Alternative: Experiments only**
```bash
python main.py --experiments-only
```
Skips pipeline steps, runs only experiments (requires existing data).

**What gets executed:**

**Core Pipeline (15 steps):**
- **Data Processing** (Steps 1-4): Inspect, clean, and merge datasets
- **Feature Engineering** (Steps 5-8): Create accounting and market features
- **Target Creation** (Step 9): Define distress events
- **Model Training** (Steps 10-12): Train and optimize models
- **Evaluation** (Steps 13-15): Assess performance and explainability

**Optimization Experiments (7 experiments):**
- **Exp 1:** Reduce overfitting (regularization)
- **Exp 4:** Optimize recall (threshold tuning)
- **Exp 5:** Add temporal features (key contribution)
- **Exp 6:** Combine all optimizations
- **Exp 13:** Calibrate probabilities (deployment-ready)
- **Exp 14:** Cross-validation (stability check)
- **Exp 16:** Feature selection (Top 10 features) ⭐ **RECOMMENDED MODEL**

### 3. Expected Output

**Core Pipeline:**
- **Trained Models:** `output/models/lightgbm_optimized.pkl`
- **Results:** `output/step13_evaluation_results.csv`
- **Figures:** `report/figures/`

**Default Output (Pipeline + Experiments):**
- **Recommended Model:** `output/experiments/models/exp16_xgboost.pkl` (Top 10 features)
- **Alternative Model:** `output/models/lightgbm_calibrated_isotonic.pkl` (Calibrated)
- **Experiment Results:** `output/experiments/`
- **Experiment Figures:** `report/figures/experiments/`

**Runtime:** 
- Complete run (default): ~30-40 minutes
- Experiments only: ~15-20 minutes

---

## 📂 Project Structure

```
CDS Project/
├── main.py                  # ⭐ Entry point - Run this!
├── README.md                # This file
├── FINAL_REPORT.md          # Complete research report
├── requirements.txt         # Python dependencies
│
├── src/                     # Core 15-step pipeline
│   ├── step1_data_inspection.py
│   ├── step2_data_quality.py
│   ├── step3_data_cleaning.py
│   ├── step4_data_merging.py
│   ├── step5_preprocessing.py
│   ├── step6_accounting_features.py
│   ├── step7_market_features.py
│   ├── step8_feature_validation.py
│   ├── step9_target_creation.py
│   ├── step10_ml_construction.py
│   ├── step11_model_training.py
│   ├── step12_model_optimization.py
│   ├── step13_model_evaluation.py
│   ├── step14_benchmark_comparison.py
│   └── step15_explainability.py
│
├── experiments/             # Model optimization experiments
│   ├── exp1_reduce_overfitting.py
│   ├── exp4_optimize_recall.py
│   ├── exp5_temporal_features.py      # ⭐ Key contribution
│   ├── exp6_combined_optimization.py
│   ├── exp13_model_calibration.py     # 🎯 Calibration
│   ├── exp14_cross_validation.py      # Stability check
│   └── exp16_temporal_feature_selection.py  # 🏆 RECOMMENDED MODEL
│
├── data/                    # Raw data files
├── output/                  # Generated outputs
├── notebooks/               # Jupyter notebooks (optional)
└── report/                  # Figures and documentation
```

---

## 📊 Datasets

### 1. Compustat - Quarterly Fundamentals
- **Source**: Compustat North America
- **Frequency**: Quarterly
- **Key Variables**: Assets, liabilities, equity, income, cash, debt, etc.
- **Purpose**: Calculate accounting ratios (leverage, liquidity, profitability)

### 2. CRSP - Security Prices
- **Source**: CRSP Monthly Stock File
- **Frequency**: Monthly
- **Key Variables**: Prices, returns, shares outstanding, market cap
- **Purpose**: Calculate market-based features (volatility, momentum, beta)

### 3. GVKEY-CUSIP Mapping
- **Purpose**: Link Compustat (GVKEY) with CRSP and CDS data (CUSIP)

### 4. CDS Spreads
- **Source**: IHS Markit
- **Frequency**: Quarterly
- **Key Variables**: 5-year senior CDS spreads
- **Purpose**: Create target variable (spread widening)
- **Status**: ✅ Integrated

---

## 🔬 Advanced: Run Experiments

To reproduce the complete optimization and validation:

```bash
# Core Optimization (5 experiments)
python experiments/exp1_reduce_overfitting.py      # Regularization
python experiments/exp4_optimize_recall.py         # Threshold tuning
python experiments/exp5_temporal_features.py       # Temporal features ⭐
python experiments/exp6_combined_optimization.py   # Final model 🏆
python experiments/exp13_model_calibration.py      # Calibration 🎯

# Advanced Validation (2 experiments)
python experiments/exp14_cross_validation.py       # Time-series CV 📊
python experiments/exp15_lstm_baseline.py          # Deep learning 🧠
```

**Or run all experiments:**
```bash
python main.py --with-experiments
```

See `experiments/README.md` for detailed documentation of each experiment.

---

## 🏆 Key Results

### Recommended Model: XGBoost with Top 10 Features (Exp 16) ⭐
- **Model:** XGBoost with feature selection (10 features)
- **Test AUC:** 0.640
- **Precision:** 30%, Recall: 72%, F1: 0.420
- **Catches:** 1,054 / 1,463 distressed firms (72%)
- **Improvement:** 58% better than CDS-only baseline
- **Advantages:** Simpler (66% fewer features), faster, more interpretable

### Alternative: Calibrated LightGBM (29 features)
- **Test AUC:** 0.662
- **Recall:** 69.1%, Precision: 32.2%
- **Calibration (ECE):** 0.0140 (excellent)
- **Use when:** Probability estimates are critical

### Top 10 Features (Exp 16):
1. **cds_spread_lag1** - Recent CDS trajectory
2. **altman_z_score** - Financial health
3. **return_1m** - Recent stock performance
4. **volatility_12m** - Market uncertainty
5. **momentum_12m** - Long-term trend
6. **profit_margin** - Profitability
7. **debt_to_assets** - Leverage
8. **cds_spread_lag4** - CDS history
9. **debt_to_equity** - Capital structure
10. **momentum_3m** - Short-term momentum

### Optimization Journey:
- **Baseline:** AUC 0.6481, Recall 46.4%
- **+ Regularization:** Overfitting reduced by 57%
- **+ Threshold tuning:** Recall improved to 72.6%
- **+ Temporal features:** AUC +4.7% (key contribution)
- **+ Calibration:** ECE improved by 93.6%
- **Final:** AUC 0.6622, Recall 69.1% ✅

---

## 📝 Data Requirements

Place these files in the `data/` folder:
- `CDS firms fundamentals Quarterly.csv` - Compustat quarterly data
- `Security prices.csv` - CRSP market data
- `firm_cusip_mapping_for_cds.csv` - GVKEY-CUSIP mapping
- `GVKEY US Firms csv.csv` - Firm identifiers

**Sample Size:**
- **608 unique firms** (586 train, 538 test)
- **28,247 firm-quarter observations** (21,971 train, 6,276 test)
- **Temporal split:** 2010-2020 (train), 2021-2023 (test)
- **Distress rate:** 19.6% (train), 23.3% (test)

---

## 🧪 Testing & Validation

### Run Unit Tests
Verify data integrity, model correctness, and prevent data leakage:

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Test Coverage:**
- ✅ Data leakage prevention (no future information in features)
- ✅ Temporal integrity (strict train/test split)
- ✅ Model predictions validity (probabilities in [0,1])
- ✅ Performance metrics (AUC > 0.5, recall > 0)
- ✅ Reproducibility (consistent results)

### Compute Confidence Intervals
Get statistical confidence intervals for model metrics:

```bash
python src/step13b_confidence_intervals.py
```

This computes 95% confidence intervals via bootstrap (1,000 iterations):
- **AUC:** 0.6677 ± 0.0187 (95% CI: [0.6248, 0.6996])
- **Recall:** 0.691 ± 0.023 (95% CI: [0.645, 0.737])
- **Precision:** 0.322 ± 0.018 (95% CI: [0.286, 0.358])

---

## 📖 Documentation

- **Complete Report:** `FINAL_REPORT.md` - Full methodology and results
- **Experiments:** `experiments/README.md` - Optimization journey details
- **Notebooks:** `notebooks/` - Interactive exploration (optional)
- **Tests:** `tests/` - Unit tests for data integrity and model validation

---

## 👤 Author

Altan Karagulle

---

## 📅 Last Updated

December 1, 2025 - Reorganized for final submission
