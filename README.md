# Early Battery Failure Prediction

Interpretable machine learning that identifies lithium-ion cells likely to fail early from their first 50 cycles,
reducing durability-test time.

[Interactive dashboard](https://nikz055.github.io/Interpretable-Hybrid-Ensemble-for-Battery-RUL-Prediction/)

## Results

83 LFP cells, Microsoft [BatteryML](https://github.com/microsoft/BatteryML) MATR-1 benchmark. Early failure: life of 488 cycles or less.

| Metric (after 50 cycles) | This work | Forced yes/no alarm |
|---|---|---|
| False alarms | 10% | 23% |
| Missed failures | 7% | 13% |
| Test time saved | 74% | — |
| Life prediction error (MAE) | 56 cycles | 70 cycles |

## Approach

- **Features:** capacity, voltage, resistance and temperature signals from cycles 1–50.
- **Prediction:** cycle-life model with calibrated uncertainty intervals.
- **Decision:** fail, pass, or continue testing when uncertain, which halves false alarms.
- **Explanation:** SHAP attribution per cell.

## Key findings

- Discharge-voltage drift is the strongest early indicator of failure; capacity fade is weaker.
- A linear model (PLSR) matched a tuned ensemble of Random Forest, XGBoost and MLP.
- Predictions from 5–10 cycles were not reliable enough for use.

## Validation

- 10 × 5-fold cross-validation split by battery; no cell is scored by a model trained on it.
- Automated leakage tests; reproduces 9 of 10 BatteryML baselines; deterministic reruns.
- Limitation: single dataset; validation on other chemistries is required.

## Usage

```bash
pip install -r requirements.txt
```

```bash
python run_all.py --data-dir path/to/processed/MATR
```

Data: prepared with BatteryML (`batteryml download MATR`, `batteryml preprocess MATR`).

## Documentation

[Technical details](docs/TECHNICAL_DETAILS.md) · [Results report](results/REPORT.md) · [Feature dictionary](docs/FEATURE_DICTIONARY.md) · [Audit](docs/AUDIT.md)

Python · scikit-learn · XGBoost · SHAP · Plotly
