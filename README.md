# Interpretable Early-Life Battery Failure Prediction for Predictive Maintenance

Identifies lithium-ion cells likely to fail early from their first 50 cycles, reducing durability-test time.

[Interactive dashboard](https://nikz055.github.io/Interpretable-Hybrid-Ensemble-for-Battery-RUL-Prediction/)

## Highlights

- Built a reproducible data pipeline and Plotly dashboard to analyse battery test data.
- Applied SHAP to identify discharge-voltage drift as the leading early-failure indicator by cycle 50.
- Developed a calibrated early-warning system, cutting false alarms from 23% to 10% and test time by 74%.

## Results

83 LFP cells, Microsoft [BatteryML](https://github.com/microsoft/BatteryML) MATR-1 benchmark. Early failure: life of 488 cycles or less.

| Metric (after 50 cycles) | Early-warning system | Forced yes/no alarm |
|---|---|---|
| False alarms | 10% | 23% |
| Missed failures | 7% | 13% |
| Test time saved | 74% | — |
| Life prediction error (MAE) | 56 cycles | 70 cycles |

## Approach

- **Data pipeline:** extracts capacity, voltage, resistance and temperature features from cycles 1–50.
- **Explainability:** SHAP attribution per cell, confirmed by retraining without each feature group.
- **Early warning:** calibrated decision of fail, pass, or continue testing when uncertain.

## Validation

- 10 × 5-fold cross-validation split by battery, so no cell is scored by a model trained on it.
- Automated leakage tests, reproduction of 9 of 10 BatteryML baselines and deterministic reruns.
- Limitation: single dataset. Validation on other chemistries is required.

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
