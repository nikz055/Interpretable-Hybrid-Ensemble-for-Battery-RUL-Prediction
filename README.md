# Spotting Batteries That Will Fail Early

Battery durability tests charge and discharge every cell until it wears out, which can take hundreds to over
2,000 cycles. This project asks a simple question:

**After only the first 50 cycles, can we tell which cells will fail early, and why?**

**[Open the interactive dashboard](https://nikz055.github.io/Interpretable-Hybrid-Ensemble-for-Battery-RUL-Prediction/)**

## The answer

Yes, for most cells. Tested on 83 lithium-ion cells from the Microsoft
[BatteryML](https://github.com/microsoft/BatteryML) MATR-1 benchmark:

| After 50 cycles | Result |
|---|---|
| Healthy cells wrongly flagged | **10%** (a simple yes/no alarm: 23%) |
| Early failures missed | **7%** (a simple yes/no alarm: 13%) |
| Durability-test time saved | **74%** compared with testing every cell to failure |
| Error in predicted battery life | **56 cycles** (first version: 70) |

"Early failure" means a cell that wears out within 488 cycles.

## How it works

1. **Measure.** Turn the first 50 cycles of data into features describing capacity, voltage, resistance and temperature.
2. **Predict.** Estimate each cell's total life, with an uncertainty range.
3. **Decide.** Give each cell one of three answers: *fail*, *pass*, or *not sure yet, keep testing*. Letting uncertain
   cells keep cycling, instead of forcing a yes or no, is what halves the false alarms.
4. **Explain.** SHAP shows which signals drove each cell's prediction, so an engineer can check the reasoning.

## What we learned

- **Voltage gives the earliest warning.** Cells whose discharge voltage drifts the most in the first 50 cycles tend to
  fail first. Capacity loss, the usual health indicator, is a much weaker early signal.
- **Before about 50 cycles, predictions lean on how a cell is charged** and its starting condition, not on how it is ageing.
- **Simple beats complex here.** A linear model (PLSR) was as accurate as a tuned ensemble of Random Forest, XGBoost and
  a neural network.
- **Very early warning did not work.** Predictions after only 5 to 10 cycles flagged too many healthy cells to be useful.

## Why the numbers can be trusted

- Every cell is judged only by models that never saw it (10 rounds of 5-fold cross-validation, split by battery).
- Automated tests check that no feature can use data from after the decision point.
- The pipeline reproduces nine of Microsoft BatteryML's ten published baseline results, and reruns give identical outputs.
- An earlier headline result from this repository was audited and withdrawn: see [the audit](docs/AUDIT.md).

**Limitation:** one dataset (83 LFP cells from one lab). The method should be re-checked on other battery types before use.

## Run it

```bash
pip install -r requirements.txt
```

Get the data with BatteryML (`batteryml download MATR` and `batteryml preprocess MATR`), then run everything:

```bash
python run_all.py --data-dir path/to/processed/MATR
```

## Learn more

- [Technical details](docs/TECHNICAL_DETAILS.md): methods, all experiments, limitations
- [Full results report](results/REPORT.md): every table, with confidence intervals
- [Feature dictionary](docs/FEATURE_DICTIONARY.md): what each feature measures and why

Tools: Python, scikit-learn, XGBoost, SHAP, Plotly.
