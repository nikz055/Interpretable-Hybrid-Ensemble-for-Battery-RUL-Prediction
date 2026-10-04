# Audit of the original implementation

Scope: the repository state at commit `9fa09a4` ("Add files via upload"):
`README.md` (title only) and `xai-battery-degradation.ipynb` (now in
`notebooks/legacy/`). Every claim below can be re-checked with
`python scripts/audit_legacy_notebook.py`, which re-implements the notebook
cells 8–16 on the same BatteryML-processed MATR pickles.

## 1. Can the reported MATR-1 RMSE of 103.5 be reproduced?

**Yes, exactly, but it should not be used as a result.**

| Variant | Test RMSE (own label) | Test RMSE (BatteryML label) | Training 5-fold CV RMSE |
|---|---|---|---|
| L0 legacy pipeline as written (label = number of recorded cycles) | **103.5** | 105.7 | 210.8 |
| L1 same pipeline, BatteryML 80 %-SOH label | 104.5 | 104.5 | 207.6 |

Feature values for `b1c1` printed by the re-implementation match the notebook's
printed output to six decimals, so the data, features and model really are the same.

Why the number cannot stand as a headline result:

1. **Wrong target definition.** The notebook uses `label = len(cell.cycle_data)`,
   the number of recorded cycles. BatteryML's `RULLabelAnnotator` uses the first
   cycle whose discharge capacity is ≤ 0.8 × 1.1 Ah. The two differ for all
   83 MATR-1 cells (legacy is longer by 12.4 cycles on average, up to 36).
   See `results/tables/labels.csv`.
2. **One untuned configuration, scored only on the test set.** XGBoost
   `n_estimators=100, max_depth=3, learning_rate=0.1` was never selected by
   validation. Across 36 neighbouring configurations, test RMSE ranges from
   101.5 to 178.0 (median 114.4; `results/tables/legacy_hparam_sensitivity.csv`).
   A test number that moves this much with untuned hyperparameters is not a
   reliable estimate.
3. **Test RMSE is unusually low compared with validation.** The same pipeline has a
   grouped 5-fold CV RMSE of ≈ 208 on the training cells. The gap is driven by
   one training cell, `b1c1` (2160 cycles). When it is held out, the trees
   cannot predict beyond the longest remaining training life (1434) and miss
   by ~1100 cycles; without it the CV RMSE is 117. So a 42-cell test RMSE of
   about 100 hides heavy-tailed errors on long-lived cells. This is why we
   report MAE, median AE, per-group errors and bootstrap CIs alongside RMSE.
4. **Single model only.** The repository title promises a hybrid
   RF + XGBoost + MLP ensemble, but no RF, MLP or ensemble code exists.

## 2. Component-by-component status

Status legend: ✅ implemented & verified · 🟡 implemented, needs validation ·
🟠 incomplete · ❌ incorrect / leakage-prone · ♻️ redundant · 💡 proposed, not
implemented · 📌 required for final project.

| Component | Legacy status | Finding | Action in this repo |
|---|---|---|---|
| Directory structure | 🟠 | One notebook, no package, scripts, configs or tests | `batteryrul/` package, `scripts/`, `configs/`, `tests/` |
| Data loading | 🟡 | Minimal BatteryData clone, correct format; `except Exception: pass` silently drops cells | `batteryrul/io.py`, which fails loudly on missing or invalid cells |
| Dataset scope | ❌ | Mixes HUST into an MATR-1 study | MATR-1 only; HUST → future work |
| Target / label | ❌ | `len(cycle_data)` ≠ BatteryML label | Port of `RULLabelAnnotator` (`batteryrul/labels.py`), unit-tested |
| Train/test split | ✅ | Uses BatteryML `MATRPrimaryTest` ids (b2c1 already removed) | Same ids, saved as `splits/*.csv` |
| Validation split | 🟠 | None; hyperparameters and test evaluation coincide | Grouped 5-fold CV manifest on training cells |
| Feature extraction | 🟡 | Severson features with `critical_cycles=[1,9,99]` (BatteryML config uses `[2,9,99]`), `ddof=0` variance, `nan_to_num` before smoothing | Exact BatteryML port = Feature Set A; legacy variant kept only in the audit script |
| Feature naming | ❌ | "Avg_Charge_Time" sums time where I < 0, which is the **discharge** step. "Temperature_Integral" is log mean temperature | Documented in the feature dictionary; Set B uses correctly defined durations |
| Data artifacts | 🟠 | Not checked. Batch-1 cells have IR = 0 on one early cycle (so `Min_Internal_Resistance` = 0 for 10 cells). b1c0 and b1c18 have single-cycle Qd spikes (1.54, 2.88 Ah) within the first 100 cycles | Set B/C treat IR ≤ 0 as missing and Hampel-filter capacity trajectories |
| Missing values | 🟠 | Hard-coded 0 fill | Train-fitted median imputer inside the model pipeline |
| Scaling | ✅ | Fit on train only | Same, inside an sklearn `Pipeline` so CV folds refit it |
| Outlier clipping | 💡 | None | Train-fitted winsorisation (1st/99th pct) |
| Random Forest | 💡 | Not implemented | Tuned by grouped CV |
| XGBoost | 🟡 | Untuned, test-only evaluation | Tuned by grouped CV |
| MLP | 💡 | Not implemented | sklearn MLP, early stopping, 10 seeds |
| Ensemble | 💡 | Not implemented | Equal, validation-weighted and constrained OOF stacking |
| Evaluation | 🟠 | RMSE only, single seed, no CI | 9 metrics, 10 seeds, bootstrap CIs, paired tests, life groups |
| SHAP | 🟡 | TreeExplainer on test cells (OK), but no stability check, physics claims overstated ("we verify it learned real physics") | Held-out SHAP for RF/XGB/MLP/stack, fold/seed stability, explicit claim boundaries |
| Saved models / predictions | 🟠 | XGB JSON saved to the Kaggle working dir only, no predictions | Per-cell prediction CSVs, models, configs in `results/` |
| Reproducibility | 🟠 | No requirements, seeds partly set, Kaggle-specific paths | `requirements.txt`, config file, one-command `run_all.py`, Kaggle notebook |
| Documentation | 🟠 | README title only | Full README, this audit, feature dictionary |

## 3. Other observations

* The notebook's interpretation section asserts mechanisms (lithium plating,
  SEI growth) from SHAP values. SHAP shows what the model has learned to associate,
  not causation; the new analysis words claims accordingly.
* `use_precalculated=True` (Qdlin from the raw MATR files) is the BatteryML
  MATR-1 setting and is retained.
