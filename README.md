# ML Competition Ensemble

Tabular ML competition pipeline. CatBoost ensembling, stacking, feature engineering. Anonymized; not tied to any specific competition.

## Approach

- Multi-seed CatBoost bagging with stratified K-fold averaging.
- Stacking layer: CatBoost + LightGBM out-of-fold predictions feeding a logistic meta-learner.
- Specialized sub-models blended at inference: global classifier, recent-window model, suspicious-vs-negative head, positive-vs-ambiguous head, multiplied in logit space.
- Time-aware features: sequence stats (events before today, seconds since previous same-type event, deltas), category priors with smoothing, screen and device parsing.
- Holdout selected by date, not random, to avoid leakage on time-ordered data.

## Stack

Python, polars, pandas, numpy, scikit-learn, CatBoost, LightGBM.

## Quickstart

```bash
pip install -r requirements.txt
python solution.py
```

The scripts assume parquet inputs under `data/` (`train_part_*.parquet`, `test.parquet`, `train_labels.parquet`, `sample_submit.csv`). Original competition data is not included; the pipeline is reusable on any tabular regression or classification dataset by adapting the feature list and target column.

## Layout

- `solution.py`, `solution_on_pc_1.py` - main multi-seed CatBoost pipelines
- `solut_tig.py` - full anti-fraud notebook export with category priors and 4-model blend
- `solutions/` - alternate variants and stacking experiments
- `analiz.py`, `sad.py`, `analyze_predictions.py` - EDA and prediction-disagreement analysis
- `test.py` - quick correlation check between submissions

## License

MIT
