# Traffic recipe screening

Code and aggregate results accompanying **“Recipe screening for traffic forecasting at native sampling intervals: accuracy–cost trade-offs and cross-city evaluation.”**

- `src/`: frozen recipe predictor and its dependencies.
- `configs/recipes.json`: 27 design configurations with three-digit IDs; 17 pass the city-panel input constraints. `lookback` is in five-minute design units and is converted by `native_config` to native time steps.
- `results/cross_city_scores.csv`: all 342 neural-fit scores and 36 seed-shared simple-method scores for Madrid, Greater Geelong and Latrobe at 15/60 minutes. Together these form the paper's 450 evaluation records.
- `results/locked_recommendations.csv`: 18 task-level choices for source-fixed 005, nearest-neighbour matching and S09 in the 23-candidate pool. Each choice is identical across meta-seeds 20261010–20261012 and predictor seeds 20261006–20261008; duplicate rows are omitted.

Recompute the recorded comparisons with Python 3.10+ (standard library only):

```bash
python scripts/recompute_cross_city.py
```

Check recipes 004, 005 and 009 on synthetic inputs:

```bash
python -m pip install -r requirements-model.txt
python scripts/smoke_models.py
```

Positive **gain** and negative **relative MAE change** both mean lower error. Costs describe RTX 4090 workflows with model-specific micro-batches. The scripts check saved-score arithmetic and model execution; full training pipelines and external baseline implementations are outside this package.

The reported region set was defined after the full results were available. The Victorian target regions share a provider and month with two source regions. The comparisons support task-dependent accuracy–cost trade-offs; source-based selection did not establish a consistent gain over fixed 005.

Original observations are available from [UTD19](https://utd19.ethz.ch/), [GO-MO](https://huggingface.co/datasets/dmariaa70/GO-MO/tree/1aa5f0dd4814c88607f5578d8e34ba27d70722cb) and [Victoria SCATS](https://opendata.transport.vic.gov.au/dataset/traffic-signal-volume-data/resource/fef12215-683b-4c5e-8210-5c152036a4d2), subject to the providers' terms. This repository contains no observation data or trained weights.
