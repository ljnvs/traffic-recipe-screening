"""Recompute a compact subset of the reported results from saved scores."""

from __future__ import annotations

import csv
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results/cross_city_scores.csv"
CHOICES = ROOT / "results/locked_recommendations.csv"
CITIES = ("Madrid", "Greater Geelong", "Latrobe")
SEEDS = (20261006, 20261007, 20261008)
HORIZONS = (15, 60)
METHODS = ("S00_fixed", "S02_nearest", "S09_primary")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    scores = rows(SCORES)
    choices = rows(CHOICES)
    neural = [row for row in scores if row["role"] != "simple"]
    simple = [row for row in scores if row["role"] == "simple"]
    assert len(neural) == 342 and len(simple) == 36
    assert {row["city"] for row in scores} == set(CITIES)

    by_score = {}
    for row in neural:
        key = (row["city"], int(row["horizon_min"]), int(row["forecast_seed"]), row["model_id"])
        assert key not in by_score, key
        by_score[key] = row
    model_ids = {f"{n:03d}" for n in
            (3, 4, 5, 6, 8, 9, 11, 13, 14, 15, 18, 19, 20, 22, 23, 24, 25)} | {"GraphWaveNet", "STAEformer"}
    assert set(by_score) == set(product(CITIES, HORIZONS, SEEDS, model_ids))
    simple_ids = {"Persistence", "DayWeekAverage", "Ridge_l12_daily", "Ridge_l12_none",
                  "Ridge_l36_daily", "Ridge_l36_none"}
    assert all(row["forecast_seed"] == "" for row in simple)
    assert {(r["city"], int(r["horizon_min"]), r["model_id"]) for r in simple} == set(
        product(CITIES, HORIZONS, simple_ids))

    print("Task-weighted mean relative MAE change (%) vs complete baseline; negative is better")
    expected = {
        ("004", "GraphWaveNet"): -2.1822083385586493,
        ("004", "STAEformer"): 0.5447736707502222,
        ("005", "GraphWaveNet"): -1.2814927089251453,
        ("005", "STAEformer"): 1.1838980064961642,
    }
    for (candidate, baseline), reference in expected.items():
        task_means = []
        for city in CITIES:
            for horizon in HORIZONS:
                paired = []
                for seed in SEEDS:
                    a = float(by_score[city, horizon, seed, candidate]["test_mae"])
                    b = float(by_score[city, horizon, seed, baseline]["test_mae"])
                    paired.append(100 * (a / b - 1))
                task_means.append(mean(paired))
        value = mean(task_means)
        assert abs(value - reference) < 1e-9, (candidate, baseline, value)
        print(f"  {candidate} vs {baseline}: {value:+.4f}")

    print("Mean parameters and recorded workflow seconds across six tasks and three seeds")
    expected_parameters = {
        "004": 11846.0, "005": 82646.0, "009": 51404.5,
        "GraphWaveNet": 271038.5, "STAEformer": 1089994.5,
    }
    for model_id, reference in expected_parameters.items():
        records = [row for row in neural if row["model_id"] == model_id]
        assert len(records) == 18
        parameters = mean(float(row["parameters"]) for row in records)
        seconds = mean(float(row["workflow_seconds"]) for row in records)
        assert abs(parameters - reference) < 1e-9
        print(f"  {model_id}: {parameters:.1f} parameters, {seconds:.2f} s")

    assert len(choices) == 18
    assert {(r["method"], r["city"], int(r["horizon_min"])) for r in choices} == set(
        product(METHODS, CITIES, HORIZONS))
    grouped = defaultdict(list)
    for row in choices:
        assert row["source_baseline_id"] == "005"
        # Meta-seeds give identical choices; repeating them leaves the mean unchanged.
        for seed in SEEDS:
            key = (row["city"], int(row["horizon_min"]), seed, row["selected_id"])
            selected_mae = float(by_score[key]["test_mae"])
            fixed_key = key[:3] + (row["source_baseline_id"],)
            fixed_mae = float(by_score[fixed_key]["test_mae"])
            grouped[row["method"]].append(100 * (1 - selected_mae / fixed_mae))
    print("Mean gain over source-fixed 005 (%); positive is better")
    for method, expected_gain in (("S00_fixed", 0.0), ("S02_nearest", 0.7882703621546132), ("S09_primary", 0.0)):
        assert len(grouped[method]) == 18
        value = mean(grouped[method])
        assert abs(value - expected_gain) < 1e-9, (method, value)
        print(f"  {method}: {value:+.4f}")


if __name__ == "__main__":
    main()
