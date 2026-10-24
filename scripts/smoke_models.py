"""Check frozen recipe-model shapes using synthetic observations only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from native_city_model_v37 import NativePredictor, initialize_named, native_config  # noqa: E402


def main() -> None:
    recipes = json.loads((ROOT / "configs/recipes.json").read_text(encoding="utf-8"))
    torch.manual_seed(20261024)
    for recipe_id, horizon in (("004", 1), ("005", 1), ("009", 4)):
        original = next(row for row in recipes if row["recipe_id"] == recipe_id)
        config = native_config(original, {"frequency_seconds": 900})
        model = NativePredictor(config, nodes=8, horizon=horizon, slots_per_day=96).eval()
        initialize_named(model, 20261024)
        history = torch.randn(2, config["lookback"], 8, 1)
        daily_slots = torch.zeros(2, config["lookback"], dtype=torch.long)
        periodic = torch.randn(2, 8, 2 * horizon)
        with torch.no_grad():
            output = model(history, daily_slots, periodic)
        assert tuple(output.shape) == (2, horizon, 8), (recipe_id, tuple(output.shape))
        assert torch.isfinite(output).all(), recipe_id
        print(f"{recipe_id}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
