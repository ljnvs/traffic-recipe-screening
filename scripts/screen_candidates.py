"""Inspect input feasibility before reading any performance scores."""
from pathlib import Path
import argparse
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from component_constraints_v5 import violations

CAPS = dict(provided_graph=False,symmetric_graph=False,relative_daily_slot=True,
            verified_calendar=False,training_semantic_graph=False,
            training_pattern_prototypes=False,laplacian_positions=False)


def screen(step=900):
    bank = json.loads((ROOT/'configs/recipes.json').read_text(encoding='utf-8'))
    feasible,excluded = [],[]
    for original in bank:
        row = dict(original)
        reasons = []
        if row['lookback']*300 % step:
            reasons.append('Nonintegral physical lookback')
        else:
            row['lookback'] = row['lookback']*300//step
        reasons += violations(row,CAPS)
        if row['assembly']=='graph_sandwich' and row['lookback']-8<=1:
            reasons.append('Valid convolution blocks do not fit physical history')
        (excluded if reasons else feasible).append(dict(recipe_id=row['recipe_id'],reasons=reasons))
    return dict(frequency_seconds=step,feasible=feasible,excluded=excluded,performance_scores_read=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frequency-seconds',type=int,default=900)
    args=parser.parse_args()
    print(json.dumps(screen(args.frequency_seconds),indent=2))
