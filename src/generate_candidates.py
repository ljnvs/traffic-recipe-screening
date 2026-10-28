"""Generate deterministic covering designs from source contracts, never from scores."""
import itertools
import json
import math
import sys
from pathlib import Path
import numpy as np
from component_constraints_v5 import is_valid, violations

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs"
DESIGN = json.loads((PLAN / "traffic_component_design_v5.json").read_text(encoding="utf-8"))
FACTORS = list(DESIGN["dimensions"])
VALUES = DESIGN["dimensions"]
PAIRS = list(itertools.combinations(range(len(FACTORS)), 2))


def key(config):
    return tuple(config[f] for f in FACTORS)


def matrix_report(rows, all_valid):
    labels = ["intercept"] + [f"{f}={v}" for f in FACTORS for v in VALUES[f][1:]]
    def matrix(configs):
        return np.array([[1.] + [float(c[f] == v) for f in FACTORS for v in VALUES[f][1:]] for c in configs])
    x, full = matrix(rows), matrix(all_valid)
    rank, full_rank = int(np.linalg.matrix_rank(x)), int(np.linalg.matrix_rank(full))
    active = [i for i in range(len(labels)) if bool(np.any(full[:, i] != 0))]
    # A subset cannot remove aliases forced by structural legality.
    return {"columns": len(labels), "active_columns": len(active), "inactive_labels": [label for i,label in enumerate(labels) if i not in active],
            "labels": labels, "pool_rank": rank, "legal_space_rank": full_rank,
            "structural_aliases": full_rank < len(labels), "rank_gap_vs_legal_space": full_rank-rank,
            "condition_number": float(np.linalg.cond(x)) if rank == len(labels) else None,
            "interpretation": "Constrained coverage is not an independent-factor orthogonal experiment. Use matched subspaces or coupled-block effects for aliased dimensions."}


def make_pool(name, capabilities):
    legal = []
    for row in itertools.product(*(VALUES[f] for f in FACTORS)):
        c = dict(zip(FACTORS, row))
        if is_valid(c, capabilities): legal.append(c)
    lookup = {key(c): i for i, c in enumerate(legal)}
    universe = sorted({(FACTORS[a], c[FACTORS[a]], FACTORS[b], c[FACTORS[b]]) for c in legal for a,b in PAIRS}, key=str)
    pair_id = {pair: i for i, pair in enumerate(universe)}
    masks = []
    for c in legal:
        mask = 0
        for a,b in PAIRS: mask |= 1 << pair_id[(FACTORS[a], c[FACTORS[a]], FACTORS[b], c[FACTORS[b]])]
        masks.append(mask)
    selected, anchors, excluded = [], [], []
    for source in DESIGN["anchors"]:
        original = {f: source[f] for f in FACTORS}
        c = dict(original)
        changes = []
        if c["calendar"] == "verified_day_week" and not capabilities["verified_calendar"]:
            c["calendar"] = "relative_daily_slot" if capabilities["relative_daily_slot"] else "none"
            changes.append("Source weekday/TOD replaced by relative daily-slot control; not original calendar semantics.")
        if c["node_encoding"] == "laplacian_position" and not capabilities["laplacian_positions"]:
            c["node_encoding"] = "none"; changes.append("Laplacian position disabled until validated.")
        reasons = violations(c, capabilities)
        if reasons:
            excluded.append({"model": source["model"], "reasons": reasons}); continue
        index = lookup[key(c)]
        if index not in selected: selected.append(index)
        anchors.append({"source_model": source["model"], "selected_index": index, "adaptations": changes,
                        "fidelity": source["fidelity"], "config": c})
    uncovered = (1 << len(universe)) - 1
    for index in selected: uncovered &= ~masks[index]
    while uncovered:
        best = max(range(len(legal)), key=lambda i: (masks[i] & uncovered).bit_count())
        new_count = (masks[best] & uncovered).bit_count()
        if not new_count: raise RuntimeError("Legal pair coverage stalled")
        selected.append(best); uncovered &= ~masks[best]
    rows = [legal[i] for i in selected]
    report = matrix_report(rows, legal)
    # Fill only a rank gap caused by sampling, never pretend to cure structural aliases.
    if report["rank_gap_vs_legal_space"]:
        def vector(c):
            return np.array([1.] + [float(c[f] == v) for f in FACTORS for v in VALUES[f][1:]])
        current = np.stack([vector(c) for c in rows]); current_rank = report["pool_rank"]
        for i,c in enumerate(legal):
            if i in selected: continue
            candidate = np.vstack((current, vector(c)))
            rank = int(np.linalg.matrix_rank(candidate))
            if rank > current_rank:
                selected.append(i); rows.append(c); current, current_rank = candidate, rank
            if current_rank == report["legal_space_rank"]: break
        report = matrix_report(rows, legal)
    result_rows = [{"recipe_id": f"components_v5_{name}_{i:03d}", **c, "execution_status": "symbolic_contract_only_adapter_validation_required"} for i,c in enumerate(rows)]
    pair_counts = {pair: sum(all(c[field] == value for field,value in ((pair[0],pair[1]), (pair[2],pair[3]))) for c in rows) for pair in universe}
    audit = {"profile": name, "capabilities": capabilities, "theoretical_combinations": math.prod(len(v) for v in VALUES.values()),
             "valid_combinations": len(legal), "pool_size": len(rows), "extendable_pair_count": len(universe),
             "covered_pairs": sum(count > 0 for count in pair_counts.values()), "minimum_pair_occurrences": min(pair_counts.values()),
             "pair_coverage_fraction": sum(count > 0 for count in pair_counts.values())/len(universe),
             "mechanism_anchors": anchors, "excluded_anchors": excluded, "main_effect_design": report,
             "prediction_scores_used": False, "new_training_runs": 0, "execution_ready": False}
    (PLAN / f"traffic_component_pool_v5_{name}.json").write_text(json.dumps(result_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    global PLAN
    PLAN = args.output
    PLAN.mkdir(parents=True, exist_ok=True)
    audit = make_pool('current_pems_metadata', DESIGN['capability_profiles']['current_pems_metadata'])
    (PLAN/'coverage.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({k:audit[k] for k in ('pool_size','valid_combinations','covered_pairs','extendable_pair_count')}))

if __name__ == '__main__': main()
