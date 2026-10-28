# Traffic recipe screening

Code and aggregate results for **“Task-conditioned recipe screening for traffic forecasting at native sampling intervals.”**

Version: **2026.10.29** (Git tag **v2026.10.29**). Author: [Jianing Long](https://orcid.org/0009-0000-0074-7786). Code licence: [MIT](LICENSE). See [CITATION.cff](CITATION.cff), [VERSION.json](VERSION.json) and [CHANGELOG.md](CHANGELOG.md). Supplementary Material S1, `Supplementary_S1_v2026.10.29.zip`, is an archive of this same version.

This package makes the decision chain inspectable: configuration and input feasibility, a representative task's causal histories and valid targets, predictor fitting, validation selection, source recommendations, and saved-score evaluation. Source rules are compared in parallel. S09 is a display name with no priority status. This release adds the executable workflows and complete four-region exports to the earlier public snapshot. The original scientific experiments, predictor implementations, candidate definitions and recorded scores are unchanged. The earlier commit `213fc0f556cede6a88e6ccfe71a142285daa2105` remains in Git history; provenance is recorded in `docs/PROVENANCE.json`.

Use Python 3.10 or later from the package root. Install `requirements-analysis.txt` for all no-fit commands below. The original analysis environment used NumPy 1.26.4. Learned source rules additionally use `requirements-source.txt` (scikit-learn 1.4.2, SciPy 1.13.1, joblib 1.4.2). The training entry point uses `requirements-training.txt`, with PyTorch 2.6.0; original GPU fits used PyTorch 2.6.0+cu126. Choose the corresponding PyTorch wheel for your hardware. No provider observation data or trained weights are included.

## Inspect and recompute without fitting

```text
python scripts/screen_candidates.py
python src/generate_candidates.py --output generated
python scripts/replay_source_rules.py --output source_replay.json
python scripts/recompute_scores.py --output recomputed.json
python scripts/representative_task.py demo --output example.npz
python scripts/representative_task.py dry-run --input example.npz --output interface_check.json
python scripts/representative_task.py dry-run --input example.npz --horizon-minutes 60 --rate-factor 4 --output interface_check_60.json
```

The generator uses the original deterministic legal enumeration, anchor adaptation, pair coverage and rank-completion functions. It regenerates the original 27 configurations under the declared capability profile without reading scores. The screening script applies the no-supplied-graph and physical-history checks, leaving 17 neural candidates. Filtering the original design does not imply complete pair coverage of the remaining legal space.

The default source replay executes the frozen fixed and nearest-neighbour rules in both pools at all three meta-seeds, checking all eight target recommendations. It performs zero estimator fits. `source_data.json` contains 20 labelled source tasks, their aggregate errors and descriptors, and eight target tasks with training-prefix descriptors only. No target validation/test label is present in that input. Target scores are kept separately. All 14 rules' selected configurations, both pools and all meta-seeds are in `configs/locked_source_rules.json`.

The score command expands a simple method's one effect only for alignment with three predictor seeds. It computes task/seed paired error ratios before averaging, reports the 17-neural and 23-candidate validation choices separately, recomputes source outer-holdout summaries and the saved-daily-error time-block sensitivity, and checks the supplied full-precision expected values. The 504 unique target rows comprise 456 neural/external fits and 48 simple effects. Neither that expansion nor three meta-seeds creates independent regions. Time-block intervals condition on the saved predictors, selected recipes, regions and test week; they are not population confidence intervals or significance tests.

## Execute learned source rules

After installing the source dependencies, these explicit commands fit source estimators from the aggregate source corpus and use the original locked configurations:

```text
python scripts/replay_source_rules.py --methods S07_FFORMS S09 --output learned_replay.json
python scripts/replay_source_rules.py --methods S04_direct S05_matched S08_ID --output other_rules.json
```

Any of the 14 method names in the configuration file is accepted. Add `--recalibrate` to perform the original finite source-only configuration search, using the flow-only region-balanced objective and the original worst-region criterion. This can be expensive. The functions retain source-region preprocessing, candidate identity/description representations, parent-baseline comparisons and delete-source dispersion. `docs/PROVENANCE.json` records the extraction and adaptations. The package's original scientific result exports are never replaced by these runs. Estimator-version differences can change predictions, so use the pinned source requirements when comparing locked choices.

## Representative task with author-supplied observations

Prepare an NPZ file with three arrays: `raw` as float values with shape `[2880, N]` in native units and provider-reported 15-minute order, `input_available` as a boolean array of that shape, and a separate boolean `target_valid` array. Invalid targets may be NaN but valid targets must be finite and nonnegative. `demo` above creates a synthetic format example only. For provider acquisition and preprocessing, see `docs/DATA.md`.

The representative panel fixes the information prefix at days [0,16), supervised targets at [7,16), validation at [16,23), and test at [23,30). It freezes per-node population moments on input-available prefix observations, carries inputs forward for at most four ticks and otherwise uses the frozen mean. Targets are never filled. Recent histories and corresponding day/week segments precede each origin. Both horizons share the four-step boundary restriction. `rate-factor 4` converts Victorian native counts to hourly rates internally, then restores native units for scoring; other panels use factor one.

Run training only when desired, after installing the training requirements:

```text
python scripts/representative_task.py train --input prepared_panel.npz --output local_run --recipes 004 005 --horizon-minutes 15 --seed 20261006 --device cpu
python scripts/representative_task.py train --input prepared_panel.npz --output local_full_pool --recipes all --include-simple --horizon-minutes 60 --rate-factor 4 --device cuda
```

The first command demonstrates neural candidate selection. The second includes all 17 feasible recipes and six simple candidates. Each output directory must be new. Models use the copied canonical parameter initialisation, per-origin stable forward path and masked backward function. The protocol has 180 rounds, 336 sampled origins per round, logical batch 16, Adam, cosine rate schedule, norm-one clipping and validation checkpoint eligibility at rounds 65–180. The best eligible checkpoint is retained; recipe selection reads validation MAE only, followed by selected-model test scoring in nonnegative native units. Simple candidates preserve the documented persistence, day/week averaging and shared-node, per-output Ridge definitions; Ridge alpha is selected from 0.1, 1, 10, 100 by validation MAE.

The representative orchestration is a portable extraction of the original workflow, not the original manifest/resume/GPU scheduling system. It omits external baseline implementations, source-provider parsers, hardware admission, intermediate resume checkpoints, complete raw-data provenance and the resource-timing harness. It does not promise bitwise reproduction of published scores or reproduce recorded workflow times. The original protected experiment files are unchanged.

## Verification status of this revision

`docs/VALIDATION.json` records the original no-fit checks and the runtime checks added in this version. `docs/RUNTIME_VERIFICATION.json` gives their exact scope. An existing Python 3.12.4 environment with NumPy 1.26.4, PyTorch 2.6.0+cu126, scikit-learn 1.4.2, SciPy 1.13.1 and joblib 1.4.2 executed the checks; predictor verification used CPU.

The predictor checker executes the actual portable training loop for 004 and 005 at both 15 and 60 minutes on a three-node synthetic panel. It explicitly shortens the schedule to two rounds with six training origins and four origins each for validation and test, producing four synthetic predictor fits. It checks finite forward/backward execution, parameter updates, checkpoint saving/loading, validation-only selection, exact prediction agreement after reload, and native-unit test rescoring. The normal `train` command still reads the unchanged 180-round protocol. These synthetic scores are not paper results.

Locked S07 and S09 replay on the existing source corpus matches all 48 target choices across three meta-seeds in the full pool, using 30 estimator fits. One additional estimator checks a 16-task outer training fold with the held-out region excluded. Preprocessing task counts are also checked for all eight outer splits: 16 tasks for Essen/Torino holdouts and 18 otherwise. These checks do not execute the full nested calibration grid or all 14 learned rules.

To repeat the minimal checks with the training and source dependencies installed:

```text
python scripts/runtime_smoke.py predictor --output runtime_predictor
python scripts/runtime_smoke.py source --output runtime_source
```

Use new output directories. The check scripts do not overwrite any supplied scores, configurations or original experiment records. The full 180-round scientific protocol, original provider-data training, published GPU scores and resource timings have not been independently reproduced by these checks. This revision must not be described as full experimental reproduction.

Code is under the accompanying MIT License. Third-party source data and external implementations retain their own terms. `SHA256SUMS.txt` covers package files other than itself. `docs/PROVENANCE.json` distinguishes byte-identical model files from adapted orchestration and sanitised aggregate exports.

## Earlier public interfaces

The original `scripts/recompute_cross_city.py`, `scripts/smoke_models.py`,
`requirements-model.txt`, `results/cross_city_scores.csv` and
`results/locked_recommendations.csv` are retained unchanged for compatibility.
Those result files cover the earlier three-region scope. Use
`results/four_region_scores.csv`, `configs/locked_source_rules.json` and
`scripts/recompute_scores.py` for the current four-region paper materials.
The five original predictor/dependency files and `configs/recipes.json` are
byte-identical to the earlier public commit.
