# Changelog

## 2026.10.29

- Add deterministic candidate generation and compatibility screening.
- Add representative-task causal input construction, masked training,
  validation selection, checkpoint handling and native-unit scoring.
- Add source rules, locked configurations and source-only replay/calibration interfaces.
- Add the four-region target scores, source outer-holdout records and
  saved daily errors used to recompute reported comparisons and temporal intervals.
- Include the 28 October 2026 verification records: four synthetic predictor
  fits and 31 source-estimator checks; all 48 learned-rule choices match.
- Retain original public interfaces and three-region exports unchanged.

This release publishes the reviewed executable materials. It does not add
scientific experiments, alter frozen paper scores or claim full reproduction
of original-data training, nested calibration or GPU resource measurements.
Runtime-checked code is unchanged from Supplementary Material S1 version
2026-10-28-r2. Supplementary Material S1 now archives this repository version.

## Earlier public snapshot

Commit `213fc0f556cede6a88e6ccfe71a142285daa2105` supplied the core predictor
models, recipe configurations and three-region aggregate comparisons under MIT.
