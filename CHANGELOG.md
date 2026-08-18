# 3.4.1

## Changed
* Bundled XGBoost moved to **upstream v3.4.1**. The submodule pointed at a fork
  (`agene0001/xgboost@serving-patches-3.3.0`) carrying three serving hot-path patches;
  all three are now in upstream verbatim, so the fork is dropped and `.gitmodules`
  points at `dmlc/xgboost`:
  - cached the `MakeDeviceOrd` regexes and added a `device=cpu` fast path
  - skipped the per-tree depth walk for single-row prediction
  - removed redundant allocations in the JSON text parser
* Crate versions follow the bundled XGBoost: 3.4.1.
* `xgboost-sys` is now a workspace member and in `default-members`, so its own tests
  run under `cargo test`. They never had, which is how a broken `read_matrix` test
  calling the deprecated `XGDMatrixCreateFromFile` went unnoticed.

## Fixed
* Replaced the last use of `XGDMatrixCreateFromFile`, which XGBoost 3.4 removes, with
  `XGDMatrixCreateFromURI`.
* Prediction and logloss assertions that used a one-sided `assert!(a - b < eps)` are now
  two-sided, so they can actually catch downward numeric drift on a version bump.

## Notes on upstream 3.4 behaviour
* Split-gain values reported by `dump_model` shift in the last few ulps (12 fields, at
  most 6.2e-07 relative). 3.3 had a closed-form fast path for split gain in
  `TreeEvaluator::CalcGainGivenWeight`, used when `max_delta_step == 0` and there are no
  monotone constraints, explicitly to reduce average floating-point error; 3.4 removed it
  when unifying the single- and multi-target split evaluators and now computes the gain
  from the leaf weight instead. Same value algebraically, different rounding. Tree
  structure, leaf weights, covers, predictions and metrics are all unchanged.
* `min_child_weight` no longer gates the gain and weight calculation itself (upstream
  #12322); it is enforced as a split constraint. No effect on the tests here, but it is
  a behavioural change upstream flagged as breaking.
* `booster=gblinear` and `booster=dart` remain available (still deprecated, dart still
  remapped to gbtree). Column-split support was removed upstream; this crate never
  exposed it.

# 3.0.6 (Unreleased)

## Fixed
* Fixed deprecation warnings from XGBoost C API:
  - Replaced `XGBoosterBoostOneIter` with `XGBoosterTrainOneIter`
  - Replaced `XGDMatrixCreateFromCSREx` with `XGDMatrixCreateFromCSR`
  - Replaced `XGDMatrixCreateFromCSCEx` with `XGDMatrixCreateFromCSC`
  - Replaced `XGDMatrixSetUIntInfo` with `XGDMatrixSetInfoFromInterface`
  - Replaced `XGDMatrixCreateFromFile` with `XGDMatrixCreateFromURI`

## Added
* Added `BinaryError` variant to `EvaluationMetric` for default 0.5 threshold (simpler alternative to `BinaryErrorRate(0.5)`)
* Added `validate_features()` method to `Booster` for checking feature name/count consistency
* Added callback support to `TrainingParameters`:
  - New `CallbackEnv` struct with iteration info and evaluation results
  - New `TrainingCallback` type for callback functions
  - Callbacks can return `false` to stop training early

## Changed
* `Booster::update_custom()` now requires an `iteration` parameter
* Added safety documentation for `Booster::new_with_cached_dmats()` explaining DMatrix lifetime

# 0.1.4 (2019-03-05)

* `Booster::load_buffer` method added (thanks [jonathanstrong](https://github.com/jonathanstrong))
