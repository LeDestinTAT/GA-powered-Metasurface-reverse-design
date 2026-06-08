# 文件清单

## 00_current_python_pipeline

### src

- `src/__init__.py`
- `src/checkpoint_utils.py`
- `src/fullfield_dual_surrogate.py`
- `src/material_dispersion.py`
- `src/project_paths.py`

### scripts/train

- `scripts/train/run_fullfield_train_background.bat`
- `scripts/train/train_fno_curve_only_pycharm.py`
- `scripts/train/train_fno_curvefield_hybrid.py`
- `scripts/train/train_fno_field_maxwell_pycharm.py`
- `scripts/train/train_fno_fullfield_maxwell_pycharm.py`
- `scripts/train/train_fno_fullfield_peakfocus.py`
- `scripts/train/train_fno_maxwell.py`
- `scripts/train/train_fno_try2_transfer_freeze_maxwell_pycharm.py`
- `scripts/train/train_fno_try2_transfer_maxwell.py`
- `scripts/train/train_fno_try2_transfer_maxwell_pycharm.py`

### scripts/infer

- `scripts/infer/infer_pattern_to_field_absorption_pycharm.py`
- `scripts/infer/predict_plot_fullfield_pycharm.py`

### scripts/optimize

- `scripts/optimize/run_optimize2.py`
- `scripts/optimize/run_optimize_dual.py`
- `scripts/optimize/run_optimize_dual_closed_loop.py`

### scripts/tools

- `scripts/tools/build_curve_dataset_cache_pycharm.py`
- `scripts/tools/compare_curve_vs_maxwell_metrics.py`
- `scripts/tools/export_curve_cache_to_legacy_fno_mats.py`
- `scripts/tools/generate_background_need_slide_figure.py`
- `scripts/tools/generate_convergence_slide_figure.py`
- `scripts/tools/generate_multi_solution_slide_figure.py`
- `scripts/tools/generate_paper_figures.py`
- `scripts/tools/generate_try2_optimization_figures.py`
- `scripts/tools/generate_two_model_spectra_compare.py`
- `scripts/tools/list_best_checkpoints_pycharm.py`
- `scripts/tools/regen_ch1_overview.py`
- `scripts/tools/regen_ch4_encoding.py`
- `scripts/tools/regen_ch4_nsga2_flow.py`
- `scripts/tools/regen_ch4_pareto.py`
- `scripts/tools/regen_ch5_closed_loop.py`
- `scripts/tools/regen_ch5_convergence.py`
- `scripts/tools/regen_ch5_multi_solution.py`

### configs

- `configs/optimize/nsga2_dual_fullfield_example.json`
- `configs/optimize/nsga2_dual_fullfield_precise.json`
- `configs/optimize/nsga2_sparam_example.json`

## 01_model_and_data_references

- `MODEL_DATA_REFERENCES.md`

## 02_matlab_comsol_reference

- `bitmap_to_sim_Ver1.m`
- `bitmapbuild.m`
- `bitmaps_to_sim_Ver2.m`
- `bitmaptrans.m`
- `final.m`
- `jointogether.m`
- `testFullfield.m`
- `optimized_patterns/best_matrix_from_runs_peak_nsga2_v2.m`

## 03_legacy_fno_reference

- `config.json`
- `curveloss.py`
- `curvepredict.py`
- `fullcurve.py`
- `little_tool.py`
- `mat_to_curve.py`
- `mat_to_curvePredict.py`
- `pointloss.py`
- `possible_best.py`
- `possible_best_predict.py`
- `predict.py`
- `readme`
- `run_optimize.py`
- `run_optimize2.py`
- `run_try2_current_dataset.py`
- `try1 pre.py`
- `try1.py`
- `try2.py`
- `runs_peak_nsga2_v2/best_matrix.m`
- `runs_peak_nsga2_v2/best_report.json`
- `runs_peak_nsga2_v2/pareto_summary.json`
- `runs_peak_nsga2_v2/progress.json`

## 04_auxiliary_tools

- `combine_figures.py`
- `fill_defense_application.py`
- `paper_examples/utils.py`
