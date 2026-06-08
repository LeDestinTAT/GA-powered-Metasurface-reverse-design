# 最终版本代码整理说明

这个目录是按“当前实际使用的代码主线”重新整理的代码包。

重要说明：原项目中的 `final/` 文件夹没有被默认视为最终版本。它只被放入 `02_matlab_comsol_reference/`，作为 MATLAB/COMSOL 仿真接口和历史参考代码保存。当前深度学习与优化主线以 `src/`、`scripts/`、`configs/`、`models/current/` 这一套项目结构为准。

## 目录结构

- `00_current_python_pipeline/`
  - 当前主要 Python 代码。
  - 包含模型定义、训练、推理、优化、论文图生成和配置文件。
- `01_model_and_data_references/`
  - 模型权重、数据集和训练结果的引用说明。
  - 大型 `.pt`、`.mat`、`.mph` 文件没有复制进来，只记录原始位置。
- `02_matlab_comsol_reference/`
  - 从原 `final/` 文件夹整理出的 MATLAB/COMSOL 相关脚本。
  - 不把这个目录命名为“最终版”，只作为仿真与版图生成参考代码。
- `03_legacy_fno_reference/`
  - 早期 standalone FNO 代码和 `runs_peak_nsga2_v2` 的参考结果。
  - 其中 `run_try2_current_dataset.py` 与 `current91` 数据有关，保留用于追溯迁移学习来源。
- `04_auxiliary_tools/`
  - 答辩、论文图和文档处理辅助脚本。

## 当前主线判断依据

当前版本的主要证据如下：

- `models/current/` 中最新权重为 `fno_try2_transfer_maxwell_best.pt` 和 `fno_try2_transfer_maxwell_final.pt`。
- 对应训练记录在 `outputs/train_runs/20260429-232953/train_summary.json`，模型族为 `try2_curve_field_transfer_v1`。
- 当前训练入口为 `scripts/train/train_fno_try2_transfer_maxwell.py`，PyCharm 包装入口为 `scripts/train/train_fno_try2_transfer_maxwell_pycharm.py`。
- 当前共享模型代码在 `src/fullfield_dual_surrogate.py`。
- 当前优化代码在 `scripts/optimize/run_optimize_dual.py` 和 `scripts/optimize/run_optimize_dual_closed_loop.py`。
- 当前论文图/章节图生成代码主要在 `scripts/tools/generate_paper_figures.py` 与 `scripts/tools/regen_ch*.py`。

## 推荐使用顺序

1. 需要展示最终算法代码：优先使用 `00_current_python_pipeline/`。
2. 需要说明训练和模型来源：参考 `01_model_and_data_references/`。
3. 需要展示仿真建模或版图生成：使用 `02_matlab_comsol_reference/`。
4. 需要说明早期 FNO 或 current91 数据来源：使用 `03_legacy_fno_reference/`。
5. 需要论文图、答辩或文档工具：使用 `04_auxiliary_tools/`。

## 运行提示

这些代码原本依赖项目根目录下的数据、模型、输出目录。为了真实运行，建议仍在原项目根目录运行原始路径中的脚本；本目录主要用于代码整理、提交、查阅和备份。

例如当前优化主线：

```powershell
cd D:\field_batch_output_compressed_air
python scripts\optimize\run_optimize_dual.py --config configs\optimize\nsga2_dual_fullfield_precise.json
```

当前迁移学习训练主线：

```powershell
cd D:\field_batch_output_compressed_air
python scripts\train\train_fno_try2_transfer_maxwell.py
```

