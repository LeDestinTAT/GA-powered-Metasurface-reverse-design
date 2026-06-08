# 模型和数据引用说明

本目录只记录大型二进制文件的位置，不复制模型权重、数据集和 COMSOL 文件。

## 当前模型权重

- `D:\field_batch_output_compressed_air\models\current\fno_try2_transfer_maxwell_best.pt`
- `D:\field_batch_output_compressed_air\models\current\fno_try2_transfer_maxwell_final.pt`
- `D:\field_batch_output_compressed_air\models\current\fno_fullfield_maxwell_dual_best.pt`
- `D:\field_batch_output_compressed_air\models\current\fno_fullfield_maxwell_dual_final.pt`
- `D:\field_batch_output_compressed_air\models\current\fno_curve_only_best.pt`
- `D:\field_batch_output_compressed_air\models\current\fno_curve_only_final.pt`

## 最新迁移学习训练记录

- 训练记录：`D:\field_batch_output_compressed_air\outputs\train_runs\20260429-232953\train_summary.json`
- 模型族：`try2_curve_field_transfer_v1`
- 初始化权重：`D:\field_batch_output_compressed_air\final\fno_peak_curve_best_current91.pt`
- 最佳 epoch：20
- 最佳验证损失：0.432894563768059

## 主要数据路径

- 全场数据目录：`D:\field_batch_output_compressed_air\data\field_batch_output_compressed_air\`
- 采样元数据：`D:\field_batch_output_compressed_air\data\field_batch_output_compressed_air\sampling_meta.mat`
- 曲线缓存目录：`D:\field_batch_output_compressed_air\data\curve_cache\`

## 历史或迁移学习依赖

以下文件位于原 `final/` 文件夹，但只作为历史依赖或初始化来源，不代表整个 `final/` 文件夹都是最终代码：

- `D:\field_batch_output_compressed_air\final\fno_peak_curve_best_current91.pt`
- `D:\field_batch_output_compressed_air\final\fno_peak_curve_final_current91.pt`
- `D:\field_batch_output_compressed_air\final\Sparams_dataset_current91.mat`
- `D:\field_batch_output_compressed_air\final\training_patterns_11x11_current91.mat`

## COMSOL / MATLAB 大文件

原 `final/` 文件夹中存在 `.mph`、`.mat` 等大文件，未复制到代码包：

- `D:\field_batch_output_compressed_air\final\model_selective_squares_field_sampling_final.mph`
- `D:\field_batch_output_compressed_air\final\model_selective_squares_only.mph`
- `D:\field_batch_output_compressed_air\final\field_dataset_single_sample.mat`

