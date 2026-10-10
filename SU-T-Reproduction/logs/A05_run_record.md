# A05 固定检测消融运行记录

- 状态：检测缓存完成、七组 tracker 回放完成、TrackEval 评分完成。
- 日期：2026-10-10（本机环境；具体包与设备信息见 `../data/run_metadata.json`）。
- 数据：作者 MFT25 train `val_half`，8 序列、16,035 帧、132,057 GT 框。
- 检测：使用作者 detector，在输入 800×1440、置信度 0.1、NMS 0.7 下只推理一次，生成 148,250 个 post-NMS 检测框；检测缓存及每帧序列映射见 `../data/detection_cache_A05/`。
- 回放：七个 tracker 配置读取同一个检测缓存；每配置八个预测文件，见 `../runs/fixed_detection_A05/` 和 `replay_manifest.json`。
- 评分：统一 TrackEval 1.3.0；完整 A05 评分终端日志为 `trackeval_fixed_detection_A05.log`，原始指标和逐视频细节在 `../metrics/fixed_detection_A05/`。
- 结果：见 `../固定检测消融_A05报告.md`。

记录完整性说明：A05 检测缓存与 tracker 回放阶段的原始逐行 stdout 当时未落盘为独立日志。这里没有伪造逐行日志；运行状态以实际生成的 `cache_manifest.json`、`replay_manifest.json`、八序列预测文件及完整 TrackEval 评分日志交叉核验。早期 A03/调试阶段保留的原始日志、含失败尝试，仍原样归档在本目录。

## 可追踪脚本

- `../cache_detector_valhalf.py`：冻结 detector，生成一次检测缓存。
- `../replay_fixed_detections.py`：在同一检测缓存上执行 A05 的七个 tracker 变体。
- `../score_fixed_detection_ablation.py`：校验预测帧范围并用统一 TrackEval 评分。
- `../prepare_val_half.py`：由 MFT25 源 zip 重建验证拆分清单与 GT。
