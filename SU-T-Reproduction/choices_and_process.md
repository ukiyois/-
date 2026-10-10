# 复现选择与过程记录

## 研究对象选择

SU-T 发表在 AAAI 2026 主会正式技术轨，满足此前确定的“近五年、知名期刊/主会、不选 workshop”的范围。正式版 PDF 和作者扩展版 PDF 都在 `references/papers/`。项目使用 YOLOX 检测器和现成 ReID 网络；贡献集中在鱼类多目标跟踪系统与关联/运动处理，不是重新设计一个检测网络骨干。

## 数据集核验

作者公开 MFT25 训练压缩包含有 `gt_train_half` 和 `gt_val_half`。由于公开代码中的验证配置指向无标签的测试配置，且测试集 GT 不公开，不能从隐藏测试集重新计算论文数字。用户确认不提交外部平台结果后，确定只用本地 `val_half` 作方法核验，并明确结果不冒称官方测试复现。

数据构建把作者的 `val_half` 转成可供当前评测入口使用的 COCO 标注和 MOT GT；记录了各序列帧号映射、帧数和框数，并核对其与原始标注。数据拆分详见 `data/dataset_manifest.json` 和 `data/dataset.md`。

## 用户确定的边界

- 不训练：不重训 detector、ReID 或 tracker。
- 不提交：不上传或提交到 Codabench / leaderboard。
- 比较方法组件时固定同一检测器、同一批检测框和同一 TrackEval 评测代码。
- 报告为本地 `val_half` 受控消融，论文测试集和本地验证集不直接排分。
- 归档应有过程记录和可检查的修改代码，不放模型权重。

## A05 比较组

| 变体 | 运动模型 | 关联度量 | ReID | 目的 |
|---|---|---|---|---|
| KF_IoU | KF | IoU | 关 | SU-T 内部对照锚点 |
| UKF_IoU | UKF | IoU | 关 | 单独看 UKF 相对 KF 的变化 |
| KF_FishIoU | KF | FishIoU | 关 | KF 条件下单独看 FishIoU |
| UKF_FishIoU | UKF | FishIoU | 关 | SU-T 组件组合基线 |
| UKF_FishIoU_ReID | UKF | FishIoU | 开 | 单独看 ReID 相对组合基线的变化 |
| ByteTrack | tracker 原生关联 | tracker 原生关联 | 关 | 同检测输入下的系统级外部参考 |
| OCSort | tracker 原生关联 | tracker 原生关联 | 关 | 同检测输入下的系统级外部参考 |

对照实现把原来默认 UKF 的 SU-T Kalman tracker 增加 `motion_filter` 选择；KF 采用相同 9 维状态、5 维观测及相同 F/H/P/Q/R 初始化，改动位置在 `SU-T/trackers/SUT_UKF.py`，完整源码差异在根目录 `source-worktree.patch`。其余关联设置沿用当前作者代码和冻结参数。

原仓库没有可直接复用的 HybridSORT 跟踪器实现，所以外部比较选择仓库内已有、能消费同一检测输入的 ByteTrack 和 OC-SORT。它们用于系统级参考，不作为 SU-T 单个模块的因果消融。

## 执行记录

1. 保留环境：本机 Python 3.10.9 主环境，不新建虚拟环境；CUDA 可用，NVIDIA GeForce RTX 5060 Laptop GPU。具体包版本见 `data/run_metadata.json`。
2. 由 detector 对作者 `val_half` 图像只运行一次，按 800×1440、置信度 0.1、NMS 0.7 做后处理，保存 148,250 个检测框和序列映射在 `data/detection_cache_A05/`。
3. 所有跟踪配置读取同一个逐帧缓存，包括空检测帧；生成七组预测到 `runs/fixed_detection_A05/`。
4. 所有输出由同一 TrackEval 1.3.0 配置评估，指标与逐序列输出在 `metrics/fixed_detection_A05/`。
5. 与先前作者入口 A03 全量 ReID 核验交叉检查：A05 的 UKF+FishIoU+ReID 汇总值接近此前独立路径结果；差异详见 A05 报告。

## 结果解释约束

以 A05 报告的 pooled 与逐视频 macro 指标为准。UKF 在 IoU 条件下是小幅变化；FishIoU 的总体方向依序列加权而异；ReID 的整体身份关联指标改善最清楚但也不覆盖所有视频；同检测输入下 ByteTrack 的 HOTA/IDF1 高于 SU-T+ReID。报告保留了这些反向与混合结果，不把它们改写成论文结论。

若 detector / ReID checkpoint 的训练数据范围无法独立确认，就不能把验证分数解释为无泄漏的泛化测评。完整数据、权重和隐藏测试标注的限制见报告与 `data/dataset.md`。
