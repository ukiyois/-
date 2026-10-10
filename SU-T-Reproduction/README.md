# SU-T ReID 本地复现记录

本目录记录作者公开权重下的 SU-T-ReID 推理复现，包括运行配置、日志、预测文件和本地跟踪指标。由于 MFT25 测试集标注不可见，本地指标使用训练视频各自的 `val_half` 时间段，只用于方法与代码核验，不能当作论文测试集分数。

## 复现范围

- 不重新训练模型，直接加载作者提供的检测器与 ReID 权重。
- 检测权重 `Pretrained/Checkpoint.pth.tar` 和 ReID 权重 `Pretrained/ReID-Checkpoint.pth` 曾用于本次评测；两者 SHA-256 记录在运行元数据中，文件已在评测完成后从本机删除。ReID 推理会跳过训练分类头 `heads.weight`，其类别维度不同属预期，其余权重加载正常。复跑推理需按 `Pretrained/README.md` 重新获取权重。
- 使用作者的 `SU-T-ReID` 配置和 MFT25 跟踪器实现。
- 在 MFT25 训练集的 8 段视频上评估 `val_half`：BT-001、BT-003、BT-005、MSK-002、PF-001、SN-001、SN-013、SN-015。
- 保存作者入口生成的逐帧跟踪结果，并使用 TrackEval 计算 HOTA、DetA、AssA、MOTA、IDF1 等指标。

## 运行约束

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，显存 8151 MiB。
- 推理采用单卡、batch size 1、FP16，不启用模型融合；输入尺寸保持作者配置的 800 × 1440。
- Python 3.10 主环境；PyTorch 2.10.0+cu128，CUDA 可用。

## 文件布局

- `configs/`：本地验证配置快照。
- `data/`：验证标注、数据清单和数据映射。
- `logs/`：完整终端日志与环境记录。
- `metrics/`：汇总指标和逐序列指标。
- `runs/`：作者评测入口写出的 MOT 格式跟踪结果。

原始 MFT25 压缩包硬链接保存在档案 `SU-T/MFT25/`，解压后的图像当前位于仓库旁的 `SU-T-Data/mft25/`，并通过 `SU-T/datasets/mft25/` Junction 访问。不复制整套原始图像到 Git；数据清单会记录来源、校验信息和路径。

## 结果解释

论文表中的 SU-T† 测试集结果为 HOTA 34.067、IDF1 44.643。这里使用的是训练视频的后半段，数据分布、训练视频身份与隐藏测试集都不同，因此本地分数仅用于核验评测链路和实现行为，不与论文数字直接比较。

## 全量核验结果（A03）

- 状态：已完成。使用 8 段视频、16,035 帧和 132,057 个标注目标；没有重新训练。
- 作者评测入口检测 AP：AP@[0.50:0.95] = 0.579，AP50 = 0.922，AP75 = 0.644。
- TrackEval 1.3.0：HOTA = 50.19，DetA = 69.92，AssA = 36.23，MOTA = 85.53，IDF1 = 57.27，IDSW = 391。
- 平均耗时：模型前向 155.85 ms/帧，跟踪 12.46 ms/帧，合计 168.31 ms/帧；全程约 46 分 22 秒。
- 运行期间抽查显存约 3.2–3.4 GiB、GPU 温度约 73–86°C；全程无显存溢出，推理正常结束。
- 完成的八个 MOT 结果文件都通过 10 列格式、帧号范围和末帧覆盖检查。原始预测在 `runs/SU-T-ReID-ValHalf-A03/`，TrackEval 原始结果与分序列报告在 `metrics/trackeval/`。
- `metrics/run_summary.json` 汇总硬件、依赖、权重哈希、运行配置、耗时和最终指标；`metrics/sequence_summary.csv` 保存综合及逐序列指标；`metrics/trackeval_summary.json` 保留 TrackEval 原始指标结构。
- `logs/sut_reid_valhalf_attempt03_console.log` 保存完整作者入口输出；`logs/trackeval_valhalf_a03_console.log` 保存完整 TrackEval 评分日志。

运行日志中的进度条最后一次刷新显示 15,828/16,035，但作者循环随后在最后一批写出 SN-015 并完成 AP 评估。八个结果文件都覆盖到各自 `val_half` 的最后帧，最终检测与 TrackEval 汇总也成功完成，因此这是进度条显示滞后，不是漏跑帧。
