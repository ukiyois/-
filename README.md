# SU-T 论文与本地复现档案

本目录把论文、上游代码快照、MFT25 数据来源、本地评测方案、代码改动、完整可保留的运行记录和 A05 固定检测消融结果放在一起。目录已初始化为独立 Git 仓库。

## 本地评测结论范围

A05 在 MFT25 作者提供的训练视频 `val_half` 上运行，8 段视频、16,035 帧；检测器只推理一次，缓存 148,250 个检测框，之后所有跟踪器读取同一批检测框并由同一个 TrackEval 1.3.0 配置评分。比较包括 KF+IoU、UKF+IoU、KF+FishIoU、UKF+FishIoU、加入 ReID 的 SU-T、ByteTrack 和 OC-SORT。

这是用于核验组件增量的本地受控验证对照。它不是论文隐藏测试集复现、官方榜单结果或独立留出泛化结论。用户明确选择不训练、不提交结果；论文测试集的标注不可见。

阅读顺序：先看 [`SU-T-Reproduction/复现总结.md`](SU-T-Reproduction/复现总结.md) 了解本地复现结论，再看 [`references/paper_reading.md`](references/paper_reading.md) 了解论文贡献、方法、实验和局限。

## 内容导航

- `references/papers/`：AAAI 正式出版版 PDF、作者扩展 arXiv 版本及来源说明。
- `references/bibliography.bib`：论文引用条目。
- `UPSTREAM.md`：GitHub 仓库地址、基准提交和本地工作树快照说明。
- `source-worktree.patch`：从基准提交到本地已改代码的补丁；完整当前源码快照位于 `SU-T/`。
- `SU-T/`：可读的上游代码快照，包含运行所需代码和本地源码改动；不含 `.git`、权重与图像数据。
- `SU-T-Reproduction/固定检测消融_A05报告.md`：完整结果与逐模块解读。
- `SU-T-Reproduction/评测方案_冻结权重消融.md`：预先约定的比较条件与边界。
- `SU-T-Reproduction/choices_and_process.md`：复现选择和过程决策记录。
- `SU-T-Reproduction/logs/`：已有原始日志，以及 A05 阶段记录。
- `SU-T-Reproduction/data/`：`val_half` 标签、数据清单、检测缓存及检测权重/依赖运行元数据。
- `SU-T-Reproduction/runs/fixed_detection_A05/`：七个配置的 MOT 格式逐视频预测文件。
- `SU-T-Reproduction/metrics/fixed_detection_A05/`：TrackEval 原始输出和汇总指标。
- `SU-T/MFT25/MFT25-train.zip`：本机 MFT25 训练集压缩包硬链接（Git 忽略）。
- `SU-T/datasets/mft25/`：本机已解压数据的 Junction（Git 忽略，指向原工作区的解压目录）。
- `Pretrained/`：仅有权重放置说明；不含权重，目录内权重文件均被 Git 忽略。

## 数据与权重

本地压缩包大小为 1,199,319,979 bytes，SHA-256 为 `ab8d90cfeb96cfcdbb7d43d753cc50158097e51d4703a713000de222852b4ad3`。压缩包以同卷硬链接纳入目录，不重复占用 1.2 GB 磁盘空间；Git 不跟踪此压缩包或解压图像。验证标注、GT、冻结检测缓存和预测文件在 Git 中跟踪，足以检查本次对照输入与输出。

检测器和 ReID 权重都没有复制进此目录，也不会进入提交。可在 `Pretrained/README.md` 查看原本运行脚本使用的文件名和上游下载位置。运行元数据只记文件名、原工作区路径和 SHA-256，不含权重内容。

## 打开 / 评分

如果只想阅读结论，先看 A05 报告。如果要在有 Python 3.10、TrackEval 1.3.0、同一 MFT25 文件及作者权重的本机环境重算评分，请在本目录运行 `python SU-T-Reproduction/score_fixed_detection_ablation.py`。若要重新推理，还需要先把权重放到 `Pretrained/` 并确保 `SU-T/datasets/mft25/` 指向已解压的 MFT25 图像；推理和回放脚本及参数快照均已归档。归档时的运行环境记录在 `SU-T-Reproduction/data/run_metadata.json`。

## Git

此目录是独立 Git 仓库；原来的 `SU-T` 工作区未被重置或改写。`.gitignore` 排除权重、原始数据、图片目录、缓存字节码和虚拟环境。首次提交只包含来源、代码、文本记录、实验输入缓存、预测及指标，不包含权重或原始图片。
