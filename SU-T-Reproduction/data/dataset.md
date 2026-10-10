# 数据集来源、拆分与本机文件

## 来源

- 数据集：Multiple Fish Tracking 2025 (MFT25)。
- 官方 Hugging Face 数据页：<https://huggingface.co/datasets/Vranlee/MFT25>。
- 论文和项目来源链接见 [AAAI 论文](https://ojs.aaai.org/index.php/AAAI/article/view/37574) 与 [作者项目页](https://vranlee.github.io/SU-T/)。
- Hugging Face 卡片标示 Apache-2.0；下载数据文件前需登录并同意页面列出的条件（分享联系信息）。以下载时数据页的许可和条款为准。

## 本次用到的部分

评测用的是作者训练集的 `val_half`，不是论文隐藏测试集：8 段训练视频后半段、16,035 帧、132,057 个 GT 框。逐序列帧数和核对信息见 `dataset_manifest.json`；转换后的 COCO 标签在 `annotations/val_half.json`，MOT 格式 GT 在 `ground_truth/`，序列表在 `seqmap.txt`。`val_half` 是作者提供的时间切分标签，序列与论文附录所列测试视频不同。

MFT25 测试集标注不可见，本档案没有访问或生成官方榜单结果。不得把此处验证结果称为论文测试集复现。

## 当前工作区中的数据

- `../../SU-T/MFT25/MFT25-train.zip`（从 `SU-T-Reproduction/data` 相对路径计）是本机原压缩包的硬链接。大小 1,199,319,979 bytes，SHA-256 `ab8d90cfeb96cfcdbb7d43d753cc50158097e51d4703a713000de222852b4ad3`。
- `../../SU-T/datasets/mft25/` 是本地 Junction，指向仓库旁的 `SU-T-Data/mft25/`。数据目录和压缩包均不进入 Git。
- `detection_cache_A05/` 是同一个作者检测器在 `val_half` 上生成的一次冻结 post-NMS 检测结果缓存；七种 tracker 共用。
- 标签、GT、缓存及预测在本 Git 仓库中保留；完整帧图像和原始大压缩包不进入 Git 历史。

重新获取数据后，将压缩包放入档案根目录 `SU-T/MFT25/MFT25-train.zip`，解压到仓库旁的 `SU-T-Data/mft25/`，并将 `SU-T/datasets/mft25/` Junction 指向该目录。`prepare_val_half.py` 可根据该压缩包重建验证标签和 GT；本次已使用的文件快照仍已随档案跟踪。
