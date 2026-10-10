# 上游代码来源与本地改动

- 官方 GitHub：<https://github.com/vranlee/SU-T>
- 上游远程：`https://github.com/vranlee/SU-T.git`
- 归档基准提交：`479be89a27129f6879c62710a83abac6a8b261ee`
- 正式论文：见 `references/papers/AAAI2026_SU-T_official.pdf`。

`SU-T/` 是原仓库在该提交上的当前工作树快照，包含已经进行的本地源码修改；551 个上游跟踪文件均已逐文件校验。归档快照排除了 `.git`、MFT25 数据目录、`__pycache__` 和模型权重。`source-worktree.patch` 是原 `SU-T` 仓库相对上述基准提交的二进制安全 Git diff，可在干净 checkout 上审阅或应用；不含未跟踪的 Python 字节码。

模型权重已删除。原始 `SU-T/` 工作副本的本机删除操作被策略拦截，当前仍留在工作区；源码完整快照和差异补丁已存入本归档。MFT25 解压数据保留在仓库旁的 `SU-T-Data/mft25/`，档案内 `SU-T/datasets/mft25/` Junction 指向该处。源码差异摘要见 `source-changes.txt`。本机 A05 专用脚本与配置另在 `SU-T-Reproduction/`。
