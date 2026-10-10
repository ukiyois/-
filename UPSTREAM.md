# 上游代码来源与本地改动

- 官方 GitHub：<https://github.com/vranlee/SU-T>
- 上游远程：`https://github.com/vranlee/SU-T.git`
- 归档基准提交：`479be89a27129f6879c62710a83abac6a8b261ee`
- 正式论文：见 `references/papers/AAAI2026_SU-T_official.pdf`。

`SU-T/` 是本机原仓库在该提交上的当前工作树快照，包含已经进行的本地源码修改。复制时排除了 `.git`、MFT25 数据目录、`__pycache__` 和模型权重。`source-worktree.patch` 是原 `SU-T` 仓库相对上述基准提交的二进制安全 Git diff，可在干净 checkout 上审阅或应用；不含未跟踪的 Python 字节码。

本归档只把改动复制进新仓库，未改写或清理旁边的原始 `SU-T/` 工作区。源码差异摘要见 `source-changes.txt`。本机 A05 专用脚本与配置另在 `SU-T-Reproduction/`。
