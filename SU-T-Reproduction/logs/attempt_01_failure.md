# 首次验证运行失败记录

- 时间：2026-10-10 14:25（本地时区）。
- 阶段：主模型和 ReID 权重均加载成功，处理验证集第一帧时停止。
- 原因：跟踪器将 `np.float` 用作数组类型；当前主环境 NumPy 2.2 已移除该别名。
- 处理：将运行路径中旧的 NumPy `np.float`/`np.bool` 别名替换为等价的 Python `float`/`bool`，不改跟踪参数和算法逻辑。
- 证据：完整 traceback 保存在 `su_t_reid_valhalf_console.log`；入口的 Loguru 捕获使进程码误报为 0，因此该次明确记为失败。
- GPU：失败时 RTX 5060 Laptop GPU 显存仍有约 6 GiB 空闲，未发生显存不足。
