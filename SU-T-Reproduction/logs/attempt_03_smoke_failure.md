# 第三次验证运行记录

- 时间：2026-10-10 14:29（本地时区）。
- 运行范围：BT-001 `val_half` 的前 10 帧，用于短程跟踪链路检查。
- 已完成：检测、ReID 特征提取、UKF 轨迹更新；生成了 10 帧的 `BT-001.txt` 跟踪结果。
- 停止位置：评测收尾时导入仓库可选的 `yolox._C` COCO 加速扩展失败。该 C++ 扩展未在当前 Windows 环境编译。
- 处理：在 `MOTEvaluator` 中加入 pycocotools 标准 COCOeval 回退。此处只计算附带的检测 AP 汇总，不改变预测框或跟踪结果。
- 完整控制台 traceback：`sut_reid_smoke10_console.log`。
