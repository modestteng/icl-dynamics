# 单种子主实验协议

用户范围：仅尝试复现 ICL 先涌现后减弱、CIWL 增强的行为动态，不运行机制干预、参数扫描、额外种子、最小数学模型或 12 层扩展。

来源：主要论文第 2 节、图 1b；作者 `coopetition_paper_sweep.py` 中 `default_eval` 和 `main`。核心代码快照及哈希见 `../exp_001_hardware_check/source_manifest.json`。作者原始文件不改写。用户已允许在缺少 `$implement-research-code` 时做独立最小适配。

模型与训练：2 层 attention-only、d_model=64、8 heads、APE、LayerNorm/残差与作者一致；512 维 ResNet18 特征、12800 输出标签；固定类别标签、上下文两个样本标签对、burstiness=1、每类 20 样本；batch=32、Adam、lr=1e-5、weight_decay=0；init_seed=5、train_seed=2、eval_seed=1。不调整模型、精度、batch、学习率或数据噪声来迁就设备。

时间窗口：作者主运行共 64M 序列（2M 次更新），图 1b 展示前 20M 序列。`run_main.py --paper-window` 保存/评价至 20M 边界（625000 次更新），作者循环之后还会执行一批 32 序列，该额外更新不属于保存的 20M 检查点。该窗口只缩短执行前缀，不改变前缀的采样或优化；不保证恰好在此窗口观察到原文趋势。默认不传此参数则保留作者 64M 总长度。

评价：固定保存每集 5000 序列，共 train_eval、ICL、pure IWL、CIWL、Flip。每 100000 训练序列评价，每 50000 序列保存作者格式检查点（模型、Adam 状态、PRNG 状态、实际 iter）。非 32 整除的阈值在下一批次边界执行。图 1b 口径为 in_context_acc；pure IWL 单独使用全标签空间 acc，不混用机会水平。

数据缺口：当前公开特征提取代码的排列问题见 `../exp_001_hardware_check/feature_index_audit.json`。独立 `extract_features.py` 仅提出按论文的固定增强类别/20 样本定义修正索引，复用作者数据类、图像变换、旋转翻转与 ResNet18 IMAGENET1K_V1。用户已明确批准按论文定义修正排列并继续主实验；批准记录见 data_order_approved.txt。作者历史原特征、类别随机选择清单及全套环境锁文件未公开；新产物不声称逐字节一致。

算力：遵循 `$research-compute`，先在 Windows RTX 4070 Laptop/WSL2 上用作者完整模型的计算路径估时。随机占位特征仅用于资源估算，不输出科研准确率。用户最新明确指令覆盖原时间路由门槛：大致估计即可，使用 RTX 4070 推进本次单个主实验，不再反复追求精确估时。继续确认执行路径、保留日志和断点；不转到 Mac 长跑。

状态：已按用户要求在 20M 整数检查点收束，20M 检查点反序列化及有限值核验通过，Adam 更新数 625000；完整作者 64M 参考长度未跑满。已完成本次单种子行为复现目标，最终曲线有 201 点。

持久执行限制：实测这台 Windows 的 WSL 会在 SSH 退出后终止 detached Linux 进程。因此 `launch.sh` 仅用于 Linux 云端；Windows 必须通过计划任务保持 WSL 前台进程，并从第二个 SSH 会话核验任务、进程和日志。资源短测的计划任务材料位于 `../exp_001_hardware_check/run_probe_task.ps1` 和 `register_probe_task.ps1`。
