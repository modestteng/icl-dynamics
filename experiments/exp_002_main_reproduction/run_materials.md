# 单次主实验的执行材料

当前只复现一个 seed 的 2 层主实验。作者核心代码位于 `../exp_001_hardware_check/source.tar.gz` 的 `source/`，来源与哈希在 `source_manifest.json`。尚无主实验训练结果；不运行改进、干预、额外 seed 或 12 层扩展。

用户已确认按论文定义修正特征排列，并明确使用 Windows RTX 4070、大致估时即可。正式启动仍须先使完整计算路径可运行。短测与原始历史产物之间没有替代关系，`*.timing.eqx` 禁止用于正式训练的初始化/恢复。

Linux 云端只在用户提供服务器后使用。建议先提供 Ubuntu 22.04、Python 3.10、CUDA 可用的单张 NVIDIA GPU，主机内存以作者作业配置 32 GB 为参考；最低显存及最终吞吐须以实测为准，不把 H100 80 GB 的 12 层配置作为本实验硬下限。环境安装脚本是 `cloud_setup.sh`。不自动购买或开通。

以下路径是运行模板，由执行环境设置，不是已经生成的数据。将作者源码解包到 `SOURCE`、新增脚本放在 `ADAPTER`、本次实验根目录设为 `ROOT`，解释器为 `PYTHON`：

```bash
# 仅做编码估时；不生成完整科研特征
"$PYTHON" "$ADAPTER/extract_features.py" --source "$SOURCE" --output "$ROOT/features" --benchmark-only

# 仅在用户明确同意排列修正后生成完整特征
"$PYTHON" "$ADAPTER/extract_features.py" --source "$SOURCE" --output "$ROOT/features" --data-order-approved

# 生成并固定作者的五个评价集（四个策略评价 + train_eval）
"$PYTHON" "$ADAPTER/run_main.py" --source "$SOURCE" --root "$ROOT" --stage eval

# 图 1b 的 20M 序列窗口；只有算力路由确认后才启动
"$PYTHON" "$ADAPTER/run_main.py" --source "$SOURCE" --root "$ROOT" --stage train --paper-window

# 绘图：不得将占位计时数据传入
"$PYTHON" "$ADAPTER/plot_results.py" "$ROOT/runs/main/log.h5" --output "$ROOT/figures"
```

若需要作者完整的 64M 总长度，则去掉 `--paper-window`，该执行规模必须重新合并估时；本次现象窗口与完整 64M 不混称。恢复示例是在同一 root 下使用 `--resume-checkpoint "$ROOT/runs/main/checkpoints/实际文件名.eqx"`，恢复模型、优化器和 PRNG。恢复会写入独立 `main_resume_检查点名` 目录，以保留原日志；合并绘图前核对边界迭代、重复评价和各段配置。

Windows 不能用 `launch.sh` 直接脱离 SSH 后让 WSL 后台长跑；实测进程会被终止。必须复用已验证的 Windows 计划任务持有 WSL 前台会话，记录实际任务状态、PID、日志和退出码。资源短测任务名为 `icl-dynamics-exp001-probe`，它只安装环境并做一次短测，不启动本主实验。

完成后取回 config/argv、完整 log.h5、评价集哈希、必要检查点、环境、stdout/stderr、实际运行时间与图表。特征和检查点通过哈希与配置核验，不能仅凭文件名复用。

本次已启动 Windows 计划任务 `icl-dynamics-exp002-main`。正式流水线使用作者原 main 的 64M 总长度，而上面的 `--paper-window` 只是可选前缀模板；本次不运行 CIWL-only 或后续改进。初次安装曾中断，已保留记录并恢复；最新独立会话核验为 extract_features，官方数据已下载、ResNet18 权重下载中。尚无正式主训练指标。Windows 运行材料为 `run_pipeline_task.ps1`、`register_pipeline_task.ps1`、`run_pipeline_wsl.sh`。查看 Linux 的 `pipeline.stage`、`pipeline.exit_code` 及 logs，各阶段完成会自动进入下一阶段；训练失败时先查日志，再从已核验的检查点恢复。

2026-10-03 00:52 PDT：任务因 Windows 控制台中断类型退出而停止过，已核验并从 runs/main/checkpoints/00002950016.eqx 恢复。当前主任务 action 为 run_resume_task.ps1，Linux 脚本 resume_main_wsl.sh；实际续跑目录 runs/main_resume_00002950016，日志 logs/train_resume_00002950016.log。查询 read_progress.py 时需传 --run main_resume_00002950016。最终图表会使用 --prefix-log 合并原段和续跑段，原段日志不覆盖。
