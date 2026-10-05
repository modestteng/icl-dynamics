# 原训练序列的 ICL / CIWL 影响分析：第一轮协议

## 论文与作者实现

依据本地第六篇论文第 2.2–2.3、4.1–4.2、5.1–5.5、6、7 节，以及作者 `coopetition_paper_sweep.py`、`samplers.py`、`main.py`、`opto.py`。主数据同时允许两种策略；不能从“可解”直接推导永久二分类。

ICL：L1 previous-token heads 将上下文样本信息传到标签位置，L2 利用查询与相关样本的匹配并复制标签。CIWL：查询与训练中的静态标签关联，借助 L1 attending-to-self、L2 skip-trigram-copiers 从上下文复制正确标签。L2 共享子电路产生合作，L1 动态产生竞争。第 5.2–5.3 节的权重移植提供合作证据，第 7 节的 exact matching exemplars 提供数据干预证据；这些均不等价于逐条训练贡献标签。

作者 ICL evaluator 使用 `fs_relabel='01'`；CIWL evaluator 使用 no-support + 随机将查询标签放入一个上下文位置（`iwl_copy_avail`）。同时报告 pure IWL 全标签准确率，避免把 IWL 混称 CIWL；Flip 测量策略主导关系。保持原 CE 和 `in_context_acc`，不改指标。

## 冻结范围与选择

- 原基线：20M，init=5/train=2/eval=1，batch32，float32，原 feature manifest 与评价文件。
- 候选集合：原序列 ID [0, 10000000)，试验集合：[0, 1000000)。确定性前缀，无重新抽样。前缀选择不是均匀抽取整个训练历程；原数据分布平稳。
- 重放复用原采样器与 JAX PRNG 拆分顺序，并与 1M 检查点 RNG 核验。核验失败则停止，不报告为原训练序列。
- 模型状态：2M（涌现前）、4.4M（本 seed ICL 峰值）、8M（转变阶段）。基线文件只读；三个 checkpoint 的选择是本项目新增分析，非作者预设。
- 原固定评估集每组5000条，前2500条用于评分方向，后2500条仅用于对照验证。该划分属于新分析，最终不得将半集指标冒充原5000条基线指标。

## 候选评分与拒判

新增分析参考 TracIn 的一阶梯度思想（https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html）。单个检查点 k 的分数：

`s_E(z,k) = <grad CE_train(z,k), grad mean CE_E(k)> / ||grad mean CE_E(k)||`。

正值表示一个无穷小 SGD 更新预计降低该评价损失。使用 JVP 计算同一个内积，避免保存每条完整梯度。这是归一化的一阶候选评分，不是原论文方法，也不是恢复原 Adam 每次更新的精确因果贡献；实际 Adam 影响须由对照训练验证。不把不同 checkpoint 分数强行平均；保留阶段差异。

`relative_icl = 0.5 + 0.5*(s_icl-s_ciwl)/(abs(s_icl)+abs(s_ciwl)+eps)`，范围[0,1]，只是相对倾向，不是概率。分差比例绝对值<=0.1、胜方分数<=0或两项接近0时拒判；其余二分。额外记录两项均正的共同受益标记和 exact matching 字段；保留所有拒判行。

## 数值核验与组级效果验证

- 在真实序列上验证 JVP 分数与显式 reverse-mode 梯度内积一致、批量与单例一致、评分有限、数据规则正确。
- 首次默认GPU矩阵精度下的一致性检查失败（相对差异约1.5%）；保留失败日志。评分改为显式 `jax_default_matmul_precision=highest` 以检查内部低精度矩阵计算的影响，参数/特征仍float32，不放宽断言。实际Adam验证及其前后评价恢复原基线默认矩阵精度。该计算精度适配只影响新增评分，不修改作者源码或基线。
- 4.4M处，从两项胜方为正且分差最明显的各前10%候选中抽取4096条；随机组从同一1M集合抽取4096条。记录ID、类别分布、重复/重叠统计。两次选择seed=103、104，组间同seed保持模型PRNG一致。
- 每组从相同4.4M模型、Adam状态、训练模型PRNG出发，调用作者 `train_step`，原batch32/lr/float32，训练4096序列。每组保存一个终点checkpoint，不逐条保存模型。
- 独立后2500条评估 ICL、CIWL、Flip、pure IWL和train_eval。门槛：两次选择中，ICL组相对随机组的ICL损失下降都更大，CIWL组相对随机组的CIWL损失下降都更大；同时报告准确率、不确定性和副作用。不以大模型意见替代效果证据。这是探索性组级验证，不证明每一行的精确因果贡献。
- 未通过门槛则不扩大10M，不改变门槛追求正结果。导出极端、中间、共同受益、阶段冲突样本供本对话大模型审核；任何脚本调节需独立版本和后续留出验证。

## 执行与产物

Windows RTX4070 / WSL 原JAX环境。独立目录保存源码哈希、checkpoint和评价哈希、配置、日志、完成行数、评分HDF5、组级验证、审核样本。原作者源码与20M基线保留。沿用先前缺少 implement-research-code 时允许独立最小适配的授权。
