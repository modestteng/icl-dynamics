# AGENTS.md

## Goal

围绕第六篇论文《Strategy Coopetition Explains the Emergence and Transience of In-Context Learning》，复现上下文学习（in-context learning, ICL）的涌现与暂时性，检验论文提出的 strategy coopetition（策略间同时合作与竞争）解释：ICL 与 context-constrained in-weights learning（CIWL）通过共享子电路产生合作，同时存在竞争。

复现内容包括 ICL 与 CIWL 的行为评估及机制干预（第 3–5 节）、最小数学模型（第 6 节），以及使上下文与查询中的对应样本完全匹配以促进 ICL 持续存在的实验（第 7 节）。在复现基础上，定位原方法或解释的明确局限，并在相同实验与评估条件下验证最小改进。论文已有的数据干预属于复现内容，不作为本项目新增改进。

本项目主要论文为 [06_Strategy_Coopetition_Explains_the_Emergence_and_Transience_of_In_Context_Learning.pdf](06_Strategy_Coopetition_Explains_the_Emergence_and_Transience_of_In_Context_Learning.pdf)，讨论论文内容、术语和方法时优先查阅此文件。仓库同时包含作者前两篇相关工作的代码；以 `README.md` 中的论文对应关系为准，不混用不同论文的实验设置或复现结论。

## Project Structure

现有文件与目录：

* `06_Strategy_Coopetition_Explains_the_Emergence_and_Transience_of_In_Context_Learning.pdf` — 第六篇论文原文，本项目研究依据
* `README.md` — 三篇相关工作的概览、代码说明和各论文复现入口
* `setup.md` — 环境安装、依赖版本与 CPU / CUDA 配置说明
* `main.py` — 模型训练、评估与检查点保存主入口
* `main_utils.py` — 命令行参数、数据与模型构建等辅助函数
* `models.py` — 基于 JAX / Equinox 的 Transformer 与序列分类模型，以及中间激活记录和干预接口
* `samplers.py` — 训练与评估序列采样、类别与样本选择、标签重映射
* `opto.py`、`artificial_optogenetics_guide.md` — artificial optogenetics 框架的干预实现与使用说明
* `coopetition_paper_sweep.py`、`coopetition_paper_appendix_sweep.py` — 本项目主要论文的正文与附录实验配置和批量运行入口
* `coopetition_paper_plots.ipynb`、`coopetition_paper_appendix_plots.ipynb` — 本项目主要论文的正文与附录结果分析、绘图入口
* `coopetition_model_solver.py` — 策略合作与竞争的最小数学模型及其优化求解
* `omniglot_dataset.py`、`omni_features_extract.py` — Omniglot 数据加载与特征提取
* `omniglot_resnet18_randomized_order_s0.h5` — 仓库附带的 Omniglot 特征；使用前核对其是否符合目标实验的数据设置
* `visualize_runs.py`、`plot_utils.py` — 训练动态、评估结果与干预结果的可视化工具
* `ih_paper_runs.sh`、`ih_paper_appendix_runs.sh`、`ih_paper_plots.ipynb`、`ih_paper_appendix_plots.ipynb`、`ih_paper_plot_utils.py`、`ih_paper_additional_seeds/` — induction heads 相关前作的实验、绘图与额外随机种子结果
* `simple_model_solver.py` — induction heads 相关前作使用的简化数学模型求解器
* `llama_sweep_example.py`、`fixed_omni_emb_sweep_example.py` — ICL 暂时性相关前作中固定特征实验的运行示例
* `example_cache_explanation.md` — 模型前向计算返回的激活缓存结构与张量形状示例

研究记录与新增产物按需创建：

* `notes.md` — 论文阅读、想法、发现、冻结的实验协议和当前进度
* `experiments/` — 按实验 ID 保存配置快照、命令、环境、日志、结果和代码差异；汇总指标记录在 `experiments/results.tsv`
* `paper/` — 本项目研究正文与最终图表

保留上游代码结构；原始实验配置主要位于上述运行脚本中，新增实验使用独立配置快照和输出目录。

## Naming

* 新增文件和目录使用清晰的小写名称，保留约定名称和上游命名。
* 实验命名为 `exp_001_short_name`，配置、日志和结果使用同一 ID。
* 基线与改进实现保持可区分，使用独立输出目录。

## Rules

* 放弃时仅撤销该实验改动，保留实验记录。
* 先复现基线，再进行改进；始终保持原始基线可运行。
* 一次只改变一个主要想法，在相同评估设置下比较方法。
* 先做小规模验证，再启动昂贵实验；跑通脚本不等于复现论文结果。
* 未经批准，不改变研究问题、基线、模型结构或检查点、数据分布、训练设置、干预对象或评估指标。
* 修改科研代码时使用 `$implement-research-code`，先检索并复用权威实现，优先最小适配。
* 保留上游 JAX / Equinox 实现及本地修改；不擅自替换依赖或改变科学计算语义。
* 复用特征文件、评估数据、模型检查点或缓存前核对模型、数据、配置、随机种子和版本，不仅凭文件名判断。
* 记录成功、失败和不确定实验；禁止虚构结果、引用或证据。
* 训练曲线或内部激活的相关性不能代替干预实验的因果证据；不得为正结果更换评估口径。
* 不修改本机代理配置或操作代理软件。

## 论文语言与术语

* 本项目中的回答、解释、研究记录和正文应沿用所讨论论文的术语、定义、符号和学术表述；讨论本项目主要论文时以其原文为准，涉及相关前作或其他方法时以对应原论文为准。
* 不得随意创造专有名词、方法名称、缩写或概念分类，也不得擅自给论文已有概念改名。论文未使用的表述不得冒充作者的术语。
* 关键术语首次出现时给出论文中的英文原词或缩写及准确的中文译法，后续保持一致；没有通行中文译法时优先保留英文，不强行造词。
* 解释应自然、清楚、准确，可用通俗语言说明原理，但须保留原概念的含义和适用条件；类比和辅助解释应明确作为解释，不包装成新的学术概念。
* 区分论文原有定义、作者主张、实验结果与本项目的推断或改进设想；新增设想应明确标注，不归于原论文。
* 对术语、定义或符号不确定时，先查阅论文原文及必要的官方材料，再作解释；涉及关键定义或有歧义的表述时提供可核查的章节、公式或原文出处，不凭印象编造。

## Experiments

每次记录：`idea → change → result → keep / discard / inconclusive`。
保存代码版本与差异、配置、命令、环境、种子、模型与数据标识、日志和指标。

## Before Stopping

研究工作结束前更新 `notes.md`；有实验变化时同步更新 `experiments/results.tsv`。
目标未完成时继续推进已授权工作；遇到阻塞记录具体缺口和下一步，不把局部成功标成完成。
