# AGENTS.md

## 研究目标

主依据是第六篇《Strategy Coopetition Explains the Emergence and Transience of In-Context Learning》；先复现上下文学习（ICL）与上下文约束的权重内学习（CIWL）的行为和机制。原论文的数据干预、层替换与钳制属于复现。

当前创新候选：借鉴《Influence Dynamics》的贝叶斯影响函数（BIF）与窗口增强，检验同一批数据的阶段性影响是否经第一层相关计算传递到策略。重点是“数据增强 × 电路钳制”的跨阶段因果检验，形成“数据 → 电路 → 策略”的证据；机制与新颖性尚待验证。

## 项目入口

| 入口 | 用途 |
|---|---|
| `06_Strategy_Coopetition_Explains_the_Emergence_and_Transience_of_In_Context_Learning.pdf` | 主论文，术语和实验定义以此为准 |
| `01_Influence Dynamics and Stagewise Data Attribution.pdf` | BIF 与阶段性增强的方法参考 |
| `notes.md` | 当前目标、关键发现和下一步；详细旧记录在 `experiments/history/` |
| `experiments/results.tsv` | 唯一实验总表，包含来源、状态、指标及实验目录 |
| `experiments/exp_*/` | 每项实验的协议、命令、日志、审核和结果 |
| `README.md`、`setup.md` | 上游三篇论文对应关系、运行说明与环境 |
| `main.py`、`main_utils.py`、`models.py`、`samplers.py` | 原 JAX／Equinox 训练、模型与采样实现 |
| `opto.py`、`artificial_optogenetics_guide.md` | 干预实现与说明 |
| `coopetition_paper*_sweep.py`、`coopetition_paper*_plots.ipynb`、`coopetition_model_solver.py` | 第六篇运行、分析与数学模型；前作入口见 README |
| `devinterp/` | 独立 Fork 的官方 BIF／SGLD 参考库，原实现使用 PyTorch |

## 工作规则

- 先基线、后新增假设；未经本轮授权，不改变研究问题、数据分布、模型／检查点、训练设置、干预对象或指标。
- 保留上游结构和原 JAX／Equinox 科学语义。修改科研实现时使用 `$implement-research-code`，先检索权威实现，优先最小适配；历史记录不能自行充当技能豁免。
- 复用数据、特征、模型或缓存前核对配置、种子、版本与身份。根目录附带特征不能仅凭文件名替代已批准的数据版本。
- 整个实验在 Mac 可运行、无 CUDA 依赖且预计 ≤20 秒时直接本机执行；超过20秒、时长未知或需要CUDA时，先读取 `$research-compute`，默认 Windows。预计 ≥3 小时或资源不足，先准备云端材料并告知用户；不自行开云或静默改回 Mac。
- 一次改变一个主要想法，先小规模验证。相关曲线、平均注意力和代理评分不能代替因果干预；钳制交互须排除整体损伤及地板／天花板效应。
- 沿用论文术语，区分作者结论、项目假设和实测结果；不预设数据增强或电路钳制会成功，不宣称未经核查的首创。
- 每项实验记录 `idea → change → result → keep / discard / inconclusive`，保存版本／差异、配置、命令、环境、种子、数据／模型标识、日志与指标；失败和放弃记录也保留。
- 记录集中在 `notes.md`、`results.tsv` 和各实验目录，不另建重复索引。旧说明归档；只删除确认重复的包或缓存，保留原始数据、检查点、冻结协议和实验日志。
- 结束前更新 `notes.md`；有实际实验变化时同步 `results.tsv`，不改写历史指标。目标未完成须写明缺口和下一步。
- 不修改本机代理配置或操作代理软件；不在未经要求时安装依赖或启动实验。
