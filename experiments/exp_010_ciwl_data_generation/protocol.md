# CIWL-only 增强候选数据生成

日期：2026-10-04。实验 ID：exp_010_ciwl_data_generation。

用户授权：按论文构造一批独立数据。仅生成与审核数据；不训练、不评价模型、不改变原基线与原固定评价集。

## 文献依据与原实现

- 第六篇主要论文第 2.2–2.3 节、图 1a、第 5.3 节。CIWL-only 训练使用与 CIWL evaluator 相同的任务构造。
- 作者 `coopetition_paper_sweep.py` 的 `ciwl_only` 配置：`pt_burstiness=[0]`、`pt_no_support=[1]`、`pt_unique_rest=[1]`、`assign_query_label_random=True`，不做 few-shot 随机重映射。
- 直接调用未经修改的 `samplers.get_constant_burst_seq_idxs`、`samplers.get_exemplar_inds`、`samplers.make_data_sampler`。独立编排只记录原采样输出、索引与审核结果，不重写采样算法、不构建或更新模型。复用原项目中已记录的独立最小适配授权；未找到 implement-research-code skill，不修改作者科研源码。

## 本批配置

- 数量：5000 条。用户未指定数量，采用现有单评价集的规模，作为第一批固定增强候选。
- 独立 JAX PRNG seed：20261004；一次调用生成 5000 条，不重放原训练或原评价随机流。
- 类别：原基线的 12800 个训练类，均匀抽样（原 zipf_alpha=0）。样本：每类原 20 个，独立抽取。
- 特征：原基线已批准并核验的 `omniglot_features_reordered.h5`，shape=(12984,20,512)，float32。核对原 feature/config/source/eval SHA256；不重新编码图像。
- 每条存储三个样本特征与三个标签；模型输入对应 `[x1,y1,x2,y2,xq]`，预测目标为最后一个标签。
- 三个样本类别互不相同。查询类不在上下文；查询类的固定标签随机覆盖一个上下文位置的标签；另一个上下文标签保持其固定映射。正确标签恰好出现一次。
- 不强行平衡正确标签位置，不筛选模型表现，不添加标签噪声或损失权重。本批未确定后续训练窗口和增强强度。

## 必要审核

- 捕获原采样器实际索引；与不捕获索引的原 `make_data_sampler` 同 seed 输出逐元素一致。
- 检查每条类别/样本范围、三类互异、查询无同类上下文、正确标签出现一次、未覆盖标签保持固定映射、特征有限并与原特征索引完全对应。
- 按完整三特征向量及三个标签哈希，检查本批内部重复和与原五个固定评价集（共25000条）的精确序列重叠；共享类别/图像不等于完整序列重叠。
- 原固定评价集曾用于反复分析及评分；本次不改变其用途，也不把它称为从未参与研究选择的最终确认集。后续严格确认需另外冻结独立序列。
- 不把“CIWL 可解的数据构造”写成“已证实增强后促进最终 CIWL”。CIWL evaluator 同时可由 pure IWL 解决；学习机制仍需配套评价。

## 产物与记录

- `outputs/ciwl_augmentation.h5`：`ciwl_augmentation` 组内含 float32 `examples`、int32 `labels`、类别/样本索引、正确标签位置、序列 ID。
- `outputs/sequences.jsonl`：全部 5000 条的可读索引与标签记录。
- `outputs/preview.json`：前 8 条真实生成记录，非手编示例。
- `outputs/audit.json`、`outputs/provenance.json`、`outputs/environment.txt`、日志与 SHA256 清单。
- Windows LAPTOP-A9ON60MQ / WSL / 原 JAX 环境。资源足够；预计几分钟内完成，保留持久任务与退出码。

`idea → change → result → keep / discard / inconclusive`：按论文提供面向 CIWL 的固定增强候选 → 复用作者原采样器、独立随机流生成 → 待结构/身份/重叠审核 → 数据生成状态以实测更新；促进 CIWL 的效果未检验。
