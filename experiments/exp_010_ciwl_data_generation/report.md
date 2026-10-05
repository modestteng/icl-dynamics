# CIWL-only 第一批增强候选数据

日期：2026-10-04。状态：Windows 数据生成、审核及本地回传全部完成；结果包和内部 12 项文件 SHA256 均通过核验，详见 `local_delivery_verification.json`。

已按第六篇论文《Strategy Coopetition Explains the Emergence and Transience of In-Context Learning》第 2.3、5.3 节与作者原采样器，生成 5000 条独立随机流的 CIWL-only 训练候选序列。未训练或评价模型，未测量 CIWL 能力提升。

## 构造

从原基线 12800 个训练类中均匀抽取三个不同的类别，各从原 20 个样本中独立抽一个；前两个样本放在上下文，最后一个作为查询。保持类别的固定训练标签，用查询类别的标签随机覆盖一个上下文标签位置，另一个上下文标签保持其原映射。模型输入为 `[x1,y1,x2,y2,xq]`，目标为查询的固定标签。

因此，上下文中没有查询的同类样本，但查询的正确标签恰好出现一次。它对应作者的 `burstiness=0`、`no_support=True`、`unique_rest=True`、`assign_query_label_random=True`，不进行 few-shot 标签重映射。这是作者称为 CIWL-only 的构造；pure IWL 也可解，不能仅凭构造断言模型实际使用 CIWL。

独立 seed 为 20261004。本批采用 5000 条是本项目的首批规模选择，与现有单个评价集规模一致；不是论文新规定的增强训练量。原训练数据与评价文件保持不变。

## 数据与核验

| 项目 | 实测 |
|---|---:|
| 生成序列 | 5000 |
| `examples` 形状 | (5000, 3, 512) |
| `labels` 形状 | (5000, 3) |
| 正确标签在第一个上下文位置 | 2478 |
| 正确标签在第二个上下文位置 | 2522 |
| 不同查询类别数 | 4097 |
| 本批重复的完整序列 | 0 |
| 与原 CIWL 5000 条完整序列重叠 | 0 |
| 与原五组评价共 25000 条完整序列重叠 | 0 |

已检查全部序列的类别和样本范围、查询无同类上下文、三类互异、正确标签只出现一次、未覆盖标签保持固定映射、特征有限和原特征索引匹配。捕获索引的作者 API 调用与未捕获索引的作者原调用在同一 seed 下逐元素完全一致。保存后重新读取 HDF5 逐元素核对通过。

完整序列重叠指三个实际 float32 特征向量及三个标签共同组成的内容；本批与评价集仍可共享类别和原图像。不能将零完整序列重叠表述成类别或图像互斥。

原特征为已批准的 `omniglot_features_reordered.h5`，完整 shape=(12984,20,512)，SHA256=`3df152d4ddddbb022036a01b1664e2118c7ad683eb1ea18148cfc1ea6dcbf41c`。原配置、特征、评价文件与五个科研源码的 SHA256 在前后均核对一致。没有修改上游模型、采样器或训练实现。

实际生成的序列 0：上下文样本 `(class=6330, exemplar=7)` 搭配标签 3281；上下文样本 `(class=6554, exemplar=16)` 搭配标签 6554；查询 `(class=3281, exemplar=7)` 的目标为 3281。这里样本索引从 0 开始，完整序列哈希为 `6aefee1c77f973b7b9e66d994a2ad007d254a1a637489b8434044f713f9e4de3`。这是实际生成记录，不是手编示例。

## 使用与产物

- `outputs/ciwl_augmentation.h5` 的 `ciwl_augmentation` 组保存实际 `examples`/`labels`、`class_idx`、`exemplar_idx`、`item_type`、`correct_context_position` 与 `sequence_id`。正确标签位置按 0/1 编号，标签最后一列是预测目标。
- `outputs/sequences.jsonl` 是全部 5000 条的可读索引、标签和完整序列哈希。它不包含 512 维特征；特征在 HDF5 中。
- `outputs/preview.json` 为前 8 条实际生成样本，非手编例子。
- `outputs/audit.json`、`outputs/provenance.json`、`outputs/environment.txt`、日志、配置、命令与 SHA256 清单均保留。

本批用途为增强训练候选；增强强度、混合比例、训练窗口与训练预算尚未选定。后续阶段性干预应保持这一批序列身份固定，继续采用同一套行为指标，并另行冻结最终确认数据。原评价集已多次用于分析和评分，本次只保证没有把其固定序列复制进训练候选。

## 执行

Windows LAPTOP-A9ON60MQ / RTX 4070 Laptop / WSL Ubuntu-22.04 / Python 3.10.12 / JAX 0.4.26 / Equinox 0.11.4，实际设备 cuda:0。持久任务 `icl-dynamics-exp010-ciwl-generation`，退出码与 LastTaskResult 均为 0。

完整生成任务 2026-10-04 14:40:08–14:40:20 UTC（用户时区 07:40:08–07:40:20），约 12 秒；Python 内部身份核验、采样、保存与审核计时 7.699 秒，不包含 Python 导入和环境导出。前期读文献、编排、连接预检与传输另计。结果包 SHA256=`87af08fc6ad5e30f0eff6a98ad3682ce219946e6b1a0209d3b24c1f5ad980f27`。

回传中途 SSH 端口超时，本地仅有 9477120 字节时拒绝解包。用户报告重启 sshd 后，IPv4/IPv6 的连接曾继续超时；后续连接自动恢复，核对主机身份后使用 SFTP `reget` 从原偏移续传。最终包为 27262521 字节，哈希与远端原包相同；没有重新生成数据，没有执行 Taildrop 发送或修改代理配置。网络恢复与传输不计入实验计算时长。

`idea → change → result → keep / discard / inconclusive`：按论文提供固定 CIWL-only 增强候选 → 复用作者原采样、独立 seed、实际特征和索引输出 → 5000 条全部结构与身份检查通过，完整序列重叠为 0 → keep 数据产物；促进 CIWL 的阶段/终点效果未验证。
