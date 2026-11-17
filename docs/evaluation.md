# RAG 检索与 Rerank 消融评测

> 本文只陈述已取得的 **开发阶段检索评测**。不是生产性能报告，也不等同于端到端答案正确率。

## 1. 数据与实验范围

- 开发数据集：`eval/datasets/company_dev_v1.jsonl`
- 挑战语料：`eval/challenge_corpus/company_v1_draft/`
- 最新实验摘要（位于本地、当前被 `.gitignore` 排除）：`eval/results/ablation/retrieval_ablation_20261117T120408_173107Z.summary.json`
- 评估范围：14 道可回答开发问题；标注事实总数 16。
- 标注状态：`auto_migrated_needs_human_review`，需人工复核。
- 最终 Top-K：3；Dense 与 BM25 候选数各 5；RRF 参数 60；重排模型 `qwen-plus`。
- 对比：Dense、BM25、Hybrid RRF、Hybrid RRF + LLM Rerank。

## 2. 指标与结果

| 指标 | Dense | BM25 | Hybrid RRF | Hybrid + Rerank |
|---|---:|---:|---:|---:|
| Gold Fact Recall@3 | 93.75% | 93.75% | 93.75% | 93.75% |
| Complete Evidence@3 | 92.86% | 92.86% | 92.86% | 92.86% |
| MRR@3 | 0.9286 | 0.8810 | 0.9286 | 1.0000 |
| 延迟 P50 (ms) | 206.35 | 2.08 | 207.45 | 4908.84 |
| 延迟 P95 (ms) | 543.36 | 4.03 | 547.20 | 8061.62 |

**指标解释：**

- **Gold Fact Recall@3**：Top-3 命中的已标注事实数 / 标注事实总数（本实验为 15/16），不是“问题命中率”。
- **Complete Evidence@3**：Top-3 覆盖该问题所需全部事实的可回答问题比例（本实验为 13/14）。
- **MRR@3**：各问题首个相关 Chunk 的倒数排名均值，超过 3 的排名记作未命中。
- **延迟**：本实验统计检索阶段。Hybrid 延迟包含共享候选的 Dense、BM25、RRF 融合，Rerank 方案另外包含模型重排；**不包含 Agent 生成与 Grounding**。

## 3. Rerank 的收益与代价

这组数据中，Rerank 将 MRR@3 从约 0.929 提高至 1.0，但 Top-3 的事实覆盖率和完整证据率没有变化。这表明它改善了相关片段的前排位置，并没有解决被 Top-K 截断的全部证据问题。

同时，检索 P50 从 207.45 ms 增至 4908.84 ms，额外成本较高。因此，系统提供 `rerank_enabled` 开关，便于根据响应时间要求决定是否启用。

本次 Rerank 阶段单独记录的模型统计：**14 次调用**、**10,890 输入 Token**、**2,799 输出 Token**、**13,689 总 Token**，`rerank_fallback_count=0`。这些数字不包括 Dense Embedding 的 Token，也不等于整个 Agent 问答的 Token 用量。

## 4. 实验可复现性与剩余限制

项目保留 `eval/retrieval_ablation.py`、数据集、标签和挑战语料。当前 GitHub 整理方案排除了生成的 `.npz` 索引和 `eval/results/` 历史输出。因此，重新运行前需要先重建挑战语料的索引，并核对评测脚本的 CLI 参数、模型 API Key 与数据路径。

当前没有证据证明全新环境已可一键复现实验；在核对构建命令并完成干净环境验证前，不应对外宣称“开箱即复现”。

**实验局限：** 开发集较小且非盲测；标签待人工复核；只覆盖可回答开发问题；没有统计完整 Agent 生成的端到端时延；Rerank 可能因供应端、负载和模型版本产生波动。

## 5. 面试时如何描述

> 在挑战开发集的 14 道可回答问题中，我对比了 Dense、BM25、Hybrid RRF 和 Hybrid + LLM Rerank。Rerank 将 MRR@3 从约 0.929 提升到 1.0，但完整证据率没有增加，P50 检索时延却由约 207 ms 增至约 4.9 s。因此我保留了可配置的 Rerank 开关，并将证据覆盖不足作为后续优化点。由于数据集较小、部分标签待复核，这些结果仅作为开发阶段实验分析。
