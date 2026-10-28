# Bidirectional Encoder Representations from Transformers (BERT)

### 选择语言 | Language

[中文简介](#%E7%AE%80%E4%BB%8B) | [English](#introduction)

### 结果 | Result

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/e6234335-3448-45e4-b61d-3fcbec00cd5f" />

---

## 简介

Bidirectional Encoder Representations from Transformers（简称BERT）由 **Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova 等人** 于2018年提出（NAACL 2019），是自然语言处理领域的里程碑式革新。其核心贡献在于**首次提出深度双向Transformer预训练框架**，通过掩码语言模型（MLM）与下一句预测（NSP）两个自监督任务在大规模无标注文本上进行预训练，再通过轻量微调适配各类下游NLP任务，一举刷新11项NLP基准任务的最优结果，奠定了预训练语言模型的主流技术范式，为后续大语言模型奠定了核心架构基础。

在 BERT 出现之前，预训练语言模型主要分为两类：一类是以GPT为代表的自回归单向语言模型，仅能利用上文信息进行建模；另一类是以ELMo为代表的双向浅层融合模型，通过拼接前后向LSTM特征实现弱双向表征。两类模型均无法实现深度的双向上下文建模，限制了模型的语义理解上限。BERT首次通过掩码语言模型任务，让模型在编码时可以同时看到左右两侧的全部上下文，实现了真正的深度双向语义表征，大幅提升了模型对自然语言的理解与泛化能力。

在 GLUE 通用语言理解基准上，BERT-large 模型取得了 **80.5% 的平均准确率**，全面超越同期所有模型；在 SQuAD 1.1 问答任务上达到 93.2% 的F1值，首次超越人类表现。本复现基于 PyTorch 从零实现完整的 BERT 编码器架构，严格遵循原论文的双向Transformer编码器结构、三类嵌入融合、MLM与NSP双任务预训练设计，采用 **NLTK 电影评论情感分类数据集** 进行下游微调验证，使用分类准确率作为核心评估指标，完整包含数据预处理、模型训练、测试评估、训练曲线可视化全流程。

## 架构

本次复现严格遵循《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》原文的标准架构，整体分为「输入嵌入层」「多层Transformer编码器堆叠」「预训练任务头」「下游任务适配头」四大模块，完整保留原论文的所有核心设计。

- **输入嵌入层**：输入的token索引同时经过词嵌入、分段嵌入、位置嵌入三个独立的嵌入层，三者逐元素相加后经过层归一化与Dropout，得到最终的输入表征。其中词嵌入建模语义信息，分段嵌入区分双句子输入，位置编码注入序列顺序信息，均采用可学习的参数矩阵。
- **Transformer编码器堆叠**：由 L 个完全相同的双向Transformer编码器层堆叠而成。每层包含两个子层：**多头自注意力层**与**位置前馈网络层**，每个子层均采用「子层运算 + 残差连接 + 层归一化」的 Post-LN 结构，保证深层网络的训练稳定性。编码器对序列所有位置进行全局双向信息交互，并行完成全序列的语义编码。
- **多头双向自注意力机制**：将 Query、Key、Value 通过线性投影后拆分为 h 个并行的注意力头，每个头独立计算缩放点积注意力，最后将多个头的输出拼接并做线性投影。与单向自回归模型不同，BERT 的注意力没有因果掩码，每个位置都可以关注序列中所有其他位置的信息，实现真正的双向上下文建模。
- **前馈神经网络（FFN）**：由两层全连接层与 GELU 激活函数组成，先将特征维度升为4倍隐藏层维度，再降回原始维度，对每个位置的特征进行独立的非线性变换，进一步增强模型的语义表达能力。
- **CLS池化层**：编码器全部运算完成后，取出序列首位的 。CLS  令牌对应的特征向量，经过一层全连接与 Tanh 激活，得到整个句子的聚合表征，用于句子级的分类与匹配任务。
- **预训练任务头**：包含两个预训练输出头：掩码语言模型（MLM）头用于预测被掩码位置的原始token，输出层与词嵌入层权重共享以压缩参数量；下一句预测（NSP）头基于CLS表征进行二分类，判断两句话是否为连续上下文。两个任务联合训练，让模型同时掌握词级与句子级的语义理解能力。
- **下游任务适配**：预训练完成后，移除预训练任务头，根据具体下游任务接入对应的输出层。文本分类任务基于CLS池化表征接入线性分类头，通过交叉熵损失进行端到端微调。



**具体网络细节**可以查看我的这篇文章：[BERT 深入浅出]()

**注意**：为降低训练开销、便于快速复现验证，本实现采用轻量化 Mini-BERT 配置（4层堆叠、256维隐藏特征、4头注意力），数据集使用小型电影评论情感分类语料，但模型结构、注意力机制、嵌入设计、训练范式均严格对齐原论文标准。

## 数据集

本次复现使用 **NLTK Movie Reviews 电影评论情感数据集**，是自然语言处理情感分类领域的经典基准数据集。

- **任务类型**：二分类情感分析任务，判断电影评论的情感倾向为正面或负面。
- **数据规模**：共包含2000条人工标注的真实电影评论文本，正面与负面评论各1000条，类别分布均衡。本复现按8:2比例划分为1600条训练集与400条测试集，适合快速验证模型能力。
- **预处理方式**：采用BERT官方WordPiece分词器进行文本分词，统一截断填充到固定序列长度；添加 。CLS  与 [SEP] 特殊令牌构造标准BERT输入格式；构造注意力掩码区分有效token与填充token。
- **评估指标**：采用文本分类领域通用标准指标 **分类准确率**，即预测标签与真实标签一致的样本占总样本的比例，数值范围为 0~100%，数值越高代表分类效果越好。同时配合交叉熵损失监控训练收敛状态，输出训练与测试的损失、准确率双维度曲线。

| 分类准确率 | 效果说明 |
| --- | --- |
| 60 ~ 70% | 具备基础情感识别能力，可区分明显的正负向表达，存在一定误判 |
| 70 ~ 80% | 分类效果良好，大部分评论情感判断准确，泛化性较好 |
| 80 ~ 90% | 高精度分类，接近轻量预训练模型微调水平，鲁棒性较强 |
| 90%+ | 优秀分类性能，接近该数据集上的模型性能上限 |

数据集官方地址：[](https://www.nltk.org/howto/corpus.html#movie-reviews-corpus)[https://www.nltk.org/howto/corpus.html#movie-reviews-corpus](https://www.nltk.org/howto/corpus.html#movie-reviews-corpus)

---

## Introduction

Bidirectional Encoder Representations from Transformers (BERT for short), proposed by **Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova et al.** in 2018 (NAACL 2019), is a milestone innovation in the field of natural language processing. Its core contribution is that it **proposes a deep bidirectional Transformer pre-training framework for the first time**. Through two self-supervised tasks, Masked Language Model (MLM) and Next Sentence Prediction (NSP), it performs pre-training on large-scale unlabeled text, and then adapts to various downstream NLP tasks through lightweight fine-tuning. It has refreshed the state-of-the-art results of 11 NLP benchmark tasks in one fell swoop, established the mainstream technical paradigm of pre-trained language models, and laid the core architectural foundation for subsequent large language models.

Before BERT, pre-trained language models were mainly divided into two categories: one was autoregressive unidirectional language models represented by GPT, which could only use context information for modeling; the other was bidirectional shallow fusion models represented by ELMo, which achieved weak bidirectional representation by splicing forward and backward LSTM features. Neither type of model could achieve deep bidirectional context modeling, which limited the upper limit of the model's semantic understanding. For the first time, BERT enables the model to see all the context on both sides at the same time during encoding through the masked language model task, realizing truly deep bidirectional semantic representation, and greatly improving the model's natural language understanding and generalization ability.

On the GLUE general language understanding benchmark, the BERT-large model achieved **80.5% average accuracy**, comprehensively surpassing all models of the same period; on the SQuAD 1.1 question answering task, it reached an F1 score of 93.2%, surpassing human performance for the first time. This reproduction implements the complete BERT encoder architecture from scratch based on PyTorch, strictly following the original paper's bidirectional Transformer encoder structure, three-type embedding fusion, and MLM+NSP dual-task pre-training design. It is validated on the **NLTK Movie Reviews sentiment classification dataset** and uses classification accuracy as the core evaluation metric, covering the full pipeline of data preprocessing, model training, test evaluation, and training curve visualization.

## Architecture

This implementation faithfully follows the standard architecture from *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, consisting of four main modules: input embedding layer, stacked Transformer encoder, pre-training task heads, and downstream task adapter, with all core designs from the original paper preserved.

- **Input Embedding Layer**: The input token indices go through three independent embedding layers: word embedding, segment embedding, and position embedding. The three are added element-wise, then passed through LayerNorm and Dropout to obtain the final input representation. Word embedding models semantic information, segment embedding distinguishes dual-sentence inputs, and position encoding injects sequence order information, all using learnable parameter matrices.
- **Stacked Transformer Encoder**: Stacked by L identical bidirectional Transformer encoder layers. Each layer contains two sub-layers: a **multi-head self-attention layer** and a **position-wise feed-forward network layer**. Each sub-layer adopts the Post-LN structure of "sub-layer computation + residual connection + LayerNorm" to ensure the training stability of deep networks. The encoder performs global bidirectional information interaction for all positions in the sequence and completes semantic encoding of the full sequence in parallel.
- **Multi-Head Bidirectional Self-Attention**: Query, Key and Value are linearly projected and split into h parallel attention heads. Each head independently computes scaled dot-product attention, and outputs of all heads are finally concatenated and linearly projected. Different from unidirectional autoregressive models, BERT's attention has no causal mask, and each position can attend to all other positions in the sequence, realizing truly bidirectional context modeling.
- **Feed-Forward Network (FFN)**: It consists of two fully connected layers and GELU activation function. The feature dimension is first upscaled to 4 times the hidden size and then reduced back to the original dimension, performing independent nonlinear transformation on features of each position to further enhance the semantic expression ability of the model.
- **CLS Pooler**: After all encoder layers are computed, the feature vector corresponding to the 。CLS  token at the first position of the sequence is extracted, passed through a fully connected layer and Tanh activation, to obtain the aggregated representation of the whole sentence for sentence-level classification and matching tasks.
- **Pre-training Task Heads**: Two pre-training output heads are included: the Masked Language Model (MLM) head is used to predict the original token at masked positions, and its output layer shares weights with the word embedding layer to compress parameter quantity; the Next Sentence Prediction (NSP) head performs binary classification based on the CLS representation to judge whether two sentences are consecutive context. The two tasks are trained jointly to enable the model to master both word-level and sentence-level semantic understanding capabilities.
- **Downstream Task Adapter**: After pre-training, the pre-training task heads are removed, and corresponding output layers are connected according to specific downstream tasks. For text classification tasks, a linear classification head is connected based on the CLS pooled representation, and end-to-end fine-tuning is performed through cross-entropy loss.

**Note**: To reduce training cost and facilitate quick reproduction and verification, this implementation uses a lightweight Mini-BERT configuration (4 stacked layers, 256-dimensional hidden features, 4 attention heads) and a small movie review sentiment classification corpus. However, model structure, attention mechanism, embedding design, and training paradigm are all strictly aligned with the original paper standard.

## Dataset

This reproduction uses the **NLTK Movie Reviews dataset**, a classic benchmark dataset in the field of NLP sentiment classification.

- **Task**: Binary sentiment classification task, judging whether the sentiment tendency of a movie review is positive or negative.
- **Data Scale**: Contains 2000 manually annotated real movie review texts, with 1000 positive and 1000 negative reviews, with balanced category distribution. This reproduction divides the data into 1600 training samples and 400 test samples in an 8:2 ratio, which is suitable for quickly verifying model capability.
- **Preprocessing**: The official BERT WordPiece tokenizer is used for text tokenization, and all sequences are truncated and padded to a fixed length. Special tokens including 。CLS  and [SEP] are added to construct standard BERT input format; attention masks are constructed to distinguish valid tokens from padding tokens.
- **Evaluation Metric**: **Classification Accuracy**, the universal standard metric for text classification. It refers to the proportion of samples whose predicted label is consistent with the ground truth label, with a value range of 0~100%. Higher values indicate better classification performance. Cross-entropy loss is also used to monitor training convergence, and dual-dimensional training curves of loss and accuracy for both training and test sets are output.

Official dataset page: [](https://www.nltk.org/howto/corpus.html#movie-reviews-corpus)[https://www.nltk.org/howto/corpus.html#movie-reviews-corpus](https://www.nltk.org/howto/corpus.html#movie-reviews-corpus)

---

## 原文章 | Original article

Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding[C]//Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019: 4171-4186.


