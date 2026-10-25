# AdamW

- 题目：Decoupled Weight Decay Regularization
- 作者：Ilya Loshchilov、Frank Hutter。
- 年份：2017 年 arXiv 初版；发表于 ICLR 2019。
- 类型：研究论文。
- [论文与摘要页面](https://arxiv.org/abs/1711.05101)
- [ICLR 条目链接](https://openreview.net/forum?id=Bkg6RiCqY7)（此次访问出现浏览器验证；内容核对采用 arXiv 页面）。
- 建议引用键：`loshchilov2019adamw`。

## 中文摘要概述

论文说明，对于 Adam 等自适应优化方法，传统 L2 正则化与权重衰减并不等价。作者提出将权重衰减与依据损失梯度计算的优化步骤解耦，并通过实验研究其泛化表现。这一方法通常称为 AdamW。

## 本报告可用的内容

- 在方法或实验设置中首次介绍 AdamW 优化器时引用一次。
- 同时写明实际学习率、weight decay、batch size、epoch 和调度方案，帮助复现。
- 当前回归头实现采用 AdamW；这属于借用的标准训练组件，不是本项目的方法创新。

可采用的表达方向：回归头使用 AdamW 优化，具体超参数在实验设置中列出。

## 不宜直接据此声称

- AdamW 在本房价任务上优于 Adam 或 SGD，因为我们没有相应优化器消融。
- 该论文能证明我们选择的学习率或权重衰减数值最优。

如果正文空间很紧，只需在训练设置的一句话中引用，无须单独写优化器综述。
